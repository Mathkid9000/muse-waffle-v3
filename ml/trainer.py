"""Shared training utilities for the MMT-BERT compound-event models.

The training loop structure is unchanged from the previous system (persisted
val split, AdamW + warmup/cosine, AMP, gradient accumulation, checkpoint every
epoch); what changed is the example shape — ``(x [T,7], y [T,7], loss_mask
[T])`` — and the loss, which is now a per-field masked cross-entropy summed
over the 7 event fields, with each field additionally masked by
``FIELD_LOSS_MASK[y_type]`` so a field an event type does not use is never
scored.
"""

from __future__ import annotations
import json
import math
import random
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ml.events import (
    F_TYPE, N_FIELDS, PAD_T, FIELD_LOSS_MASK, MAX_BARS,
    special_event, sos_event, eos_event, transpose_events,
)
from ml.tokenizer import flatten_measure_events
from ml.model import model_paths
from ml.preprocess import build_or_open_cache, _path_key, DEFAULT_CACHE_DIR

VAL_RATIO = 0.05

# Transposition augmentation range per the MMT-BERT paper: s ~ U(-5, 6).
TRANSPOSE_LOW, TRANSPOSE_HIGH = -5, 6


def _make_split(keys: list[str]) -> tuple[list[str], list[str]]:
    """Shuffle keys and split off VAL_RATIO for validation. Returns (val, train)."""
    shuffled = list(keys)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return shuffled[-n_val:], shuffled[:-n_val]


def compound_ce_loss(
    logits: list[torch.Tensor],
    y: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-field masked cross-entropy, summed over the 7 fields.

    Args:
        logits: 7 tensors [B, T, V_k] from the per-field heads.
        y: [B, T, 7] target event rows.
        loss_mask: [B, T] float, 1 at scored positions (0 at padding and, for
            FIM examples, everywhere before the target region).
    """
    mask_table = torch.tensor(FIELD_LOSS_MASK, device=y.device, dtype=loss_mask.dtype)
    field_mask = mask_table[y[..., F_TYPE]]          # [B, T, 7]
    total = torch.zeros((), device=y.device)
    for k in range(N_FIELDS):
        ce = nn.functional.cross_entropy(
            logits[k].reshape(-1, logits[k].size(-1)),
            y[..., k].reshape(-1),
            reduction="none",
        )
        m = (loss_mask * field_mask[..., k]).reshape(-1)
        total = total + (ce * m).sum() / (m.sum() + 1e-8)
    return total


def create_standard_example(
    header_events: list[list[int]],
    measure_events: dict[int, list[list[int]]],
    max_bars: int = MAX_BARS,
) -> list[list[int]]:
    """[SOS] + header + all measures (bars renumbered 0..N-1) + [EOS].

    Scores longer than ``max_bars`` measures contribute a random contiguous
    window per call, so the bar-index cap doubles as augmentation.
    """
    nums = sorted(measure_events.keys())
    if len(nums) > max_bars:
        start = random.randint(0, len(nums) - max_bars)
        nums = nums[start:start + max_bars]
    flat = flatten_measure_events(measure_events, nums, start_bar=0)
    return [sos_event()] + list(header_events) + flat + [eos_event()]


class MusicDataset(Dataset):
    """Produces standard (and optionally FIM) compound-event training examples.

    Pass fim_example_fn to enable fill-in-the-middle examples when fim_ratio > 0.
    Gap position and size are re-randomized on every __getitem__ call so the
    model sees each song with a different gap each epoch.
    """

    def __init__(
        self,
        sequences,
        max_len: int = 1024,
        fim_ratio: float = 0.0,
        transpose: bool = True,
        fim_example_fn=None,
    ):
        """
        Args:
            sequences: either an in-memory list of
                (header_events, measure_events, collected, n_parts) tuples, OR a
                map-style provider (CachedMeasureStore / SubsetView) yielding the
                same 4-tuples lazily.
            max_len: maximum sequence length in events.
            fim_ratio: proportion of examples that use the FIM format.
            transpose: apply s ~ U(-5, 6) semitone augmentation.
            fim_example_fn: Callable(header, measures, collected, n_parts,
                max_len) -> (seq, loss_start) | None
        """
        self.max_len = max_len
        self.fim_ratio = fim_ratio
        self.transpose = transpose
        self.fim_example_fn = fim_example_fn
        if isinstance(sequences, list):
            self.sequences = [t for t in sequences if len(t[1]) >= 3]
        else:
            # Cache-backed provider: already length-filtered at build time.
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        header_events, measure_events, collected, n_parts = self.sequences[idx]

        if self.fim_ratio > 0 and self.fim_example_fn is not None \
                and random.random() < self.fim_ratio:
            result = self.fim_example_fn(
                header_events, measure_events, collected, n_parts,
                max_len=self.max_len,
            )
            if result is not None:
                seq, loss_start = result
            else:
                seq = create_standard_example(header_events, measure_events)
                loss_start = 1
        else:
            seq = create_standard_example(header_events, measure_events)
            loss_start = 1

        if self.transpose:
            shift = random.randint(TRANSPOSE_LOW, TRANSPOSE_HIGH)
            seq = transpose_events(seq, shift)

        if len(seq) > self.max_len + 1:
            seq = seq[:self.max_len + 1]

        x = seq[:-1]
        y = seq[1:]

        loss_mask = [0.0] * len(y)
        for i in range(max(loss_start - 1, 0), len(y)):
            loss_mask[i] = 1.0

        pad_len = self.max_len - len(x)
        pad_row = special_event(PAD_T)
        x         = x         + [pad_row] * pad_len
        y         = y         + [pad_row] * pad_len
        loss_mask = loss_mask + [0.0]     * pad_len

        return (
            torch.tensor(x,         dtype=torch.long),
            torch.tensor(y,         dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def _train_loop(
    name: str,
    model,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float, float], None],
    stop_flag: list[bool],
    fim_ratio: float = 0.0,
    fim_example_fn=None,
    resume_optimizer: bool = False,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    warmup_ratio: float = 0.05,
    cache_dir=None,
    cache_workers: int = 1,
):
    """Core training loop shared by the generator trainers.

    Handles val-split persistence, the token cache, optimizer setup, the epoch
    loop, checkpointing, and the on_epoch callback.  Saves checkpoints to
    model_paths(name)["weights"] / ["optimizer"] every epoch.
    """
    paths = model_paths(name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    # Get cache, returns a store of (header, measure_tokens, collected, n_parts)
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    log(f"Building/opening token cache for {name} ({len(data_paths)} scores)...")
    store = build_or_open_cache(data_paths, cache_dir=cache_dir, n_workers=cache_workers, log=log)

    # Gets abs paths
    all_keys = [k for k in (_path_key(p) for p in data_paths) if store.has(k)]
    if not all_keys:
        raise RuntimeError("No sequences in token cache — run 'Download Dataset' first.")

    val_split_file = paths["val_split"]
    if resume_optimizer and val_split_file.exists():
        saved_val  = set(json.loads(val_split_file.read_text()))
        val_keys   = [k for k in all_keys if k in saved_val]
        train_keys = [k for k in all_keys if k not in saved_val]
        if not val_keys:  # old-format or mismatched split — recreate
            val_keys, train_keys = _make_split(all_keys)
            val_split_file.write_text(json.dumps(val_keys))
        log(f"Loaded preserved val split: {len(val_keys)} val / {len(train_keys)} train.")
    else:
        val_keys, train_keys = _make_split(all_keys)
        val_split_file.write_text(json.dumps(val_keys))
        log(f"Created val split: {len(val_keys)} val / {len(train_keys)} train.")

    max_len   = model.cfg["max_seq_len"]
    train_set = MusicDataset(store.subset_view(train_keys), max_len, fim_ratio, fim_example_fn=fim_example_fn)
    val_set   = MusicDataset(store.subset_view(val_keys),   max_len, fim_ratio, fim_example_fn=fim_example_fn)
    log(f"Dataset: {len(train_set)} train / {len(val_set)} val (max_len={max_len}, fim_ratio={fim_ratio})")

    loader     = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # LR schedule: linear warmup then cosine decay
    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-5 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if paths["optimizer"].exists() and resume_optimizer:
        optimizer.load_state_dict(torch.load(paths["optimizer"], map_location=device, weights_only=True))
        log("Resumed optimizer state from previous session.")
    model.to(device)

    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    amp_dtype = torch.float16

    if amp_enabled:
        log("Mixed-precision training enabled (float16).")
    if accumulation_steps > 1:
        log(f"Gradient accumulation: {accumulation_steps} steps (effective batch = {batch_size * accumulation_steps}).")

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("Training stopped by user.")
            break

        model.train()
        epoch_loss = 0.0
        batches    = 0
        optimizer.zero_grad()

        for step, (x, y, loss_mask) in enumerate(tqdm(loader, desc=f"{name.capitalize()} Epoch {epoch}")):
            if stop_flag[0]:
                break

            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(x)
                loss = compound_ce_loss(logits, y, loss_mask)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * accumulation_steps
            batches    += 1

        avg_loss = epoch_loss / max(batches, 1)

        model.eval()
        val_loss    = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, loss_mask in val_loader:
                x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(x)
                    val_loss += compound_ce_loss(logits, y, loss_mask).item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        on_epoch(epoch, avg_loss, avg_val_loss)

        torch.save(model.state_dict(), paths["weights"])
        torch.save(optimizer.state_dict(), paths["optimizer"])
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  ({name} checkpoint saved)")

    log(f"{name.capitalize()} training complete.")
