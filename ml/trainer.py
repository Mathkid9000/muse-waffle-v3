"""Shared training utilities for MusicTransformer models."""

from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from ml.tokenizer import (
    score_to_tokens_by_measure, score_to_header_tokens,
    text_to_tokens,
    _collect_events_by_measure, _emit_measure_tokens,
    MAX_REAL_PARTS,
    BOS, EOS, PAD,
    NOTE_OFF, KEY_OFF,
    CSYM_ROOT_OFF, CSYM_BASS_OFF,
)
from ml.model import MusicTransformer, model_paths, WEIGHTS_FILE
from ml.preprocess import build_or_open_cache, _path_key, DEFAULT_CACHE_DIR

_NOTE_END = NOTE_OFF + 128
_KEY_END  = KEY_OFF  + 13

VAL_RATIO = 0.05


def _make_split(keys: list[str]) -> tuple[list[str], list[str]]:
    """Shuffle keys and split off VAL_RATIO for validation. Returns (val, train)."""
    shuffled = list(keys)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return shuffled[-n_val:], shuffled[:-n_val]


def _transpose_tokens(tokens: list[int], semitones: int) -> list[int]:
    """Shift NOTE and KEY_SIG tokens by semitones; all other tokens are unchanged.

    NOTE tokens are clamped to [0, 127]. KEY_SIG tokens are shifted via the
    circle of fifths and wrapped to the representable [-6, +6] sharps range.
    """
    if semitones == 0:
        return tokens
    result = []
    for t in tokens:
        if NOTE_OFF <= t < _NOTE_END:
            new_pitch = max(0, min(127, t - NOTE_OFF + semitones))
            result.append(NOTE_OFF + new_pitch)
        elif KEY_OFF <= t < _KEY_END:
            old_sharps = (t - KEY_OFF) - 6
            new_sharps = ((old_sharps + semitones * 7) + 6) % 12 - 6
            result.append(KEY_OFF + new_sharps + 6)
        elif CSYM_ROOT_OFF <= t < CSYM_ROOT_OFF + 12:
            result.append(CSYM_ROOT_OFF + (t - CSYM_ROOT_OFF + semitones) % 12)
        elif CSYM_BASS_OFF <= t < CSYM_BASS_OFF + 12:
            result.append(CSYM_BASS_OFF + (t - CSYM_BASS_OFF + semitones) % 12)
        else:
            result.append(t)
    return result


def create_standard_example(
    header_tokens: list[int],
    measure_tokens: dict[int, list[int]],
) -> list[int]:
    """Create a standard next-token prediction example (no FIM)."""
    tokens = [BOS] + header_tokens
    for m_num in sorted(measure_tokens.keys()):
        tokens.extend(measure_tokens[m_num])
    tokens.append(EOS)
    return tokens


class MusicDataset(Dataset):
    """Dataset that produces standard (and optionally FIM) training examples.

    Pass fim_example_fn to enable fill-in-the-middle examples when fim_ratio > 0.
    When fim_example_fn is None or fim_ratio is 0.0, all examples are standard.

    Gap position and size are re-randomized on every __getitem__ call so the
    model sees each song with a different gap each epoch.
    """

    def __init__(
        self,
        sequences: list[tuple],
        max_len: int = 512,
        fim_ratio: float = 0.0,
        transpose_range: int = 6,
        fim_example_fn=None,
    ):
        """
        Args:
            sequences: either an in-memory list of
                (header_tokens, measure_tokens, collected, n_parts) tuples (legacy
                2-tuples are also accepted), OR a map-style provider (e.g. a
                CachedMeasureStore / SubsetView) yielding the same 4-tuples lazily.
            max_len: Maximum sequence length (tokens)
            fim_ratio: Proportion of examples that use FIM format
            transpose_range: Max semitone shift for pitch augmentation (0 to disable)
            fim_example_fn: Callable(header, measures, collected, n_parts, max_len)
                -> (seq, loss_start) | None
        """
        self.max_len = max_len
        self.fim_ratio = fim_ratio
        self.transpose_range = transpose_range
        self.fim_example_fn = fim_example_fn
        if isinstance(sequences, list):
            # Legacy in-memory path: normalise to 4-tuples (tolerate 2-tuples) and
            # drop scores with fewer than 3 measures.
            norm = []
            for seq in sequences:
                if len(seq) == 2:
                    h, m = seq
                    norm.append((h, m, None, 1))
                else:
                    norm.append(seq)
            self.sequences = [t for t in norm if len(t[1]) >= 3]
        else:
            # Cache-backed provider: already 4-tuples, already length-filtered at
            # cache-build time. Accessed lazily via __getitem__.
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        header_tokens, measure_tokens, collected, n_parts = self.sequences[idx]

        if self.fim_ratio > 0 and self.fim_example_fn is not None and random.random() < self.fim_ratio:
            result = self.fim_example_fn(
                header_tokens, measure_tokens, collected, n_parts, max_len=self.max_len,
            )
            if result is not None:
                seq, loss_start = result
            else:
                seq = create_standard_example(header_tokens, measure_tokens)
                loss_start = 1
        else:
            seq = create_standard_example(header_tokens, measure_tokens)
            loss_start = 1

        if self.transpose_range:
            shift = random.randint(-self.transpose_range, self.transpose_range)
            seq = _transpose_tokens(seq, shift)

        if len(seq) > self.max_len + 1:
            seq = seq[:self.max_len + 1]

        x = seq[:-1]
        y = seq[1:]

        loss_mask = [0] * len(y)
        for i in range(loss_start - 1, len(y)):
            loss_mask[i] = 1

        pad_len = self.max_len - len(x)
        x         = x         + [PAD] * pad_len
        y         = y         + [PAD] * pad_len
        loss_mask = loss_mask + [0]   * pad_len

        return (
            torch.tensor(x,         dtype=torch.long),
            torch.tensor(y,         dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def _load_measure_sequences(
    paths: list[str],
    log: Callable[[str], None],
    max_sequences: int = 0,
) -> list[tuple[list[int], dict[int, list[int]], tuple, int]]:
    """Load scores into (header_tokens, measure_tokens, collected, n_parts) tuples.

    `collected` is the raw output of `_collect_events_by_measure` (events plus
    global-symbol dicts); it lets the FIM example generator re-emit token streams
    for arbitrary staff subsets (partial-staff infill) without re-parsing music21.
    `n_parts` is the number of real parts (staves) in the score.
    """
    import music21 as m21

    sequences = []
    for path in tqdm(paths, desc="Parsing scores..."):
        if max_sequences != 0 and len(sequences) >= max_sequences:
            break
        # try:
        score = m21.converter.parse(path)
        header = score_to_header_tokens(score)
        collected = _collect_events_by_measure(score)
        events_by_measure = collected[0]
        if events_by_measure and len(events_by_measure) >= 3:
            measure_tokens = _emit_measure_tokens(*collected)
            n_parts = len(list(score.parts)[:MAX_REAL_PARTS])
            sequences.append((header, measure_tokens, collected, n_parts))
        # except Exception as e:
        #     log(f"  Skip {Path(path).name}: {e}")

    return sequences


def _train_loop(
    name: str,
    model: MusicTransformer,
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
    """Core training loop shared by all model trainers.

    Handles val split persistence, sequence loading, dataset creation,
    optimizer setup, epoch loop, checkpointing, and the on_epoch callback.
    Saves checkpoints to model_paths(name)["weights"] / ["optimizer"].

    Training features:
        - Linear LR warmup (warmup_ratio of total steps) + cosine decay
        - Automatic mixed precision (AMP) when use_amp=True and on CUDA
        - Gradient accumulation over accumulation_steps mini-batches
    """
    paths = model_paths(name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    # Parse every score exactly once into the on-disk token cache (shared across
    # models, incremental), then stream tokenized records from it — no music21 in
    # the epoch loop. The val split is persisted by cache key (resolved path).
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    log(f"Building/opening token cache for {name} ({len(data_paths)} scores)...")
    store = build_or_open_cache(data_paths, cache_dir=cache_dir, n_workers=cache_workers, log=log)

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
    train_set = MusicDataset(store.subset_view(train_keys), max_len, fim_ratio, transpose_range=6, fim_example_fn=fim_example_fn)
    val_set   = MusicDataset(store.subset_view(val_keys),   max_len, fim_ratio, transpose_range=6, fim_example_fn=fim_example_fn)
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
        optimizer.load_state_dict(torch.load(paths["optimizer"], map_location=device, weights_only=False))
        log("Resumed optimizer state from previous session.")
    model.to(device)

    # AMP setup (only effective on CUDA)
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
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat      = y.view(-1)
                mask_flat   = loss_mask.view(-1)

                loss_per_token = nn.functional.cross_entropy(
                    logits_flat, y_flat, ignore_index=PAD, reduction='none'
                )
                masked_loss = loss_per_token * mask_flat
                loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)
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
                    logits_flat = logits.view(-1, logits.size(-1))
                    y_flat      = y.view(-1)
                    mask_flat   = loss_mask.view(-1)

                    loss_per_token = nn.functional.cross_entropy(
                        logits_flat, y_flat, ignore_index=PAD, reduction='none'
                    )
                    masked_loss = loss_per_token * mask_flat
                    val_loss   += (masked_loss.sum() / (mask_flat.sum() + 1e-8)).item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        on_epoch(epoch, avg_loss, avg_val_loss)

        torch.save(model.state_dict(), paths["weights"])
        torch.save(optimizer.state_dict(), paths["optimizer"])
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  ({name} checkpoint saved)")

    log(f"{name.capitalize()} training complete.")