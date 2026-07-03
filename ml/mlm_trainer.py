"""Masked-event pretraining for the MusicBERT-style discriminator (phase b).

Follows MusicBERT's masked-language-modelling objective adapted to compound
events: a fraction of non-special events have their entire embedded
representation replaced by a learned [MASK] vector, and the encoder predicts
every field of the original event through the per-field MLM heads.

Two masking schemes:
  * "event" (default): each eligible event is masked independently with
    probability ``mask_prob``.
  * "bar": whole bars are masked (MusicBERT's anti-leakage trick — neighbours
    of a masked note inside the same bar are strong give-aways).  Bars are
    picked at random until roughly ``mask_prob`` of eligible events is covered.

Conventions mirror ml.trainer._train_loop: same token cache, persisted val
split, AdamW + warmup/cosine, AMP, checkpoint every epoch to
model_paths(name)["weights"], and the on_epoch(epoch, train, val) callback
(wire it to append_epoch_to_config for the training-curve plots).
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

from ml.events import F_TYPE, F_BAR, PAD_T, BAR_T, special_event, transpose_events
from ml.model import model_paths
from ml.musicbert import maskable_positions
from ml.preprocess import build_or_open_cache, _path_key, DEFAULT_CACHE_DIR
from ml.trainer import (
    VAL_RATIO, TRANSPOSE_LOW, TRANSPOSE_HIGH,
    _make_split, compound_ce_loss, create_standard_example,
)


class MLMDataset(Dataset):
    """Standard-format sequences plus a per-sample MLM mask."""

    def __init__(
        self,
        sequences,
        max_len: int = 1024,
        mask_prob: float = 0.15,
        mask_scheme: str = "event",
        transpose: bool = True,
    ):
        assert mask_scheme in ("event", "bar")
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_scheme = mask_scheme
        self.transpose = transpose
        if isinstance(sequences, list):
            self.sequences = [t for t in sequences if len(t[1]) >= 3]
        else:
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        header_events, measure_events, collected, n_parts = self.sequences[idx]
        seq = create_standard_example(header_events, measure_events)
        if self.transpose:
            seq = transpose_events(seq, random.randint(TRANSPOSE_LOW, TRANSPOSE_HIGH))
        seq = seq[:self.max_len]
        seq = seq + [special_event(PAD_T)] * (self.max_len - len(seq))
        x = torch.tensor(seq, dtype=torch.long)

        eligible = maskable_positions(x.unsqueeze(0))[0]
        if self.mask_scheme == "event":
            mask = (torch.rand(len(seq)) < self.mask_prob) & eligible
        else:
            # Bar scheme: mask whole bars until ~mask_prob of eligible events.
            bars = sorted({int(r[F_BAR]) for r in seq if r[F_TYPE] == BAR_T})
            mask = torch.zeros(len(seq), dtype=torch.bool)
            if bars:
                target = max(1, int(self.mask_prob * int(eligible.sum())))
                random.shuffle(bars)
                bar_field = x[:, F_BAR]
                for b in bars:
                    mask |= (bar_field == b) & eligible
                    if int(mask.sum()) >= target:
                        break
        if eligible.any() and not mask.any():
            # Guarantee at least one masked position so the loss is defined.
            first = int(eligible.nonzero()[torch.randint(int(eligible.sum()), (1,))])
            mask[first] = True
        return x, mask


def train_musicbert_mlm(
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
    mask_prob: float = 0.15,
    mask_scheme: str = "event",
    resume_optimizer: bool = False,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    warmup_ratio: float = 0.05,
    cache_dir=None,
    cache_workers: int = 1,
):
    """Pretrain a MusicBERTDiscriminator with masked-event prediction."""
    paths = model_paths(name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

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
        if not val_keys:
            val_keys, train_keys = _make_split(all_keys)
            val_split_file.write_text(json.dumps(val_keys))
        log(f"Loaded preserved val split: {len(val_keys)} val / {len(train_keys)} train.")
    else:
        val_keys, train_keys = _make_split(all_keys)
        val_split_file.write_text(json.dumps(val_keys))
        log(f"Created val split: {len(val_keys)} val / {len(train_keys)} train.")

    max_len = model.cfg["max_seq_len"]
    train_set = MLMDataset(store.subset_view(train_keys), max_len, mask_prob, mask_scheme)
    val_set   = MLMDataset(store.subset_view(val_keys),   max_len, mask_prob, mask_scheme)
    log(f"MLM dataset: {len(train_set)} train / {len(val_set)} val "
        f"(max_len={max_len}, mask_prob={mask_prob}, scheme={mask_scheme})")

    loader     = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

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

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("Training stopped by user.")
            break

        model.train()
        epoch_loss, batches = 0.0, 0
        optimizer.zero_grad()

        for step, (x, mask) in enumerate(tqdm(loader, desc=f"{name} MLM Epoch {epoch}")):
            if stop_flag[0]:
                break
            x, mask = x.to(device), mask.to(device)

            with torch.amp.autocast(device, dtype=torch.float16, enabled=amp_enabled):
                logits = model.forward_mlm(x, mask)
                loss = compound_ce_loss(logits, x, mask.float())
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
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(device), mask.to(device)
                with torch.amp.autocast(device, dtype=torch.float16, enabled=amp_enabled):
                    logits = model.forward_mlm(x, mask)
                    val_loss += compound_ce_loss(logits, x, mask.float()).item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        on_epoch(epoch, avg_loss, avg_val_loss)

        torch.save(model.state_dict(), paths["weights"])
        torch.save(optimizer.state_dict(), paths["optimizer"])
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  mlm_loss={avg_loss:.4f}  val_mlm={avg_val_loss:.4f}  ({name} checkpoint saved)")

    log(f"{name} MLM pretraining complete.")
