"""Shared training utilities for MusicTransformer models."""

from __future__ import annotations
import json
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
    BOS, EOS, PAD,
    NOTE_OFF, KEY_OFF,
)
from ml.model import MusicTransformer, model_paths, WEIGHTS_FILE

_NOTE_END = NOTE_OFF + 128
_KEY_END  = KEY_OFF  + 13

VAL_RATIO = 0.05


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
        sequences: list[tuple[list[int], dict[int, list[int]]]],
        max_len: int = 512,
        fim_ratio: float = 0.0,
        transpose_range: int = 6,
        fim_example_fn=None,
    ):
        """
        Args:
            sequences: List of (header_tokens, measure_tokens) tuples
            max_len: Maximum sequence length (tokens)
            fim_ratio: Proportion of examples that use FIM format
            transpose_range: Max semitone shift for pitch augmentation (0 to disable)
            fim_example_fn: Callable(header, measures) -> (seq, loss_start) | None
        """
        self.max_len = max_len
        self.fim_ratio = fim_ratio
        self.transpose_range = transpose_range
        self.fim_example_fn = fim_example_fn
        self.sequences = [(h, m) for h, m in sequences if len(m) >= 3]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        header_tokens, measure_tokens = self.sequences[idx]

        if self.fim_ratio > 0 and self.fim_example_fn is not None and random.random() < self.fim_ratio:
            result = self.fim_example_fn(header_tokens, measure_tokens)
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
) -> list[tuple[list[int], dict[int, list[int]]]]:
    """Load scores and convert to (header_tokens, measure_tokens) tuples."""
    import music21 as m21

    sequences = []
    for path in tqdm(paths, desc="Parsing scores..."):
        if max_sequences != 0 and len(sequences) >= max_sequences:
            break
        try:
            score = m21.converter.parse(path)
            header = score_to_header_tokens(score)
            measure_tokens = score_to_tokens_by_measure(score)
            if measure_tokens and len(measure_tokens) >= 3:
                sequences.append((header, measure_tokens))
        except Exception as e:
            log(f"  Skip {Path(path).name}: {e}")

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
):
    """Core training loop shared by all model trainers.

    Handles val split persistence, sequence loading, dataset creation,
    optimizer setup, epoch loop, checkpointing, and the on_epoch callback.
    Saves checkpoints to model_paths(name)["weights"] / ["optimizer"].
    """
    paths = model_paths(name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    val_split_file = paths["val_split"]
    if resume_optimizer and val_split_file.exists():
        saved_val        = set(json.loads(val_split_file.read_text()))
        val_data_paths   = [p for p in data_paths if str(p) in saved_val]
        train_data_paths = [p for p in data_paths if str(p) not in saved_val]
        log(f"Loaded preserved val split: {len(val_data_paths)} val / {len(train_data_paths)} train paths.")
    else:
        shuffled = list(data_paths)
        random.shuffle(shuffled)
        n_val            = max(1, int(len(shuffled) * VAL_RATIO))
        val_data_paths   = shuffled[-n_val:]
        train_data_paths = shuffled[:-n_val]
        val_split_file.write_text(json.dumps([str(p) for p in val_data_paths]))
        log(f"Created val split: {len(val_data_paths)} val / {len(train_data_paths)} train paths.")

    log(f"Parsing scores for {name} training...")
    train_sequences = _load_measure_sequences(train_data_paths, log)
    val_sequences   = _load_measure_sequences(val_data_paths,   log)
    if not train_sequences:
        raise RuntimeError("No sequences loaded — run 'Download Dataset' first.")
    log(f"Loaded {len(train_sequences)} train / {len(val_sequences)} val scores.")

    max_len   = model.cfg["max_seq_len"]
    train_set = MusicDataset(train_sequences, max_len, fim_ratio, transpose_range=6, fim_example_fn=fim_example_fn)
    val_set   = MusicDataset(val_sequences,   max_len, fim_ratio, transpose_range=6, fim_example_fn=fim_example_fn)
    log(f"Dataset: {len(train_set)} train / {len(val_set)} val (max_len={max_len}, fim_ratio={fim_ratio})")

    loader     = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    if paths["optimizer"].exists() and resume_optimizer:
        optimizer.load_state_dict(torch.load(paths["optimizer"], map_location=device, weights_only=False))
        log("Resumed optimizer state from previous session.")
    model.to(device)

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("Training stopped by user.")
            break

        model.train()
        epoch_loss = 0.0
        batches    = 0

        for x, y, loss_mask in tqdm(loader, desc=f"{name.capitalize()} Epoch {epoch}"):
            if stop_flag[0]:
                break

            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            logits = model(x)

            logits_flat = logits.view(-1, logits.size(-1))
            y_flat      = y.view(-1)
            mask_flat   = loss_mask.view(-1)

            loss_per_token = nn.functional.cross_entropy(
                logits_flat, y_flat, ignore_index=PAD, reduction='none'
            )
            masked_loss = loss_per_token * mask_flat
            loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        avg_loss = epoch_loss / max(batches, 1)

        model.eval()
        val_loss    = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, loss_mask in val_loader:
                x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
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
        scheduler.step()
        on_epoch(epoch, avg_loss, avg_val_loss)

        torch.save(model.state_dict(), paths["weights"])
        torch.save(optimizer.state_dict(), paths["optimizer"])
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  ({name} checkpoint saved)")

    log(f"{name.capitalize()} training complete.")