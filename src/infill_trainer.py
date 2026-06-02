"""Trains MusicTransformer for fill-in-the-middle (FIM) music generation.

This module provides training for infill capability where the model learns to
generate content that fits between a prefix and suffix context.

FIM Sequence Format:
    Original: [BOS] [M1] [M2] [M3] [M4] [M5] [M6] [EOS]
    FIM:      [BOS] [M1] [FILL_START] [M4] [M5] [M6] [FILL_END] [M2] [M3] [EOS]

The model sees both prefix (M1) and suffix (M4-M6) before generating the
fill content (M2-M3), enabling bidirectional context while using standard
causal attention.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.tokenizer import (
    score_to_tokens, score_to_tokens_by_measure, score_to_header_tokens,
    text_to_tokens,
    BOS, EOS, PAD, BAR, FILL_START, FILL_END,
    NOTE_OFF, KEY_OFF,
)
from src.model import MusicTransformer, WEIGHTS_FILE, OPTIMIZER_FILE, CHECKPOINT_DIR

_NOTE_END = NOTE_OFF + 128   # 133 — exclusive upper bound of NOTE token range
_KEY_END  = KEY_OFF  + 13   # 188 — exclusive upper bound of KEY_SIG token range


def _transpose_tokens(tokens: list[int], semitones: int) -> list[int]:
    """Shift NOTE and KEY_SIG tokens by semitones; all other tokens are unchanged.

    NOTE tokens are clamped to [0, 127] so extreme transpositions stay in range.
    KEY_SIG tokens are shifted via the circle of fifths (7 semitones = 1 sharp)
    and wrapped to the representable [-6, +6] sharps range.
    """
    if semitones == 0:
        return tokens
    result = []
    for t in tokens:
        if NOTE_OFF <= t < _NOTE_END:
            new_pitch = max(0, min(127, t - NOTE_OFF + semitones))
            result.append(NOTE_OFF + new_pitch)
        elif KEY_OFF <= t < _KEY_END:
            old_sharps = (t - KEY_OFF) - 6          # map token index → [-6, 6]
            new_sharps = ((old_sharps + semitones * 7) + 6) % 12 - 6
            result.append(KEY_OFF + new_sharps + 6)
        else:
            result.append(t)
    return result


def create_fim_example(
    header_tokens: list[int],
    measure_tokens: dict[int, list[int]],
    gap_ratio: float = 0.15,
    min_gap: int = 1,
    max_gap: int = 4,
    max_context_measures: int = 16,
) -> tuple[list[int], int] | None:
    """Create a single FIM training example from tokenized measures.

    Args:
        header_tokens: Instrument header tokens for this score
        measure_tokens: Dict mapping measure number to tokens for that measure
        gap_ratio: Target ratio of gap measures to total measures
        min_gap: Minimum number of measures to use as gap
        max_gap: Maximum number of measures to use as gap

    Returns:
        (input_sequence, loss_start_idx) or None if can't create
        - input_sequence: [BOS] + header + prefix + [FILL_START] + suffix + [FILL_END] + gap_content
        - loss_start_idx: Index where loss computation starts (after FILL_END)
    """
    sorted_measures = sorted(measure_tokens.keys())
    n_measures = len(sorted_measures)

    if n_measures < 3:  # Need at least prefix, gap, suffix
        return None

    # Determine gap size
    gap_size = random.randint(min_gap, min(max_gap, n_measures - 2))

    # Select gap start position (must leave room for prefix and suffix)
    max_start = n_measures - gap_size - 1  # Leave at least 1 measure for suffix
    if max_start < 1:  # Need at least 1 measure for prefix
        return None
    gap_start_idx = random.randint(1, max_start)

    # Build prefix, gap, and suffix — keeping only the nearest measures to the gap
    prefix_measures = sorted_measures[max(0, gap_start_idx - max_context_measures) : gap_start_idx]
    gap_measures    = sorted_measures[gap_start_idx : gap_start_idx + gap_size]
    suffix_measures = sorted_measures[gap_start_idx + gap_size : gap_start_idx + gap_size + max_context_measures]

    prefix_toks = [t for m in prefix_measures for t in measure_tokens[m]]
    gap_toks    = [t for m in gap_measures    for t in measure_tokens[m]]
    suffix_toks = [t for m in suffix_measures for t in measure_tokens[m]]

    # Construct FIM sequence:
    # [BOS] + header + prefix + [FILL_START] + suffix + [FILL_END] + gap_content + [EOS]
    input_seq = (
        [BOS] + header_tokens + prefix_toks + [FILL_START] +
        suffix_toks + [FILL_END] + gap_toks + [EOS]
    )

    # Loss starts after FILL_END (we want model to predict gap content)
    loss_start_idx = len([BOS] + header_tokens + prefix_toks + [FILL_START] + suffix_toks + [FILL_END])

    return input_seq, loss_start_idx


def create_standard_example(header_tokens: list[int], measure_tokens: dict[int, list[int]]) -> list[int]:
    """Create a standard next-token prediction example (no FIM)."""
    tokens = [BOS] + header_tokens
    for m_num in sorted(measure_tokens.keys()):
        tokens.extend(measure_tokens[m_num])
    tokens.append(EOS)
    return tokens


class InfillDataset(Dataset):
    """Dataset that produces both standard and FIM training examples.

    Each example is either:
    - Standard: Full sequence for next-token prediction (loss on all tokens)
    - FIM: Restructured sequence with loss only on fill tokens

    Gap position and size are re-randomized on every __getitem__ call so the
    model sees each song with a different gap each epoch.
    """

    def __init__(
        self,
        sequences: list[tuple[list[int], dict[int, list[int]]]],
        max_len: int = 512,
        fim_ratio: float = 0.5,
        transpose_range: int = 6,
    ):
        """
        Args:
            sequences: List of (header_tokens, measure_tokens) tuples
            max_len: Maximum sequence length
            fim_ratio: Proportion of examples that use FIM format
            transpose_range: Max semitone shift for pitch augmentation (0 to disable)
        """
        self.max_len = max_len
        self.fim_ratio = fim_ratio
        self.transpose_range = transpose_range
        # Only keep sequences long enough for FIM (prefix + gap + suffix)
        self.sequences = [(h, m) for h, m in sequences if len(m) >= 3]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        header_tokens, measure_tokens = self.sequences[idx]

        if random.random() < self.fim_ratio:
            result = create_fim_example(header_tokens, measure_tokens)
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

        # Create loss mask (1 = compute loss, 0 = ignore)
        loss_mask = [0] * len(y)
        for i in range(loss_start - 1, len(y)):  # -1 because y is shifted
            loss_mask[i] = 1

        # Pad sequences
        pad_len = self.max_len - len(x)
        x = x + [PAD] * pad_len
        y = y + [PAD] * pad_len
        loss_mask = loss_mask + [0] * pad_len

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def _load_measure_sequences(
    paths: list[str],
    log: Callable[[str], None],
    max_sequences=0
) -> list[tuple[list[int], dict[int, list[int]]]]:
    """Load scores and convert to (header_tokens, measure_tokens) tuples."""
    import music21 as m21

    sequences = []
    for path in tqdm(paths, desc="Parsing scores for infill training..."):
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


VAL_RATIO = 0.05


def train_infill(
    model: MusicTransformer,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float, float], None],
    stop_flag: list[bool],
    fim_ratio: float = 0.5,
):
    """Train model with mixed standard + FIM objectives.

    Args:
        model: MusicTransformer model (should have expanded vocab for infill tokens)
        data_paths: Paths to MusicXML files
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        log: Logging callback
        on_epoch: Epoch callback (epoch, train_loss, val_loss)
        stop_flag: Stop flag for interruption
        fim_ratio: Proportion of FIM examples (default 0.5 = 50%)
    """
    log("Parsing scores for infill training...")
    sequences = _load_measure_sequences(data_paths, log)
    if not sequences:
        raise RuntimeError("No sequences loaded - run 'Download Dataset' first.")
    log(f"Loaded {len(sequences)} scores.")

    random.shuffle(sequences)
    dataset = InfillDataset(sequences, model.cfg["max_seq_len"], fim_ratio, transpose_range=6)
    log(f"Dataset: {len(dataset)} examples (max_len={model.cfg['max_seq_len']}, fim_ratio={fim_ratio}, transpose_range=6)")

    n_val = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    log(f"Split: {n_train} train / {n_val} val examples.")

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    if OPTIMIZER_FILE.exists():
        optimizer.load_state_dict(torch.load(OPTIMIZER_FILE, map_location=device, weights_only=False))
        log("Resumed optimizer state from previous session.")
    model.to(device)

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("Training stopped by user.")
            break

        model.train()
        epoch_loss = 0.0
        batches = 0

        for x, y, loss_mask in tqdm(loader, desc=f"Infill Training Epoch {epoch}"):
            if stop_flag[0]:
                break

            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(x)

            # Compute loss with mask
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            mask_flat = loss_mask.view(-1)

            # Compute per-token loss
            loss_per_token = nn.functional.cross_entropy(
                logits_flat, y_flat, ignore_index=PAD, reduction='none'
            )

            # Apply mask and compute mean
            masked_loss = loss_per_token * mask_flat
            loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, loss_mask in val_loader:
                x = x.to(device)
                y = y.to(device)
                loss_mask = loss_mask.to(device)

                logits = model(x)
                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                mask_flat = loss_mask.view(-1)

                loss_per_token = nn.functional.cross_entropy(
                    logits_flat, y_flat, ignore_index=PAD, reduction='none'
                )
                masked_loss = loss_per_token * mask_flat
                loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        scheduler.step()
        on_epoch(epoch, avg_loss, avg_val_loss)

        # Save checkpoint
        torch.save(model.state_dict(), WEIGHTS_FILE)
        torch.save(optimizer.state_dict(), OPTIMIZER_FILE)
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  (infill checkpoint saved)")

    log("Infill training complete.")
