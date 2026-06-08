"""Trains MusicTransformer for fill-in-the-middle (FIM) music generation.

FIM Sequence Format:
    Original: [BOS] [M1] [M2] [M3] [M4] [M5] [M6] [EOS]
    FIM:      [BOS] [M1] [FILL_START] [M4] [M5] [M6] [FILL_END] [M2] [M3] [EOS]

The model sees both prefix (M1) and suffix (M4-M6) before generating the
fill content (M2-M3), enabling bidirectional context while using standard
causal attention.
"""

from __future__ import annotations
import random
from typing import Callable

from ml.tokenizer import BOS, EOS, BAR, FILL_START, FILL_END
from ml.model import MusicTransformer
from ml.trainer import _train_loop


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
        max_context_measures: Max measures of prefix/suffix context to include

    Returns:
        (input_sequence, loss_start_idx) or None if can't create
        - input_sequence: [BOS] + header + prefix + [FILL_START] + suffix + [FILL_END] + gap_content
        - loss_start_idx: Index where loss computation starts (after FILL_END)
    """
    sorted_measures = sorted(measure_tokens.keys())
    n_measures = len(sorted_measures)

    if n_measures < 3:
        return None

    gap_size = random.randint(min_gap, min(max_gap, n_measures - 2))
    max_start = n_measures - gap_size - 1
    if max_start < 1:
        return None
    gap_start_idx = random.randint(1, max_start)

    prefix_measures = sorted_measures[max(0, gap_start_idx - max_context_measures) : gap_start_idx]
    gap_measures    = sorted_measures[gap_start_idx : gap_start_idx + gap_size]
    suffix_measures = sorted_measures[gap_start_idx + gap_size : gap_start_idx + gap_size + max_context_measures]

    prefix_toks = [t for m in prefix_measures for t in measure_tokens[m]]
    gap_toks    = [t for m in gap_measures    for t in measure_tokens[m]]
    suffix_toks = [t for m in suffix_measures for t in measure_tokens[m]]

    input_seq = (
        [BOS] + header_tokens + prefix_toks + [FILL_START] +
        suffix_toks + [FILL_END] + gap_toks + [EOS]
    )
    loss_start_idx = len([BOS] + header_tokens + prefix_toks + [FILL_START] + suffix_toks + [FILL_END])

    return input_seq, loss_start_idx


def train_infill(
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
    fim_ratio: float = 0.5,
    resume_optimizer: bool = False,
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
        resume_optimizer: Resume optimizer state and val split from last run
    """
    _train_loop(
        name, model, data_paths, epochs, batch_size, lr, device,
        log, on_epoch, stop_flag,
        fim_ratio=fim_ratio,
        fim_example_fn=create_fim_example,
        resume_optimizer=resume_optimizer,
    )
