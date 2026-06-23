"""Trains MusicTransformer for fill-in-the-middle (FIM) music generation.

FIM Sequence Format:
    [BOS] header prefix [FILL_START] suffix [FILL_END] kept_gap [FILL_GEN] target_gap [EOS]

The model sees prefix (measures before the gap) and suffix (measures after the gap)
as horizontal context, plus kept_gap (the gap measures restricted to the staves that
are being KEPT) as vertical context, then generates target_gap (the gap measures
restricted to the staves being regenerated). Loss is computed only on target_gap.

Two scenarios are mixed:
  - full-staff infill: all staves of the gap are regenerated; kept_gap is empty.
  - partial-staff infill: only a subset of staves is regenerated; the other staves
    appear in kept_gap so the model can match them harmonically/rhythmically. This
    is the realistic "re-roll one staff, keep the rest" use case.
"""

from __future__ import annotations
import random
from typing import Callable

from ml.tokenizer import BOS, EOS, BAR, FILL_START, FILL_END, FILL_GEN, _emit_measure_tokens
from ml.model import MusicTransformer
from ml.trainer import _train_loop


def create_fim_example(
    header_tokens: list[int],
    measure_tokens: dict[int, list[int]],
    collected: tuple | None = None,
    n_parts: int = 1,
    min_gap: int = 1,
    max_gap: int = 4,
    max_context_measures: int = 16,
    partial_ratio: float = 0.5,
    max_len: int = 2048,
) -> tuple[list[int], int] | None:
    """Create a single FIM training example from tokenized measures.

    Args:
        header_tokens: Instrument header tokens for this score
        measure_tokens: Dict mapping measure number to (all-staff) tokens
        collected: Raw (events_by_measure, key_toks, time_toks, tempo_toks) from
            `_collect_events_by_measure`, used to re-emit per-staff-subset gap
            tokens. When None, only full-staff examples are produced.
        n_parts: Number of real staves in the score
        min_gap / max_gap: Min/max number of measures in the gap
        max_context_measures: Max measures of prefix/suffix context per side
        partial_ratio: Probability of producing a partial-staff (vs full-staff)
            example, when the score has >= 2 staves and `collected` is available
        max_len: Max sequence length; prefix/suffix context is trimmed (never the
            target region) so the example fits

    Returns:
        (input_sequence, loss_start_idx) or None if can't create.
        loss_start_idx is the index immediately after FILL_GEN (start of target_gap).
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

    # Decide which staves to regenerate. masked_parts=None means all staves (full-gap).
    masked_parts: set[int] | None = None
    if collected is not None and n_parts >= 2 and random.random() < partial_ratio:
        k = random.randint(1, n_parts - 1)
        masked_parts = set(random.sample(range(n_parts), k))

    if masked_parts is None:
        kept_gap_toks: list[int] = []
        target_gap_toks = [t for m in gap_measures for t in measure_tokens[m]]
    else:
        kept_parts = set(range(n_parts)) - masked_parts
        events_by_measure, key_toks, time_toks, tempo_toks = collected
        kept_mt = _emit_measure_tokens(
            events_by_measure, key_toks, time_toks, tempo_toks,
            part_filter=kept_parts, measure_nums=gap_measures,
        )
        target_mt = _emit_measure_tokens(
            events_by_measure, key_toks, time_toks, tempo_toks,
            part_filter=masked_parts, measure_nums=gap_measures,
        )
        kept_gap_toks   = [t for m in gap_measures for t in kept_mt[m]]
        target_gap_toks = [t for m in gap_measures for t in target_mt[m]]

    prefix_toks = [t for m in prefix_measures for t in measure_tokens[m]]
    suffix_toks = [t for m in suffix_measures for t in measure_tokens[m]]

    # Length guard: never truncate the target (loss) region. Trim the farthest
    # prefix/suffix context measures (longer side first) until the example fits.
    fixed = len(header_tokens) + len(kept_gap_toks) + len(target_gap_toks) + 5  # BOS+3 markers+EOS
    while (prefix_measures or suffix_measures) and \
            fixed + len(prefix_toks) + len(suffix_toks) > max_len:
        if len(prefix_toks) >= len(suffix_toks) and prefix_measures:
            prefix_measures = prefix_measures[1:]
            prefix_toks = [t for m in prefix_measures for t in measure_tokens[m]]
        elif suffix_measures:
            suffix_measures = suffix_measures[:-1]
            suffix_toks = [t for m in suffix_measures for t in measure_tokens[m]]
        else:
            break

    input_seq = (
        [BOS] + header_tokens + prefix_toks + [FILL_START] +
        suffix_toks + [FILL_END] + kept_gap_toks + [FILL_GEN] +
        target_gap_toks + [EOS]
    )
    loss_start_idx = len(
        [BOS] + header_tokens + prefix_toks + [FILL_START] +
        suffix_toks + [FILL_END] + kept_gap_toks + [FILL_GEN]
    )

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
