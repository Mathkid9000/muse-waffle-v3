"""FIM (fill-in-the-middle) training for the MMT generator over compound events.

FIM sequence format (event rows):
    [SOS] header prefix [FILL_START] suffix [FILL_END] kept_gap [FILL_GEN] target_gap [EOS]

The model sees prefix (measures before the gap) and suffix (measures after the
gap) as horizontal context, plus kept_gap (the gap measures restricted to the
staves being KEPT) as vertical context, then generates target_gap (the gap
measures restricted to the staves being regenerated).  Loss is computed only
on target_gap.

Window-relative bar numbering is the key design point: prefix bars are
0..P-1, gap bars are P..P+G-1 — with IDENTICAL indices in kept_gap and
target_gap, so the vertical alignment between kept and regenerated staves is
explicit in the bar field — and suffix bars continue at P+G.

Two scenarios are mixed:
  - full-staff infill: all staves of the gap are regenerated; kept_gap is empty.
  - partial-staff infill: only a subset of staves is regenerated; the other
    staves appear in kept_gap so the model can match them harmonically and
    rhythmically.  This is the realistic "re-roll one staff, keep the rest"
    use case.
"""

from __future__ import annotations
import random
from typing import Callable

from ml.events import sos_event, eos_event, fill_start_event, fill_end_event, fill_gen_event
from ml.tokenizer import emit_measure_events, flatten_measure_events
from ml.trainer import _train_loop


def create_fim_example(
    header_events: list[list[int]],
    measure_events: dict[int, list[list[int]]],
    collected: tuple | None = None,
    n_parts: int = 1,
    min_gap: int = 1,
    max_gap: int = 4,
    max_context_measures: int = 16,
    partial_ratio: float = 0.5,
    max_len: int = 1024,
) -> tuple[list[list[int]], int] | None:
    """Create a single FIM training example from per-measure event blocks.

    Args:
        header_events: INSTR header rows for this score.
        measure_events: {measure number: all-staff event block} (bar fields 0).
        collected: raw output of `collect_score` (+ derive_bar_chords), used to
            re-emit per-staff-subset gap blocks.  When None, only full-staff
            examples are produced.
        n_parts: number of real staves in the score.
        min_gap / max_gap: measure-count range for the gap.
        max_context_measures: max measures of prefix/suffix context per side.
        partial_ratio: probability of a partial-staff (vs full-staff) example,
            when the score has >= 2 staves and `collected` is available.
        max_len: max sequence length in events; prefix/suffix context is
            trimmed (never the target region) so the example fits.

    Returns:
        (event_rows, loss_start_idx) or None if an example can't be created.
        loss_start_idx is the index immediately after FILL_GEN.
    """
    sorted_measures = sorted(measure_events.keys())
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

    # Decide which staves to regenerate. masked_parts=None means all staves.
    masked_parts: set[int] | None = None
    if collected is not None and n_parts >= 2 and random.random() < partial_ratio:
        k = random.randint(1, n_parts - 1)
        masked_parts = set(random.sample(range(n_parts), k))

    if masked_parts is None:
        kept_blocks = None
        target_blocks = measure_events
    else:
        kept_parts = set(range(n_parts)) - masked_parts
        kept_blocks = emit_measure_events(
            collected, part_filter=kept_parts, measure_nums=gap_measures)
        target_blocks = emit_measure_events(
            collected, part_filter=masked_parts, measure_nums=gap_measures)

    def block_len(m):  # header row count of one measure block
        return len(measure_events.get(m, [])) or 1

    target_len = sum(len(target_blocks.get(m, [])) or 1 for m in gap_measures)
    kept_len = (sum(len(kept_blocks.get(m, [])) or 1 for m in gap_measures)
                if kept_blocks is not None else 0)

    # Length guard: never truncate the target (loss) region.  Trim the farthest
    # prefix/suffix context measures (longer side first) until the example fits.
    fixed = len(header_events) + kept_len + target_len + 5  # SOS + 3 markers + EOS
    prefix_len = sum(block_len(m) for m in prefix_measures)
    suffix_len = sum(block_len(m) for m in suffix_measures)
    while (prefix_measures or suffix_measures) and \
            fixed + prefix_len + suffix_len > max_len:
        if prefix_len >= suffix_len and prefix_measures:
            prefix_measures = prefix_measures[1:]
            prefix_len = sum(block_len(m) for m in prefix_measures)
        elif suffix_measures:
            suffix_measures = suffix_measures[:-1]
            suffix_len = sum(block_len(m) for m in suffix_measures)
        else:
            break

    # Window-relative bar indices: prefix 0..P-1, gap P..P+G-1, suffix P+G.. .
    p = len(prefix_measures)
    g = len(gap_measures)
    prefix_rows = flatten_measure_events(measure_events, prefix_measures, start_bar=0)
    suffix_rows = flatten_measure_events(measure_events, suffix_measures, start_bar=p + g)
    if masked_parts is None:
        kept_rows: list[list[int]] = []
        target_rows = flatten_measure_events(measure_events, gap_measures, start_bar=p)
    else:
        kept_rows   = flatten_measure_events(kept_blocks, gap_measures, start_bar=p)
        target_rows = flatten_measure_events(target_blocks, gap_measures, start_bar=p)

    input_seq = (
        [sos_event()] + list(header_events) + prefix_rows + [fill_start_event()] +
        suffix_rows + [fill_end_event()] + kept_rows + [fill_gen_event()] +
        target_rows + [eos_event()]
    )
    loss_start_idx = (1 + len(header_events) + len(prefix_rows) + 1 +
                      len(suffix_rows) + 1 + len(kept_rows) + 1)

    return input_seq, loss_start_idx


def train_infill(
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
    fim_ratio: float = 0.5,
    resume_optimizer: bool = False,
    cache_workers: int = 1,
):
    """MLE-pretrain the generator with mixed standard + FIM objectives.

    This is phase (a) of MMT-BERT training when infill capability is wanted
    (recommended: one combined model with fim_ratio=0.5, then GAN fine-tuning
    via ml.gan_trainer).
    """
    _train_loop(
        name, model, data_paths, epochs, batch_size, lr, device,
        log, on_epoch, stop_flag,
        fim_ratio=fim_ratio,
        fim_example_fn=create_fim_example,
        resume_optimizer=resume_optimizer,
        cache_workers=cache_workers,
    )
