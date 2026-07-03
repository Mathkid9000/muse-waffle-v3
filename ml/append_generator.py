"""Inference pipeline for appending new measures to an existing score.

The existing score is tokenized to compound events, renumbered to window bars
0..C-1 (left-truncating whole measures when the context would overflow the
model window or the MAX_BARS bar-index range), used as a committed prompt for
generate_standard, and the newly generated measures are extracted and appended
to the original score.  Public signatures unchanged.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Generator

from ml.events import MAX_BARS, sos_event, eos_event, bar_event
from ml.tokenizer import (
    collect_score, score_to_header_events, emit_measure_events,
    flatten_measure_events, events_to_score, score_to_xml_str, count_bars,
)
from ml.model import load_model

OUTPUT_DIR = Path("output")

DEFAULT_MODEL_NAME = "MMT-GAN-D512-L6"


def _context_events(orig_score, max_seq_len: int, num_bars: int, header_len: int, log):
    """Tokenize the existing score into committed context rows (bars 0..C-1).

    Truncates whole measures from the left until the context fits both the
    model window (reserving ~256 events for generation) and the bar-index
    range (context bars + new bars <= MAX_BARS).
    """
    collected = collect_score(orig_score)
    blocks = emit_measure_events(collected)
    nums = sorted(blocks.keys())

    max_ctx_events = max_seq_len - 256 - header_len - 1  # reserve gen + SOS
    max_ctx_bars = MAX_BARS - num_bars

    def total_len(ns):
        return sum(len(blocks[m]) for m in ns)

    trimmed = False
    while nums and (total_len(nums) > max_ctx_events or len(nums) > max_ctx_bars):
        nums = nums[1:]
        trimmed = True
    if trimmed:
        log(f"Context truncated: using last {len(nums)} of {len(blocks)} measures")

    if not nums:
        return [bar_event(0)], 1
    return flatten_measure_events(blocks, nums, start_bar=0), len(nums)


def _extract_new_measures(cont_parts, part_idx, n_ctx_bars, count):
    import music21 as m21
    if part_idx < len(cont_parts):
        cont_measures = list(cont_parts[part_idx].getElementsByClass(m21.stream.Measure))
        return cont_measures[n_ctx_bars:n_ctx_bars + count]
    return []


def _append_measures(orig_score, continuation_score, n_ctx_bars, take, pad_to, orig_last_m_num):
    """Append up to `take` new measures from the continuation onto orig_score,
    padding with full-measure rests up to `pad_to` when generation came up short."""
    import music21 as m21
    cont_parts = list(continuation_score.parts)
    for part_idx, orig_part in enumerate(orig_score.parts):
        next_m_num = orig_last_m_num + 1
        new_measures = _extract_new_measures(cont_parts, part_idx, n_ctx_bars, take)
        for new_m in new_measures:
            new_m.number = next_m_num
            orig_part.append(new_m)
            next_m_num += 1
        for _ in range(len(new_measures), pad_to):
            rest_m = m21.stream.Measure(number=next_m_num)
            rest_m.append(m21.note.Rest(quarterLength=4.0))
            orig_part.append(rest_m)
            next_m_num += 1


def _last_measure_number(orig_score) -> int:
    import music21 as m21
    parts = list(orig_score.parts)
    if not parts:
        return 0
    return max(
        (m.number or 0)
        for p in parts
        for m in p.getElementsByClass(m21.stream.Measure)
    )


def append_and_export(
    xml_str: str,
    num_bars: int = 4,
    max_new: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, str]:
    """Append num_bars newly generated measures to an existing score.

    Returns:
        Dict with path to output file: {"xml": ...}
    """
    import music21 as m21

    log("Parsing existing score...")
    orig_score = m21.converter.parseData(xml_str, format="musicxml")

    log("Loading generation model...")
    model, cfg = load_model(model_name, device)
    max_seq_len: int = cfg.get("max_seq_len", 1024)
    log(f"Model loaded ({model.num_params():,} params, max_seq_len={max_seq_len})")

    header = score_to_header_events(orig_score)
    context, n_ctx_bars = _context_events(orig_score, max_seq_len, num_bars, len(header), log)

    target_bars = n_ctx_bars + num_bars
    log(f"Generating {num_bars} new bars (context has {n_ctx_bars} bars, target={target_bars})...")

    generated = model.generate_standard(
        header_events=header,
        forced_events=context,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    full = [sos_event()] + header + context + generated + [eos_event()]
    continuation_score = events_to_score(full)

    orig_last_m_num = _last_measure_number(orig_score)
    _append_measures(orig_score, continuation_score, n_ctx_bars,
                     take=min(count_bars(generated), num_bars), pad_to=num_bars,
                     orig_last_m_num=orig_last_m_num)

    try:
        for part in orig_score.parts:
            part.makeBeams(inPlace=True)
            part.makeAccidentals(inPlace=True, overrideStatus=True)
    except Exception as e:
        log(f"makeBeams skipped: {e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    xml_path = str(OUTPUT_DIR / "appended.xml")
    try:
        orig_score.write("musicxml", fp=xml_path)
        log(f"Saved appended MusicXML -> {xml_path}")
    except Exception as e:
        log(f"MusicXML export failed: {e}")
        raise

    total_measures = orig_last_m_num + num_bars
    log(f"Score now has {total_measures} measures ({num_bars} appended)")
    return {"xml": xml_path}


def append_and_stream(
    xml_str: str,
    num_bars: int = 4,
    max_new: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = DEFAULT_MODEL_NAME,
) -> Generator[tuple[int, str, int], None, None]:
    """Stream partial MusicXML as each new measure is generated and merged.

    Yields (bar_index, xml_str, total_bars) where bar_index is the count of
    new measures appended so far.
    """
    import music21 as m21
    import copy

    log("Parsing existing score...")
    orig_score = m21.converter.parseData(xml_str, format="musicxml")

    log("Loading generation model...")
    model, cfg = load_model(model_name, device)
    max_seq_len: int = cfg.get("max_seq_len", 1024)
    log(f"Model loaded ({model.num_params():,} params)")

    header = score_to_header_events(orig_score)
    context, n_ctx_bars = _context_events(orig_score, max_seq_len, num_bars, len(header), log)
    target_bars = n_ctx_bars + num_bars

    orig_last_m_num = _last_measure_number(orig_score)
    last_snapshot: list[list[int]] = []

    def emit(snapshot, final: bool):
        new_bars = count_bars(snapshot)
        full = [sos_event()] + header + context + snapshot + [eos_event()]
        continuation_score = events_to_score(full)
        partial_score = copy.deepcopy(orig_score)
        take = min(new_bars, num_bars)
        _append_measures(partial_score, continuation_score, n_ctx_bars,
                         take=take, pad_to=num_bars if final else take,
                         orig_last_m_num=orig_last_m_num)
        try:
            for part in partial_score.parts:
                if final:
                    part.makeBeams(inPlace=True)
                part.makeAccidentals(inPlace=True, overrideStatus=True)
        except Exception:
            pass
        return new_bars, score_to_xml_str(partial_score)

    for snapshot in model.generate_standard_stream(
        header_events=header,
        forced_events=context,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
    ):
        last_snapshot = snapshot
        try:
            new_bars, xml_out = emit(snapshot, final=False)
            yield (new_bars, xml_out, num_bars)
        except Exception as e:
            log(f"Append stream XML error: {e}")

    try:
        new_bars, xml_out = emit(last_snapshot, final=True)
        log(f"Append stream complete: {new_bars} new bars")
        yield (new_bars, xml_out, num_bars)
    except Exception as e:
        log(f"Final append stream XML error: {e}")
