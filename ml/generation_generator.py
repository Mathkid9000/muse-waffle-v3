"""Inference pipeline for pure autoregressive music generation from scratch.

Public signatures are unchanged from the flat-token system — including the
legacy ``first_bar_prefix`` list of flat token ids built by
api/routers/generate.py from BAR/KEY_OFF/TIME_OFF/TEMPO_OFF — but internally
everything runs on compound events (see ml.events / ml.tokenizer).
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Generator

from ml.events import (
    F_TYPE, BAR_T, sos_event, eos_event, bar_event, key_event, time_event,
    tempo_event,
)
from ml.tokenizer import (
    events_to_score, score_to_xml_str, parse_legacy_first_bar_prefix, count_bars,
)
from ml.model import load_model

OUTPUT_DIR = Path("output")

DEFAULT_MODEL_NAME = "MMT-GAN-D512-L6"

# Instrument indices that use two staves (treble + bass).
_KEYBOARD_INSTRS = {1, 2, 3, 4}  # piano, harpsichord, organ, accordion


def _expand_instr_indices(instr_indices: list[int]) -> list[int]:
    """Expand keyboard instruments to two entries (one per staff)."""
    expanded: list[int] = []
    for idx in instr_indices:
        expanded.append(idx)
        if idx in _KEYBOARD_INSTRS:
            expanded.append(idx)  # second staff (bass)
    return expanded


def _forced_opening(first_bar_prefix: list[int] | None) -> list[list[int]]:
    """Legacy flat first_bar_prefix -> forced opening event rows for bar 0."""
    forced = [bar_event(0)]
    key_idx, time_idx, tempo_idx = parse_legacy_first_bar_prefix(first_bar_prefix)
    if key_idx is not None:
        forced.append(key_event(0, key_idx - 6))
    if time_idx is not None:
        forced.append(time_event(0, time_idx))
    if tempo_idx is not None:
        forced.append(tempo_event(0, tempo_idx))
    return forced


def generate_and_export(
    instr_indices: list[int],
    first_bar_prefix: list[int] | None = None,
    target_bars: int = 16,
    max_new: int = 2000,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, str]:
    """Generate a new score from scratch using the generation model.

    Args:
        instr_indices: Instrument category indices (0-31, one per part).
        first_bar_prefix: Legacy flat token ids forcing key/time/tempo, e.g.
            [BAR, KEY_OFF+k, TIME_OFF+t, TEMPO_OFF+b]. Defaults to a bare bar.
        target_bars: Total number of measures to generate.

    Returns:
        Dict with paths to output files: {"midi": ..., "xml": ...}
    """
    log("Loading generation model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    expanded = _expand_instr_indices(instr_indices)
    header = model.sample_header(
        sorted(expanded, reverse=True), temperature, top_k, device)
    log(f"Expanded instruments: {expanded}, sampled header: {header}")

    forced = _forced_opening(first_bar_prefix)

    generated = model.generate_standard(
        header_events=header,
        forced_events=forced,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
        max_parts=len(expanded),
    )

    full = [sos_event()] + header + forced + generated + [eos_event()]
    score = events_to_score(full)

    OUTPUT_DIR.mkdir(exist_ok=True)
    midi_path = str(OUTPUT_DIR / "generated.mid")
    xml_path  = str(OUTPUT_DIR / "generated.xml")

    score.write("midi", fp=midi_path)
    log(f"Saved MIDI -> {midi_path}")

    try:
        score.write("musicxml", fp=xml_path)
        log(f"Saved MusicXML -> {xml_path}")
    except Exception as e:
        log(f"MusicXML export skipped: {e}")
        xml_path = ""

    note_count    = sum(1 for el in score.flatten().notesAndRests if el.isNote or el.isChord)
    measure_count = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    log(f"Generated score: {measure_count} measures, {note_count} notes/chords")

    return {"midi": midi_path, "xml": xml_path}


def generate_and_stream(
    instr_indices: list[int],
    first_bar_prefix: list[int] | None = None,
    target_bars: int = 16,
    max_new: int = 2000,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = DEFAULT_MODEL_NAME,
) -> Generator[tuple[int, str, int, int], None, None]:
    """Stream partial MusicXML as each measure is generated.

    Yields (bar_index, xml_str, total_bars, num_instr) after each completed
    measure; bar_index counts all bars visible in the partial score
    (including the forced opening bar).
    """
    log("Loading generation model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    expanded = _expand_instr_indices(instr_indices)
    header = model.sample_header(
        sorted(expanded, reverse=True), temperature, top_k, device)
    log(f"Expanded instruments: {expanded}, sampled header: {header}")

    forced = _forced_opening(first_bar_prefix)
    n_forced_bars = count_bars(forced)
    last_snapshot: list[list[int]] = []

    def snapshot_xml(snapshot):
        full = [sos_event()] + header + forced + snapshot + [eos_event()]
        return score_to_xml_str(events_to_score(full))

    for snapshot in model.generate_standard_stream(
        header_events=header,
        forced_events=forced,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
        max_parts=len(expanded),
    ):
        last_snapshot = snapshot
        bar_count = n_forced_bars + count_bars(snapshot)
        try:
            xml_str = snapshot_xml(snapshot)
        except Exception as e:
            log(f"Stream XML export error: {e}")
            continue
        log(f"Streamed bar {bar_count} of {target_bars}")
        yield (bar_count, xml_str, target_bars, len(header))

    # Always emit the final complete score (last measure now has its notes)
    try:
        xml_str = snapshot_xml(last_snapshot)
        final_bar = n_forced_bars + count_bars(last_snapshot)
        log(f"Stream complete: {final_bar} bars")
        yield (final_bar, xml_str, target_bars, len(header))
    except Exception as e:
        log(f"Final stream XML export error: {e}")
