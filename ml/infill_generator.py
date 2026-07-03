"""Inference pipeline for fill-in-the-middle music generation.

This module handles:
1. Loading partial MusicXML files with empty measures
2. Detecting gaps (empty regions)
3. Generating fill content using the trained infill-capable model
4. Merging generated content back into the score
5. Exporting completed MIDI and MusicXML

Public signatures (infill_and_export / infill_and_stream) are unchanged from
the flat-token system; internals run on compound events.  The generated fill
rows carry window-relative bar indices in the gap range, so
``events_to_score`` numbers the fill measures consecutively and the merge maps
them onto the original gap measures positionally, exactly as before.
"""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Callable, Generator

from ml.events import sos_event, eos_event
from ml.tokenizer import (
    identify_empty_measures, collect_score, score_to_header_events,
    extract_prefix_suffix_events, gap_measure_events,
    events_to_score, score_to_xml_str, count_bars,
    MAX_REAL_PARTS,
)
from ml.model import load_model

OUTPUT_DIR = Path("output")

DEFAULT_MODEL_NAME = "MMT-GAN-D512-L6"


def load_partial_xml(xml_path: str) -> tuple:
    """Parse input XML and identify gaps.

    Returns (score, gap_descriptors) where gap_descriptors is a list of
    (start_measure, end_measure) tuples.
    """
    import music21 as m21

    score = m21.converter.parse(xml_path)
    gaps = identify_empty_measures(score)

    return score, gaps


def extract_prefix_suffix(
    score,
    gap_start: int,
    gap_end: int,
    target_parts: list[int] | None = None,
) -> tuple[list, list, list, int, list, int]:
    """Extract prefix, suffix, header, and kept-staff gap events around a gap.

    Returns:
        (prefix, suffix, header, target_bars, kept_gap, gap_bar_start)
        - prefix/suffix: event rows before/after the gap (all staves), bars
          renumbered window-relative (prefix 0..P-1, suffix P+G..)
        - header: INSTR header events for this score
        - target_bars: number of measures to generate (gap size)
        - kept_gap: gap measures restricted to kept staves (empty for
          full-staff infill), bars P..P+G-1
        - gap_bar_start: P, the window bar index of the first gap measure
    """
    collected = collect_score(score)
    prefix, suffix, gap_bar_start = extract_prefix_suffix_events(
        collected, gap_start, gap_end)
    header = score_to_header_events(score)
    target_bars = gap_end - gap_start + 1

    kept_gap: list = []
    if target_parts is not None:
        n_parts = len(list(score.parts)[:MAX_REAL_PARTS])
        kept_parts = set(range(n_parts)) - set(target_parts)
        if kept_parts:
            kept_gap = gap_measure_events(
                collected, gap_start, gap_end,
                part_filter=kept_parts, start_bar=gap_bar_start)

    return prefix, suffix, header, target_bars, kept_gap, gap_bar_start


def merge_fill_into_score(
    score,
    fill_events: list[list[int]],
    gap_start: int,
    gap_end: int,
    color='red',
    target_parts: list[int] | None = None,
):
    """Insert generated events back into the score at correct measure positions.

    Modifies the score in-place by decoding fill_events to a temporary score,
    then replacing the musical content of the gap measures (structural
    elements — clef, key/time sig, tempo — stay in place).
    """
    import music21 as m21

    fill_score = events_to_score([sos_event()] + list(fill_events) + [eos_event()])

    if not fill_score.parts:
        return

    # Map each generated part to an ORIGINAL part index.  events_to_score
    # compacts parts to sorted lane order, so when only a subset of staves is
    # generated the i-th generated part corresponds to the i-th (ascending)
    # target part.  Full-staff infill is identity 0,1,2,...
    gen_parts = list(fill_score.parts)
    if target_parts is not None:
        dest_indices = sorted(target_parts)
    else:
        dest_indices = list(range(len(gen_parts)))

    if len(gen_parts) != len(dest_indices):
        print(f"[infill] warning: model generated {len(gen_parts)} part(s) but "
              f"{len(dest_indices)} were expected; mapping the first "
              f"{min(len(gen_parts), len(dest_indices))} by position.")

    # fill_measures[original_part_idx][measure_number] = generated measure
    fill_measures = {}
    for i, gen_part in enumerate(gen_parts):
        if i >= len(dest_indices):
            break
        orig_idx = dest_indices[i]
        fill_measures[orig_idx] = {}
        for measure in gen_part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 1
            fill_measures[orig_idx][m_num] = measure

    target_measures = list(range(gap_start, gap_end + 1))

    for part_idx, part in enumerate(score.parts):
        if target_parts is not None and part_idx not in target_parts:
            continue
        if part_idx not in fill_measures:
            continue

        generated_nums = sorted(fill_measures[part_idx].keys())

        for i, target_m_num in enumerate(target_measures):
            if i >= len(generated_nums):
                break

            gen_measure = fill_measures[part_idx][generated_nums[i]]

            for measure in part.getElementsByClass(m21.stream.Measure):
                m_num = measure.number if measure.number is not None else 0
                if m_num == target_m_num:
                    # Remove only musical content; structural elements stay.
                    _musical = (m21.note.Note, m21.note.Rest, m21.chord.Chord, m21.stream.Voice)
                    for el in [e for e in measure if isinstance(e, _musical)]:
                        measure.remove(el)

                    for el in gen_measure:
                        if isinstance(el, (m21.note.Note, m21.note.Rest, m21.chord.Chord)):
                            if color is not None: el.style.color = color
                            measure.insert(el.offset, el)
                        elif isinstance(el, m21.stream.Voice):
                            if color is not None: el.style.color = color
                            new_voice = m21.stream.Voice(id=el.id)
                            for voice_el in el:
                                if color is not None: voice_el.style.color = color
                                new_voice.insert(voice_el.offset, voice_el)
                            measure.insert(0, new_voice)
                    break

    # Do beam and accidental correction
    for part in score.parts:
        measures = list(part.getElementsByClass(m21.stream.Measure))
        if measures and not any(
            list(m.getElementsByClass(m21.meter.TimeSignature)) for m in measures
        ):
            measures[0].insert(0, m21.meter.TimeSignature('4/4'))
    try:
        for part in score.parts:
            part.makeBeams(inPlace=True)
            part.makeAccidentals(inPlace=True, overrideStatus=True)
    except Exception as e:
        print(f"makeBeams error: {e}")


def infill_and_export(
    score,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    gaps: list[tuple[int, int]] | None = None,
    target_parts: list[int] | None = None,
) -> dict[str, str]:
    """Main entry point for infill generation.

    Processes gaps left-to-right.  After filling each gap, the filled content
    becomes part of the prefix for subsequent gaps.

    Returns:
        Dict with paths to output files: {"midi": ..., "xml": ...}
    """
    if gaps is None:
        gaps = identify_empty_measures(score)

    if not gaps:
        log("No empty measures detected. Nothing to fill.")
        return {"midi": "", "xml": ""}

    log(f"Detected {len(gaps)} gap(s): {gaps}")

    log("Loading infill model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    for i, (gap_start, gap_end) in enumerate(gaps):
        log(f"Processing gap {i+1}/{len(gaps)}: measures {gap_start}-{gap_end}")

        prefix, suffix, header, target_bars, kept_gap, gap_bar_start = \
            extract_prefix_suffix(score, gap_start, gap_end, target_parts=target_parts)
        log(f"  Prefix: {len(prefix)} events, Suffix: {len(suffix)} events, "
            f"Kept-gap: {len(kept_gap)} events, Target: {target_bars} bars "
            f"(window bars {gap_bar_start}..{gap_bar_start + target_bars - 1})")

        fill_events = model.generate_infill(
            prefix_events=prefix,
            suffix_events=suffix,
            header_events=header,
            target_bars=target_bars,
            gap_bar_start=gap_bar_start,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            device=device,
            kept_gap_events=kept_gap,
            target_parts=target_parts,
        )

        log(f"  Generated {len(fill_events)} events ({count_bars(fill_events)} bars)")

        merge_fill_into_score(score, fill_events, gap_start, gap_end, target_parts=target_parts)
        log(f"  Merged fill content into measures {gap_start}-{gap_end}")

    # Export results
    OUTPUT_DIR.mkdir(exist_ok=True)
    base_name = "final_filled"

    midi_path = str(OUTPUT_DIR / f"{base_name}.mid")
    xml_out_path = str(OUTPUT_DIR / f"{base_name}.xml")

    try:
        for part in score.parts:
            part.makeAccidentals(inPlace=True)
            part.makeBeams(inPlace=True)
    except Exception as e:
        log(f"makeBeams/makeAccidentals skipped: {e}")

    score.write("midi", fp=midi_path)
    log(f"Saved MIDI -> {midi_path}")

    try:
        score.write("musicxml", fp=xml_out_path)
        log(f"Saved MusicXML -> {xml_out_path}")
    except Exception as e:
        log(f"MusicXML export skipped: {e}")
        xml_out_path = ""

    note_count = sum(
        1 for el in score.flatten().notesAndRests
        if el.isNote or el.isChord
    )
    measure_count = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    log(f"Filled score: {measure_count} measures, {note_count} notes/chords")

    return {"midi": midi_path, "xml": xml_out_path}


def infill_and_stream(
    score,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    gaps: list[tuple[int, int]] | None = None,
    target_parts: list[int] | None = None,
) -> Generator[tuple[int, str, int], None, None]:
    """Stream partial MusicXML as each fill measure is generated.

    Yields (bar_index, xml_str, total_bars) after each completed fill measure.
    Processes gaps left-to-right, streaming within each gap.
    """
    if gaps is None:
        gaps = identify_empty_measures(score)
    if not gaps:
        log("No empty measures detected.")
        return

    total_fill_bars = sum(end - start + 1 for start, end in gaps)
    log(f"Detected {len(gaps)} gap(s): {gaps}, total fill bars: {total_fill_bars}")

    log("Loading infill model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    filled_bars = 0
    working_score = copy.deepcopy(score)

    for i, (gap_start, gap_end) in enumerate(gaps):
        log(f"Processing gap {i+1}/{len(gaps)}: measures {gap_start}-{gap_end}")
        prefix, suffix, header, target_bars, kept_gap, gap_bar_start = \
            extract_prefix_suffix(working_score, gap_start, gap_end, target_parts=target_parts)

        last_snapshot: list[list[int]] = []

        for snapshot in model.generate_infill_stream(
            prefix_events=prefix,
            suffix_events=suffix,
            header_events=header,
            target_bars=target_bars,
            gap_bar_start=gap_bar_start,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            device=device,
            kept_gap_events=kept_gap,
            target_parts=target_parts,
        ):
            last_snapshot = snapshot
            bar_count = count_bars(snapshot)

            partial_score = copy.deepcopy(working_score)
            merge_fill_into_score(partial_score, snapshot, gap_start, gap_end, target_parts=target_parts)
            try:
                xml_str = score_to_xml_str(partial_score)
                log(f"Infill stream: gap {i+1} bar {bar_count} of {target_bars}")
                yield (filled_bars + bar_count, xml_str, total_fill_bars)
            except Exception as e:
                log(f"Infill stream XML error: {e}")

        # Final emit for this gap with complete last measure
        final_score = copy.deepcopy(working_score)
        merge_fill_into_score(final_score, last_snapshot, gap_start, gap_end, target_parts=target_parts)
        try:
            xml_str = score_to_xml_str(final_score)
            filled_bars += gap_end - gap_start + 1
            yield (filled_bars, xml_str, total_fill_bars)
        except Exception as e:
            log(f"Infill final stream XML error: {e}")

        # Update working_score so subsequent gaps see filled content
        merge_fill_into_score(working_score, last_snapshot, gap_start, gap_end, target_parts=target_parts)


def create_test_partial_xml(output_path: str, empty_measures: list[int] = None):
    """Create a test MusicXML file with some empty measures for testing."""
    import music21 as m21

    if empty_measures is None:
        empty_measures = [3, 4]

    score = m21.stream.Score()
    part = m21.stream.Part()

    for m_num in range(1, 9):
        measure = m21.stream.Measure(number=m_num)

        if m_num not in empty_measures:
            if m_num % 2 == 1:
                for i, name in enumerate(['C4', 'E4', 'G4', 'C5']):
                    measure.insert(i, m21.note.Note(name, quarterLength=1))
            else:
                measure.insert(0, m21.note.Note('D4', quarterLength=2))
                measure.insert(2, m21.note.Note('F4', quarterLength=2))
        else:
            rest = m21.note.Rest(quarterLength=4)
            rest.style.hideObjectOnPrint = True
            measure.insert(0, rest)

        part.append(measure)

    score.append(part)
    score.write("musicxml", output_path)

    return output_path
