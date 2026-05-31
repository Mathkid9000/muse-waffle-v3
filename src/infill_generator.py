"""Inference pipeline for fill-in-the-middle music generation.

This module handles:
1. Loading partial MusicXML files with empty measures
2. Detecting gaps (empty regions)
3. Generating fill content using the trained infill model
4. Merging generated content back into the score
5. Exporting completed MIDI and MusicXML
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable

from src.tokenizer import (
    identify_empty_measures, extract_prefix_suffix_tokens,
    tokens_to_score, score_to_tokens_by_measure, score_to_header_tokens,
    BAR, BOS, EOS
)
from src.model import load_model

OUTPUT_DIR = Path("output")

def load_partial_xml(xml_path: str) -> tuple:
    """Parse input XML and identify gaps.

    Args:
        xml_path: Path to MusicXML file with empty measures

    Returns:
        (score, gap_descriptors)
        - score: music21 Score object
        - gap_descriptors: List of (start_measure, end_measure) tuples
    """
    import music21 as m21

    score = m21.converter.parse(xml_path)
    gaps = identify_empty_measures(score)

    return score, gaps


def extract_prefix_suffix(
    score,
    gap_start: int,
    gap_end: int,
) -> tuple[list[int], list[int], list[int], int]:
    """Extract prefix, suffix, and header tokens around a gap.

    Args:
        score: music21 Score object
        gap_start: First measure of gap (inclusive)
        gap_end: Last measure of gap (inclusive)

    Returns:
        (prefix_tokens, suffix_tokens, header_tokens, target_bars)
        - prefix_tokens: Content before gap
        - suffix_tokens: Content after gap
        - header_tokens: Instrument header for this score
        - target_bars: Number of bars to generate (gap size)
    """
    prefix, suffix = extract_prefix_suffix_tokens(score, gap_start, gap_end)
    header = score_to_header_tokens(score)
    target_bars = gap_end - gap_start + 1

    return prefix, suffix, header, target_bars


def merge_fill_into_score(
    score,
    fill_tokens: list[int],
    gap_start: int,
    gap_end: int,
    color='red'
):
    """Insert generated tokens back into the score at correct measure positions.

    This modifies the score in-place by:
    1. Converting fill_tokens to a temporary score
    2. Extracting notes from each generated measure
    3. Inserting them into the corresponding empty measures in the original

    Args:
        score: Original music21 Score (modified in-place)
        fill_tokens: Generated tokens for the gap region
        gap_start: First measure of gap
        gap_end: Last measure of gap
    """
    import music21 as m21

    # Convert fill tokens to a temporary score
    fill_score = tokens_to_score([BOS] + fill_tokens + [EOS])

    xml_out_path = str(OUTPUT_DIR / f"filled_test.xml")

    try:
        fill_score.write("musicxml", fp=xml_out_path)
        print(f"Saved MusicXML -> {xml_out_path}")
    except Exception as e:
        print(f"[yellow]MusicXML export skipped: {e}[/yellow]")
        xml_out_path = ""
    
    if not fill_score.parts:
        return

    # Get measures from the fill score
    fill_measures = {}
    for part_idx, part in enumerate(fill_score.parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 1
            if part_idx not in fill_measures:
                fill_measures[part_idx] = {}
            fill_measures[part_idx][m_num] = measure

    # Map generated measure numbers to target gap positions
    # Generated measures are numbered 1, 2, 3... but we need gap_start, gap_start+1, etc.
    target_measures = list(range(gap_start, gap_end + 1))

    # Insert into original score
    for part_idx, part in enumerate(score.parts):
        if part_idx not in fill_measures:
            continue

        generated_nums = sorted(fill_measures[part_idx].keys())

        for i, target_m_num in enumerate(target_measures):
            if i >= len(generated_nums):
                break

            gen_m_num = generated_nums[i]
            gen_measure = fill_measures[part_idx][gen_m_num]

            # Find the target measure in the original score
            for measure in part.getElementsByClass(m21.stream.Measure):
                m_num = measure.number if measure.number is not None else 0
                if m_num == target_m_num:
                    # Clear existing content and copy from generated
                    measure.elements = []

                    # Copy all elements from generated measure
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


def infill_and_export(
    score,
    max_new: int = 500,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
) -> dict[str, str]:
    """Main entry point for infill generation.

    Processes gaps left-to-right. After filling each gap, the filled content
    becomes part of the prefix for subsequent gaps.

    Args:
        xml_path: Path to MusicXML file with empty measures
        max_new: Maximum tokens to generate per gap
        temperature: Sampling temperature
        top_k: Top-k sampling
        log: Logging function
        device: Device for inference

    Returns:
        Dict with paths to output files: {"midi": ..., "xml": ...}
    """
    import torch

    # log(f"Loading partial XML: {xml_path}")
    # score, gaps = load_partial_xml(xml_path)
    gaps = identify_empty_measures(score)
    
    if not gaps:
        log("No empty measures detected. Nothing to fill.")
        return {"midi": "", "xml": ""}

    log(f"Detected {len(gaps)} gap(s): {gaps}")

    # Load infill model
    log(f"Loading infill model...")
    model, _ = load_model(device)

    log(f"Model loaded ({model.num_params():,} params)")

    # Process gaps left-to-right
    for i, (gap_start, gap_end) in enumerate(gaps):
        log(f"Processing gap {i+1}/{len(gaps)}: measures {gap_start}-{gap_end}")

        # Extract prefix, suffix and instrument header
        prefix, suffix, header, target_bars = extract_prefix_suffix(score, gap_start, gap_end)
        log(f"  Prefix: {len(prefix)} tokens, Suffix: {len(suffix)} tokens, Header: {header}, Target: {target_bars} bars")

        # Generate fill content
        fill_tokens = model.generate_infill(
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            header_tokens=header,
            target_bars=target_bars,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        generated_bars = fill_tokens.count(BAR)
        log(f"  Generated {len(fill_tokens)} tokens ({generated_bars} bars)")
        log(f"  Generated: {fill_tokens}")

        # Merge into score
        merge_fill_into_score(score, fill_tokens, gap_start, gap_end)
        log(f"  Merged fill content into measures {gap_start}-{gap_end}")

    # Export results
    OUTPUT_DIR.mkdir(exist_ok=True)
    base_name = "final_filled"

    midi_path = str(OUTPUT_DIR / f"{base_name}.mid")
    xml_out_path = str(OUTPUT_DIR / f"{base_name}.xml")

    score.write("midi", fp=midi_path)
    log(f"Saved MIDI -> {midi_path}")

    try:
        score.write("musicxml", fp=xml_out_path)
        log(f"Saved MusicXML -> {xml_out_path}")
    except Exception as e:
        log(f"[yellow]MusicXML export skipped: {e}[/yellow]")
        xml_out_path = ""

    # Summary
    note_count = sum(
        1 for el in score.flatten().notesAndRests
        if el.isNote or el.isChord
    )
    measure_count = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    log(f"Filled score: {measure_count} measures, {note_count} notes/chords")

    return {"midi": midi_path, "xml": xml_out_path}


def create_test_partial_xml(output_path: str, empty_measures: list[int] = None):
    """Create a test MusicXML file with some empty measures for testing.

    Args:
        output_path: Where to save the test file
        empty_measures: List of measure numbers to leave empty (default: [3, 4])
    """
    import music21 as m21

    if empty_measures is None:
        empty_measures = [3, 4]

    # Create a simple score with 8 measures
    score = m21.stream.Score()
    part = m21.stream.Part()

    for m_num in range(1, 9):
        measure = m21.stream.Measure(number=m_num)

        if m_num not in empty_measures:
            # Add some content
            if m_num % 2 == 1:
                n1 = m21.note.Note('C4', quarterLength=1)
                n2 = m21.note.Note('E4', quarterLength=1)
                n3 = m21.note.Note('G4', quarterLength=1)
                n4 = m21.note.Note('C5', quarterLength=1)
                measure.insert(0, n1)
                measure.insert(1, n2)
                measure.insert(2, n3)
                measure.insert(3, n4)
            else:
                n1 = m21.note.Note('D4', quarterLength=2)
                n2 = m21.note.Note('F4', quarterLength=2)
                measure.insert(0, n1)
                measure.insert(2, n2)
        # Empty measures are left with no content
        else:
            rest = m21.note.Rest(quarterLength=4)
            rest.style.hideObjectOnPrint = True
            measure.insert(0, rest)

        part.append(measure)

    score.append(part)
    score.write("musicxml", output_path)

    return output_path
