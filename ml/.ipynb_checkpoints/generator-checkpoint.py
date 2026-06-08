"""Generates music from a text prompt and exports MIDI + MusicXML."""

from __future__ import annotations
from pathlib import Path
from typing import Callable

from src.tokenizer import text_to_tokens, tokens_to_score, BOS
from src.model import load_model

OUTPUT_DIR = Path("output")


def generate_and_export(
    prompt: str,
    max_new: int,
    temperature: float,
    top_k: int,
    log: Callable[[str], None],
    device: str = "cpu",
) -> dict[str, str]:
    log(f"Loading model on {device} (generation always runs on CPU to avoid GPU crashes)...")
    device = "cpu"
    model, _ = load_model(device)
    log(f"Model loaded ({model.num_params():,} params).")

    prefix = [BOS] + text_to_tokens(prompt)
    log(f"Prompt tokens: {len(prefix)} | Generating up to {max_new} music tokens...")
    tokens = model.generate(
        prefix, max_new=max_new, temperature=temperature, top_k=top_k, device=device
    )
    log(f"Generated {len(tokens)} tokens total.")

    log("Converting tokens → music21 score...")
    score = tokens_to_score(tokens)

    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "_ " else "_" for c in prompt)[:40].strip()
    safe_name = safe_name.replace(" ", "_") or "output"

    midi_path = str(OUTPUT_DIR / f"{safe_name}.mid")
    xml_path  = str(OUTPUT_DIR / f"{safe_name}.xml")

    score.write("midi", fp=midi_path)
    log(f"Saved MIDI → {midi_path}")

    try:
        score.write("musicxml", fp=xml_path)
        log(f"Saved MusicXML → {xml_path}")
    except Exception as e:
        log(f"[yellow]MusicXML export skipped: {e}[/yellow]")
        xml_path = None

    note_count = sum(
        1 for el in score.flatten().notesAndRests
        if el.isNote or el.isChord
    )
    measure_count = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    log(f"Score summary: {measure_count} measures, {note_count} notes/chords.")

    return {"midi": midi_path, "xml": xml_path or ""}


def show_text_score(xml_path: str, log: Callable[[str], None]):
    """Print a simplified text representation of the score, one line per part per measure."""
    import music21 as m21
    score = m21.converter.parse(xml_path)
    parts = score.parts if score.parts else [score]
    log(f"  {len(parts)} part(s):")
    for p_idx, part in enumerate(parts):
        measures = list(part.getElementsByClass(m21.stream.Measure))
        log(f"  Part {p_idx}  ({len(measures)} measures)")
        for measure in measures[:8]:           # preview first 8 measures
            notes_str = []
            for el in measure.flatten().notesAndRests:
                if el.isRest:
                    notes_str.append(f"R({el.duration.type})")
                elif el.isChord:
                    pits = "+".join(str(p) for p in el.pitches)
                    notes_str.append(f"[{pits}]({el.duration.type})")
                elif el.isNote:
                    notes_str.append(f"{el.nameWithOctave}({el.duration.type})")
            log(f"    m{measure.number}: {' '.join(notes_str) or '(empty)'}")
