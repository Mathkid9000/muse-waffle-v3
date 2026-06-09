"""Inference pipeline for pure autoregressive music generation from scratch."""

from __future__ import annotations
from pathlib import Path
from typing import Callable

from ml.tokenizer import tokens_to_score, BOS, EOS, BAR
from ml.model import load_model

OUTPUT_DIR = Path("output")


def generate_and_export(
    header_tokens: list[int],
    first_bar_prefix: list[int] | None = None,
    target_bars: int = 16,
    max_new: int = 2000,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = "GEN-D1024-H8-L8",
) -> dict[str, str]:
    """Generate a new score from scratch using the generation model.

    Args:
        header_tokens: Instrument header tokens (from score_to_header_tokens)
        first_bar_prefix: Tokens to force after the header, e.g.
            [BAR, KEY_SIG_token, TIME_SIG_token, TEMPO_token]. Defaults to [BAR].
        target_bars: Total number of measures to generate
        max_new: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        log: Logging callback
        device: Device for inference

    Returns:
        Dict with paths to output files: {"midi": ..., "xml": ...}
    """
    log("Loading generation model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    forced = first_bar_prefix if first_bar_prefix is not None else [BAR]

    generated = model.generate_standard(
        header_tokens=header_tokens,
        first_bar_prefix=forced,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    full_tokens = [BOS] + header_tokens + forced + generated + [EOS]
    score = tokens_to_score(full_tokens)

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
