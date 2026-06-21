"""Inference pipeline for pure autoregressive music generation from scratch."""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Generator

from ml.tokenizer import tokens_to_score, score_to_xml_str, BOS, EOS, BAR, INSTR_OFF, CLEF_OFF
from ml.model import load_model

OUTPUT_DIR = Path("output")

# Instrument indices that use two staves (treble + bass).
_KEYBOARD_INSTRS = {1, 2, 3, 4}  # piano, harpsichord, organ, accordion


def _expand_instr_indices(instr_indices: list[int]) -> list[int]:
    """Expand keyboard instruments to two entries (one per staff).

    A single piano entry becomes two entries so the model receives separate
    INSTR+CLEF header tokens for the treble and bass staves.
    """
    expanded: list[int] = []
    for idx in instr_indices:
        expanded.append(idx)
        if idx in _KEYBOARD_INSTRS:
            expanded.append(idx)  # second staff (bass)
    return expanded


def _sample_header_clefs(model, instr_indices: list[int], temperature: float, top_k: int, device: str) -> list[int]:
    """Given instrument indices, build a full header by letting the model sample a CLEF
    token after each INSTR token, as it was trained to do.

    Returns interleaved [INSTR_0, CLEF_0, INSTR_1, CLEF_1, ...] header tokens.
    """
    import torch

    model.eval()
    max_ctx = model.cfg["max_seq_len"]
    idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
    header_tokens: list[int] = []

    instr_indices = sorted(instr_indices, reverse=True)

    with torch.no_grad():
        for instr_idx in instr_indices:
            instr_tok = INSTR_OFF + instr_idx
            idx = torch.cat([idx, torch.tensor([[instr_tok]], dtype=torch.long, device=device)], dim=1)
            header_tokens.append(instr_tok)

            # Sample the next token; the model should predict a CLEF token here.
            logits = model(idx[:, -max_ctx:])[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            clef_tok = next_t.item()

            # Clamp to valid CLEF range in case the model predicts something unexpected.
            if not (CLEF_OFF <= clef_tok < CLEF_OFF + 7):
                clef_tok = CLEF_OFF  # fall back to treble

            header_tokens.append(clef_tok)
            idx = torch.cat([idx, torch.tensor([[clef_tok]], dtype=torch.long, device=device)], dim=1)

    return header_tokens


def generate_and_export(
    instr_indices: list[int],
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
        instr_indices: Instrument category indices (0-31, one per part).
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

    expanded = _expand_instr_indices(instr_indices)
    header_tokens = _sample_header_clefs(model, expanded, temperature, top_k, device)
    log(f"Expanded instruments: {expanded}, sampled header: {header_tokens}")

    forced = first_bar_prefix if first_bar_prefix is not None else [BAR]

    generated = model.generate_standard(
        header_tokens=header_tokens,
        first_bar_prefix=forced,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
        max_parts=len(expanded),
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


def generate_and_stream(
    instr_indices: list[int],
    first_bar_prefix: list[int] | None = None,
    target_bars: int = 16,
    max_new: int = 2000,
    temperature: float = 0.9,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    device: str = "cpu",
    model_name: str = "GEN-D1024-H8-L8",
) -> Generator[tuple[int, str, int], None, None]:
    """Stream partial MusicXML as each measure is generated.

    Yields (bar_index, xml_str, total_bars) after each completed measure.
    bar_index counts total BARs visible in the partial score (including forced prefix).
    """
    log("Loading generation model...")
    model, _ = load_model(model_name, device)
    log(f"Model loaded ({model.num_params():,} params)")

    expanded = _expand_instr_indices(instr_indices)
    header_tokens = _sample_header_clefs(model, expanded, temperature, top_k, device)
    log(f"Expanded instruments: {expanded}, sampled header: {header_tokens}")

    forced = first_bar_prefix if first_bar_prefix is not None else [BAR]
    n_forced_bars = forced.count(BAR)
    last_snapshot: list[int] = []

    for snapshot in model.generate_standard_stream(
        header_tokens=header_tokens,
        first_bar_prefix=forced,
        target_bars=target_bars,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device,
        max_parts=len(expanded),
    ):
        last_snapshot = snapshot
        bar_count = n_forced_bars + snapshot.count(BAR)

        full_tokens = [BOS] + header_tokens + forced + snapshot + [EOS]
        score = tokens_to_score(full_tokens)
        xml_str = score_to_xml_str(score)
        log(f"Streamed bar {bar_count} of {target_bars}")
        yield (bar_count, xml_str, target_bars, len(header_tokens) // 2)

    # Always emit the final complete score (last measure now has its notes)
    full_tokens = [BOS] + header_tokens + forced + last_snapshot + [EOS]
    try:
        score = tokens_to_score(full_tokens)
        xml_str = score_to_xml_str(score)
        final_bar = n_forced_bars + last_snapshot.count(BAR)
        log(f"Stream complete: {final_bar} bars")
        yield (final_bar, xml_str, target_bars, len(header_tokens) // 2)
    except Exception as e:
        log(f"Final stream XML export error: {e}")
