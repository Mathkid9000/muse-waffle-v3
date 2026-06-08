"""Token-histogram similarity between a generated score and the dataset."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Callable

import music21 as m21

from src.tokenizer import (
    score_to_tokens,
    NOTE_OFF, DUR_OFF, REST_OFF, BAR,
)
from src.dataset import load_cached_paths


def _token_histogram(tokens: list[int]) -> dict[str, float]:
    pitch_counts: Counter = Counter()
    dur_counts:   Counter = Counter()
    rest_counts:  Counter = Counter()
    note_total = rest_total = bar_count = 0

    for t in tokens:
        if t == BAR:
            bar_count += 1
        elif NOTE_OFF <= t < NOTE_OFF + 128:
            pitch_counts[t - NOTE_OFF] += 1
            note_total += 1
        elif DUR_OFF <= t < DUR_OFF + 16:
            dur_counts[t - DUR_OFF] += 1
        elif REST_OFF <= t < REST_OFF + 16:
            rest_counts[t - REST_OFF] += 1
            rest_total += 1

    vec: dict[str, float] = {}
    if note_total:
        for midi, cnt in pitch_counts.items():
            vec[f"p{midi}"] = cnt / note_total
    rhythm_total = note_total + rest_total
    if rhythm_total:
        for idx, cnt in dur_counts.items():
            vec[f"d{idx}"] = cnt / rhythm_total
        for idx, cnt in rest_counts.items():
            vec[f"r{idx}"] = cnt / rhythm_total
        vec["rest_ratio"] = rest_total / rhythm_total
    if bar_count:
        vec["bar_density"] = note_total / bar_count
    return vec


def _chroma_histogram(tokens: list[int]) -> dict[int, float]:
    counts: Counter = Counter()
    total = 0
    for t in tokens:
        if NOTE_OFF <= t < NOTE_OFF + 128:
            counts[(t - NOTE_OFF) % 12] += 1
            total += 1
    if not total:
        return {}
    return {pc: cnt / total for pc, cnt in counts.items()}


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na  = math.sqrt(sum(v * v for v in a.values()))
    nb  = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def compare_to_dataset(
    xml_path: str,
    log: Callable[[str], None],
    top_n: int = 5,
) -> None:
    """Load xml_path, score it against the cached dataset, log top_n matches."""
    dataset_paths = load_cached_paths()
    if not dataset_paths:
        log("[yellow]No dataset cached — skipping similarity comparison.[/yellow]")
        return

    try:
        gen_score  = m21.converter.parse(xml_path)
        gen_tokens = score_to_tokens(gen_score)
    except Exception as e:
        log(f"[yellow]Similarity: could not parse generated score: {e}[/yellow]")
        return

    gen_vec    = _token_histogram(gen_tokens)
    gen_chroma = _chroma_histogram(gen_tokens)

    log(f"Comparing against {len(dataset_paths)} dataset scores…")

    results: list[tuple[float, float, float, str]] = []
    for path in dataset_paths:
        if not Path(path).exists():
            continue
        try:
            score  = m21.converter.parse(path)
            tokens = score_to_tokens(score)
            tok_sim    = _cosine(gen_vec,    _token_histogram(tokens))
            chroma_sim = _cosine(gen_chroma, _chroma_histogram(tokens))
            combined   = 0.6 * tok_sim + 0.4 * chroma_sim
            results.append((combined, tok_sim, chroma_sim, path))
        except Exception:
            pass

    if not results:
        log("[yellow]No dataset scores could be compared.[/yellow]")
        return

    results.sort(reverse=True)

    log("-- Dataset similarity (top matches) --")
    log(f"  {'Score':>7}  {'Token':>7}  {'Chroma':>7}  Title")
    for combined, tok_sim, chroma_sim, path in results[:top_n]:
        name = Path(path).stem.replace("scores_", "")
        # trim composer prefix: "Composer___Title" → "Title"
        if "___" in name:
            name = name.split("___", 1)[1]
        name = name[:50]
        log(f"  {combined:>7.4f}  {tok_sim:>7.4f}  {chroma_sim:>7.4f}  {name}")

    avg = sum(r[0] for r in results) / len(results)
    log(f"  mean={avg:.4f}  max={results[0][0]:.4f}  min={results[-1][0]:.4f}  (n={len(results)})")
