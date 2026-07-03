"""Tests for vertical-aware partial-staff infill over compound events.

Run: python test_partial_infill.py

Covers (no trained model required):
  1. Tokenizer part_filter + gap_measure_events (BAR emitted even for an empty
     staff; bar indices window-relative).
  2. create_fim_example partial/full format (FILL_START/FILL_END/FILL_GEN,
     loss_start, kept/target staff disjointness, bar-index continuity).
  3. merge_fill_into_score: generated target staff lands in the right part and
     the kept staff is left byte-identical (guards the part-index mapping bug).
"""

from __future__ import annotations
import random

import music21 as m21

from ml.events import (
    F_TYPE, F_BAR, F_TRACK,
    BAR_T, NOTE_T, GRACE_T, REST_T,
    FILL_START_T, FILL_END_T, FILL_GEN_T,
    MAX_VOICES_PER_PART, MAX_REAL_PARTS,
)
from ml.tokenizer import (
    collect_score, emit_measure_events, flatten_measure_events,
    gap_measure_events, score_to_header_events,
)
from ml.infill_trainer import create_fim_example
from ml.infill_generator import merge_fill_into_score


def build_score(n_parts=2, n_measures=6, empty=None):
    """Distinct content per staff/measure; `empty` = set of (part_idx, m_num) left empty."""
    empty = empty or set()
    score = m21.stream.Score()
    for pi in range(n_parts):
        part = m21.stream.Part()
        part.insert(0, m21.instrument.Piano())
        for m_num in range(1, n_measures + 1):
            meas = m21.stream.Measure(number=m_num)
            if (pi, m_num) not in empty:
                base = 48 + pi * 12 + m_num
                for b in range(4):
                    meas.insert(b, m21.note.Note(midi=base + b, quarterLength=1))
            part.append(meas)
        score.append(part)
    return score


def parts_in(rows):
    return sorted({(r[F_TRACK] - 1) // MAX_VOICES_PER_PART
                   for r in rows if r[F_TYPE] in (NOTE_T, GRACE_T, REST_T)})


def n_bars(rows):
    return sum(1 for r in rows if r[F_TYPE] == BAR_T)


def type_idx(rows, t):
    return next(i for i, r in enumerate(rows) if r[F_TYPE] == t)


def measure_pitches(score, part_idx, m_num):
    part = list(score.parts)[part_idx]
    for meas in part.getElementsByClass(m21.stream.Measure):
        if (meas.number or 0) == m_num:
            return [(round(float(n.offset), 4), n.pitch.midi)
                    for n in meas.flatten().notes]
    return None


def test_tokenizer_part_filter():
    # part 0 empty in measure 3 -> gap builder must still emit a BAR there.
    score = build_score(n_parts=2, n_measures=4, empty={(0, 3)})
    collected = collect_score(score)

    blocks_p1 = emit_measure_events(collected, part_filter={1})
    for m in (1, 2, 3, 4):
        assert parts_in(blocks_p1[m]) == [1], f"part_filter leaked: m{m} {parts_in(blocks_p1[m])}"

    g = gap_measure_events(collected, 3, 3, part_filter={0}, start_bar=2)
    assert n_bars(g) == 1 and parts_in(g) == [], f"empty-staff gap not a bare BAR: {g}"
    assert g[0][F_BAR] == 2, f"gap bar index not window-relative: {g[0]}"

    g2 = gap_measure_events(collected, 2, 4, part_filter={1}, start_bar=1)
    assert n_bars(g2) == 3 and parts_in(g2) == [1]
    assert [r[F_BAR] for r in g2 if r[F_TYPE] == BAR_T] == [1, 2, 3]
    print("test_tokenizer_part_filter: OK")


def test_fim_format():
    score = build_score(n_parts=2, n_measures=6)
    header = score_to_header_events(score)
    collected = collect_score(score)
    measure_events = emit_measure_events(collected)
    n_parts = len(list(score.parts)[:MAX_REAL_PARTS])

    random.seed(0)
    # partial
    seq = None
    for _ in range(20):
        res = create_fim_example(header, measure_events, collected, n_parts,
                                 partial_ratio=1.0, max_len=1024)
        if res:
            seq, ls = res
            break
    assert seq is not None
    for t in (FILL_START_T, FILL_END_T, FILL_GEN_T):
        cnt = sum(1 for r in seq if r[F_TYPE] == t)
        assert cnt == 1, f"marker type {t} count {cnt}"
    assert seq[ls - 1][F_TYPE] == FILL_GEN_T, "loss_start not immediately after FILL_GEN"

    fs, fe, fg = (type_idx(seq, t) for t in (FILL_START_T, FILL_END_T, FILL_GEN_T))
    prefix, suffix = seq[:fs], seq[fs + 1:fe]
    kept, target = seq[fe + 1:fg], seq[ls:-1]

    assert set(parts_in(kept)).isdisjoint(parts_in(target)), "kept/target staff overlap"
    assert parts_in(kept) and parts_in(target), "partial example missing kept or target staff"

    # Bar-index continuity: prefix 0..P-1, gap P..P+G-1 (kept == target), suffix P+G.. .
    p_bars = [r[F_BAR] for r in prefix if r[F_TYPE] == BAR_T]
    s_bars = [r[F_BAR] for r in suffix if r[F_TYPE] == BAR_T]
    k_bars = [r[F_BAR] for r in kept if r[F_TYPE] == BAR_T]
    t_bars = [r[F_BAR] for r in target if r[F_TYPE] == BAR_T]
    P, G = len(p_bars), len(t_bars)
    assert p_bars == list(range(P)), f"prefix bars {p_bars}"
    assert k_bars == t_bars == list(range(P, P + G)), f"gap bars kept={k_bars} target={t_bars}"
    assert s_bars == list(range(P + G, P + G + len(s_bars))), f"suffix bars {s_bars}"

    # full-staff: kept region empty, single FILL_GEN
    res2 = create_fim_example(header, measure_events, collected, n_parts, partial_ratio=0.0)
    seq2, ls2 = res2
    fe2, fg2 = type_idx(seq2, FILL_END_T), type_idx(seq2, FILL_GEN_T)
    assert seq2[fe2 + 1:fg2] == [], "full-staff kept region should be empty"
    print("test_fim_format: OK")


def test_merge_maps_to_correct_part():
    score = build_score(n_parts=2, n_measures=6)
    collected = collect_score(score)

    # Snapshot the KEPT staff (part 0) in the gap before any mutation.
    snap_p0 = {m: measure_pitches(score, 0, m) for m in (3, 4)}

    # Pretend-model output: part 1's real content for the gap (target staff only).
    fill_events = gap_measure_events(collected, 3, 4, part_filter={1}, start_bar=2)
    assert parts_in(fill_events) == [1]

    # Simulate the API: blank only part 1 in the gap, then merge the generated content.
    from api.routers.infill import _mark_measures_empty
    _mark_measures_empty(score, 3, 4, parts=[1])
    assert measure_pitches(score, 1, 3) == [], "blanking failed"

    merge_fill_into_score(score, fill_events, 3, 4, target_parts=[1])

    # Kept staff (part 0) must be byte-identical in the gap.
    for m in (3, 4):
        assert measure_pitches(score, 0, m) == snap_p0[m], \
            f"kept staff changed at m{m}: {measure_pitches(score,0,m)} != {snap_p0[m]}"

    # Target staff (part 1) must now have notes again, colored red.
    for m in (3, 4):
        notes = list(list(score.parts)[1].getElementsByClass(m21.stream.Measure))
        meas = [x for x in notes if (x.number or 0) == m][0]
        ns = list(meas.flatten().notes)
        assert len(ns) > 0, f"target staff empty at m{m}"
        assert all(n.style.color == 'red' for n in ns), f"target notes not colored at m{m}"
    print("test_merge_maps_to_correct_part: OK")


if __name__ == "__main__":
    test_tokenizer_part_filter()
    test_fim_format()
    test_merge_maps_to_correct_part()
    print("\nALL TESTS PASSED")
