"""Tests for vertical-aware partial-staff infill.

Run: python test_partial_infill.py

Covers (no trained model required):
  1. Tokenizer part_filter + gap_measure_tokens (BAR emitted even for empty staff).
  2. create_fim_example partial/full format (FILL_START/FILL_END/FILL_GEN, loss_start).
  3. merge_fill_into_score: generated target staff lands in the right part and the
     kept staff is left byte-identical (guards the part-index mapping bug).
"""

from __future__ import annotations
import random

import music21 as m21

import ml.tokenizer as t
from ml.tokenizer import (
    _collect_events_by_measure, _emit_measure_tokens, gap_measure_tokens,
    score_to_tokens_by_measure, score_to_header_tokens,
    MAX_REAL_PARTS, MAX_VOICES_PER_PART, PART_OFF, MAX_PARTS,
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


def parts_in(toks):
    return sorted({(x - PART_OFF) // MAX_VOICES_PER_PART
                   for x in toks if PART_OFF <= x < PART_OFF + MAX_PARTS})


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

    mt_p1 = score_to_tokens_by_measure(score, part_filter={1})
    for m in (1, 2, 3, 4):
        assert parts_in(mt_p1[m]) == [1], f"part_filter leaked: m{m} {parts_in(mt_p1[m])}"

    g = gap_measure_tokens(score, 3, 3, part_filter={0})
    assert g.count(t.BAR) == 1 and parts_in(g) == [], f"empty-staff gap not a bare BAR: {g}"

    g2 = gap_measure_tokens(score, 2, 4, part_filter={1})
    assert g2.count(t.BAR) == 3 and parts_in(g2) == [1]
    print("test_tokenizer_part_filter: OK")


def test_fim_format():
    score = build_score(n_parts=2, n_measures=6)
    header = score_to_header_tokens(score)
    collected = _collect_events_by_measure(score)
    measure_tokens = _emit_measure_tokens(*collected)
    n_parts = len(list(score.parts)[:MAX_REAL_PARTS])

    random.seed(0)
    # partial
    seq = None
    for _ in range(20):
        res = create_fim_example(header, measure_tokens, collected, n_parts,
                                 partial_ratio=1.0, max_len=2048)
        if res:
            seq, ls = res
            break
    assert seq is not None
    for tok in (t.FILL_START, t.FILL_END, t.FILL_GEN):
        assert seq.count(tok) == 1, f"marker {tok} count {seq.count(tok)}"
    assert seq[ls - 1] == t.FILL_GEN, "loss_start not immediately after FILL_GEN"
    fe, fg = seq.index(t.FILL_END), seq.index(t.FILL_GEN)
    kept, target = seq[fe + 1:fg], seq[ls:]
    assert set(parts_in(kept)).isdisjoint(parts_in(target)), "kept/target staff overlap"
    assert parts_in(kept) and parts_in(target), "partial example missing kept or target staff"

    # full-staff: kept region empty, single FILL_GEN
    res2 = create_fim_example(header, measure_tokens, collected, n_parts, partial_ratio=0.0)
    seq2, ls2 = res2
    fe2, fg2 = seq2.index(t.FILL_END), seq2.index(t.FILL_GEN)
    assert seq2[fe2 + 1:fg2] == [], "full-staff kept region should be empty"
    print("test_fim_format: OK")


def test_merge_maps_to_correct_part():
    score = build_score(n_parts=2, n_measures=6)

    # Snapshot the KEPT staff (part 0) in the gap before any mutation.
    snap_p0 = {m: measure_pitches(score, 0, m) for m in (3, 4)}

    # Pretend-model output: part 1's real content for the gap (target staff only).
    fill_tokens = gap_measure_tokens(score, 3, 4, part_filter={1})
    assert parts_in(fill_tokens) == [1]

    # Simulate the API: blank only part 1 in the gap, then merge the generated content.
    from api.routers.infill import _mark_measures_empty
    _mark_measures_empty(score, 3, 4, parts=[1])
    assert measure_pitches(score, 1, 3) == [], "blanking failed"

    merge_fill_into_score(score, fill_tokens, 3, 4, target_parts=[1])

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
