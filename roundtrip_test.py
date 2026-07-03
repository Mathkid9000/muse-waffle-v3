"""Round-trip tests for the compound-event tokenizer (MMT-BERT representation).

Covers (no dataset or trained model required):
  1. Schema invariants: every emitted row respects field vocab sizes and the
     FIELD_LOSS_MASK structural zeros.
  2. Encode -> decode -> re-encode event-level equality on a fixture score with
     two parts, two voices, ties, a grace note, rests, key/time/tempo changes.
  3. Transpose augmentation +s then -s identity.

Run: python roundtrip_test.py
"""

from __future__ import annotations
import music21 as m21

from ml.events import (
    F_TYPE, F_BAR, F_POS, F_PITCH, F_DUR, F_TRACK, F_VALUE, N_FIELDS,
    NOTE_T, GRACE_T, REST_T, CHORD_T, BAR_T, KEY_T, TIME_T, TEMPO_T,
    FIELD_VOCAB_SIZES, FIELD_LOSS_MASK, TYPE_NAMES,
    transpose_events,
)
from ml.tokenizer import (
    collect_score, emit_measure_events, flatten_measure_events,
    score_to_header_events, events_to_score,
    sos_event, eos_event,
)


def build_fixture() -> m21.stream.Score:
    score = m21.stream.Score()

    # Part 0: melody with tie, grace note, rest, and two voices in measure 3.
    p0 = m21.stream.Part()
    p0.insert(0, m21.instrument.Flute())

    m1 = m21.stream.Measure(number=1)
    m1.insert(0, m21.meter.TimeSignature("4/4"))
    m1.insert(0, m21.key.KeySignature(2))
    m1.insert(0, m21.tempo.MetronomeMark(number=100))
    m1.insert(0.0, m21.note.Note("C5", quarterLength=1.0))
    m1.insert(1.0, m21.note.Note("D5", quarterLength=1.0))
    m1.insert(2.0, m21.note.Rest(quarterLength=1.0))
    tied = m21.note.Note("E5", quarterLength=1.0)
    tied.tie = m21.tie.Tie("start")
    m1.insert(3.0, tied)
    p0.append(m1)

    m2 = m21.stream.Measure(number=2)
    tied2 = m21.note.Note("E5", quarterLength=2.0)
    tied2.tie = m21.tie.Tie("stop")
    m2.insert(0.0, tied2)
    grace = m21.note.Note("F5", quarterLength=1.0).getGrace()
    m2.insert(2.0, grace)
    m2.insert(2.0, m21.note.Note("G5", quarterLength=2.0))
    p0.append(m2)

    m3 = m21.stream.Measure(number=3)
    v1 = m21.stream.Voice(id="1")
    v1.insert(0.0, m21.note.Note("A5", quarterLength=2.0))
    v1.insert(2.0, m21.note.Note("B5", quarterLength=2.0))
    v2 = m21.stream.Voice(id="2")
    v2.insert(0.0, m21.note.Note("C5", quarterLength=4.0))
    m3.insert(0, v1)
    m3.insert(0, v2)
    p0.append(m3)

    m4 = m21.stream.Measure(number=4)
    m4.insert(0, m21.key.KeySignature(-1))       # key change
    ch = m21.chord.Chord(["C5", "E5", "G5"], quarterLength=4.0)
    m4.insert(0.0, ch)
    p0.append(m4)
    score.append(p0)

    # Part 1: bass line.
    p1 = m21.stream.Part()
    p1.insert(0, m21.instrument.Violoncello())
    for m_num in range(1, 5):
        meas = m21.stream.Measure(number=m_num)
        if m_num == 1:
            meas.insert(0, m21.clef.BassClef())
            meas.insert(0, m21.meter.TimeSignature("4/4"))
        for b in range(2):
            meas.insert(b * 2.0, m21.note.Note(midi=40 + m_num + b, quarterLength=2.0))
        p1.append(meas)
    score.append(p1)

    return score


def encode(score) -> list[list[int]]:
    header = score_to_header_events(score)
    collected = collect_score(score)
    blocks = emit_measure_events(collected)
    flat = flatten_measure_events(blocks)
    return [sos_event()] + header + flat + [eos_event()]


def fmt(row):
    return f"{TYPE_NAMES[row[F_TYPE]]:<10} {row[1:]}"


def check_schema(events: list[list[int]]) -> None:
    for i, row in enumerate(events):
        assert len(row) == N_FIELDS, f"row {i} has {len(row)} fields"
        for f in range(N_FIELDS):
            assert 0 <= row[f] < FIELD_VOCAB_SIZES[f], \
                f"row {i} field {f} out of range: {row}"
        t = row[F_TYPE]
        for f in range(1, N_FIELDS):
            if FIELD_LOSS_MASK[t][f] == 0:
                # bar is contextual for measure-block rows even where unmasked;
                # every other masked-out field must be a structural zero.
                if f == F_BAR:
                    continue
                assert row[f] == 0, \
                    f"row {i} ({TYPE_NAMES[t]}) field {f} should be 0: {row}"
    print(f"check_schema: OK ({len(events)} events)")


def test_roundtrip():
    score = build_fixture()
    events1 = encode(score)
    check_schema(events1)

    recon = events_to_score(events1)
    events2 = encode(recon)

    n1 = [r for r in events1 if r[F_TYPE] in (NOTE_T, GRACE_T, REST_T)]
    n2 = [r for r in events2 if r[F_TYPE] in (NOTE_T, GRACE_T, REST_T)]
    if events1 != events2:
        print(f"event mismatch: {len(events1)} vs {len(events2)} rows")
        for i in range(max(len(events1), len(events2))):
            a = events1[i] if i < len(events1) else None
            b = events2[i] if i < len(events2) else None
            flag = "   " if a == b else ">> "
            print(f"{flag}[{i:3d}] {fmt(a) if a else '-':<40} | {fmt(b) if b else '-'}")
        raise AssertionError("encode->decode->encode not identical")
    assert len(n1) == len(n2)
    print(f"test_roundtrip: OK ({len(events1)} events, {len(n1)} note/rest rows)")


def test_transpose_identity():
    score = build_fixture()
    events = encode(score)
    for s in (3, -4, 6):
        back = transpose_events(transpose_events(events, s), -s)
        assert back == events, f"transpose +{s}/-{s} not identity"
    # Circle-of-fifths wrap: D major (+2 sharps) up a semitone is Eb (-3 sharps).
    key_rows = [r for r in transpose_events(events, 1) if r[F_TYPE] == KEY_T]
    assert key_rows[0][F_VALUE] - 1 - 6 == -3, key_rows[0]
    print("test_transpose_identity: OK")


if __name__ == "__main__":
    test_roundtrip()
    test_transpose_identity()
    print("\nALL ROUND-TRIP TESTS PASSED")
