"""Round-trip test for chord-symbol (lead-sheet harmony) CHORD events.

Builds a small lead sheet (melody + chord symbols, including a slash chord),
encodes it to compound events, decodes it back, and checks that:
  * every chord symbol survives with the correct figure (root/kind/bass), and
  * melody notes are NOT corrupted by the chord symbols (the ChordSymbol-as-Chord
    latent bug).

Also checks derive_bar_chords analysis on a chordless score.

Run: python roundtrip_test_chordsym.py
"""

import music21 as m21

from ml.events import F_TYPE, CHORD_T, NOTE_T
from ml.tokenizer import (
    collect_score, emit_measure_events, flatten_measure_events,
    score_to_header_events, events_to_score, derive_bar_chords,
    sos_event, eos_event,
)

FIGURES = [
    [("Cmaj7", 0.0), ("Dm7", 2.0)],
    [("G7", 0.0)],
    [("C/G", 0.0), ("Am", 2.0)],   # C/G is a slash chord (bass != root)
]


def build_lead_sheet() -> m21.stream.Score:
    s = m21.stream.Score()
    p = m21.stream.Part()
    p.insert(0, m21.instrument.Piano())
    for mi, chords in enumerate(FIGURES):
        m = m21.stream.Measure(number=mi + 1)
        if mi == 0:
            m.insert(0, m21.meter.TimeSignature("4/4"))
        for fig, off in chords:
            m.insert(off, m21.harmony.ChordSymbol(fig))
        m.insert(0.0, m21.note.Note("C4", quarterLength=2.0))
        m.insert(2.0, m21.note.Note("E4", quarterLength=2.0))
        p.append(m)
    s.append(p)
    return s


def encode(score):
    header = score_to_header_events(score)
    collected = collect_score(score)
    flat = flatten_measure_events(emit_measure_events(collected))
    return [sos_event()] + header + flat + [eos_event()]


def test_chordsym_roundtrip():
    score = build_lead_sheet()
    expected = [fig for measure in FIGURES for fig, _ in measure]
    expected_notes = 2 * len(FIGURES)

    seq = encode(score)
    n_chord_rows = sum(1 for r in seq if r[F_TYPE] == CHORD_T)
    assert n_chord_rows == len(expected), f"{n_chord_rows} CHORD rows != {len(expected)}"

    recon = events_to_score(seq)
    decoded = list(recon.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    decoded_figs = [c.figure for c in decoded]
    note_count = len(list(recon.recurse().notes)) - len(decoded)

    print(f"expected chord symbols : {expected}")
    print(f"decoded  chord symbols : {decoded_figs}")
    print(f"melody notes: expected {expected_notes}, got {note_count}")

    assert decoded_figs == expected, f"chord symbols not preserved: {decoded_figs} != {expected}"
    assert note_count == expected_notes, f"melody corrupted: {note_count} != {expected_notes}"
    print("test_chordsym_roundtrip: OK")


def test_derive_bar_chords():
    # A chordless two-part score: C major triad arpeggio + held chord tones.
    score = m21.stream.Score()
    p0 = m21.stream.Part()
    p1 = m21.stream.Part()
    for m_num in (1, 2):
        ma = m21.stream.Measure(number=m_num)
        if m_num == 1:
            ma.insert(0, m21.meter.TimeSignature("4/4"))
        for i, name in enumerate(["C4", "E4", "G4", "C5"]):
            ma.insert(i, m21.note.Note(name, quarterLength=1.0))
        p0.append(ma)
        mb = m21.stream.Measure(number=m_num)
        if m_num == 1:
            mb.insert(0, m21.meter.TimeSignature("4/4"))
        mb.insert(0.0, m21.note.Note("C3", quarterLength=4.0))
        p1.append(mb)
    score.append(p0)
    score.append(p1)

    collected = derive_bar_chords(score, collect_score(score), mode="auto")
    events_by_measure = collected[0]
    chord_rows = {m: [r for r in rows if r[F_TYPE] == CHORD_T]
                  for m, rows in events_by_measure.items()}
    assert all(len(v) >= 1 for v in chord_rows.values()), f"missing bar chords: {chord_rows}"
    # Root should be C (pitch class 0 -> field value 1) in both bars.
    for m, rows in chord_rows.items():
        assert rows[0][3] == 1, f"bar {m} root {rows[0][3] - 1} != C"
    print("test_derive_bar_chords: OK")


if __name__ == "__main__":
    test_chordsym_roundtrip()
    test_derive_bar_chords()
    print("\nALL CHORD-SYMBOL TESTS PASSED")
