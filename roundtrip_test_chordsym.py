"""Round-trip test for chord-symbol (lead-sheet harmony) tokens.

Builds a small lead sheet (melody + chord symbols, including a slash chord),
tokenizes it, decodes it back, and checks that:
  * every chord symbol survives with the correct figure (root/kind/bass), and
  * melody notes are NOT corrupted by the chord symbols (the ChordSymbol-as-Chord
    latent bug).

Run: python roundtrip_test_chordsym.py
"""

import music21 as m21

import ml.tokenizer as tk

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


def encode(score) -> list[int]:
    header = tk.score_to_header_tokens(score)
    measures = tk.score_to_tokens_by_measure(score)
    seq = [tk.BOS] + header
    for mn in sorted(measures):
        seq += measures[mn]
    seq.append(tk.EOS)
    return seq


def main():
    score = build_lead_sheet()
    expected = [fig for measure in FIGURES for fig, _ in measure]
    expected_notes = 2 * len(FIGURES)

    seq = encode(score)
    assert max(seq) < tk.VOCAB_SIZE, "token id out of vocab range"

    recon = tk.tokens_to_score(seq)
    decoded = list(recon.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    decoded_figs = [c.figure for c in decoded]
    note_count = len(list(recon.recurse().notes)) - len(decoded)

    print(f"expected chord symbols : {expected}")
    print(f"decoded  chord symbols : {decoded_figs}")
    print(f"melody notes: expected {expected_notes}, got {note_count}")

    assert decoded_figs == expected, f"chord symbols not preserved: {decoded_figs} != {expected}"
    assert note_count == expected_notes, f"melody corrupted: {note_count} != {expected_notes}"
    print("\nChord-symbol round-trip OK.")


if __name__ == "__main__":
    main()
