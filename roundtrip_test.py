"""
Round-trip tests for the tokenizer.

1. Basic round-trip: fixed score -> tokenize -> detokenize -> compare note counts.
2. FIM round-trip: random InfillDataset item -> decode the exact FIM sequence the
   transformer sees -> reconstruct score in natural order -> color gap notes red.
"""

import sys, os, random
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(__file__))

import music21 as m21
from ml.tokenizer import (
    score_to_tokens, tokens_to_score,
    PAD, BOS, EOS, BAR, TIE, GRACE, NOTE_OFF, DUR_OFF, REST_OFF,
    KEY_OFF, TIME_OFF, TEXT_OFF, PART_OFF, STEP_OFF, MAX_STEP, MAX_PARTS,
    FILL_START, FILL_END,
    VOCAB_SIZE, DURATIONS, STEP_SIZE,
)
from pathlib import Path
from ml.dataset import load_cached_paths

load_dotenv('env.env')
STORAGE_DIR = Path(os.environ.get('STORAGE_DIR'))
paths = load_cached_paths()
SCORE_PATH = paths[1]

# -- Load ------------------------------------------------------------------
print("Loading score ...")
original = m21.converter.parse(SCORE_PATH)
parts = list(original.parts)
print(f"  {len(parts)} part(s), "
      f"{len(list(original.recurse().getElementsByClass(m21.stream.Measure)))} measures total")

# -- Tokenize --------------------------------------------------------------
tokens = score_to_tokens(original)
print(f"\nToken sequence length : {len(tokens)}")
print(f"Vocabulary size       : {VOCAB_SIZE}")

def tok_name(t):
    if t == PAD:        return "PAD"
    if t == BOS:        return "BOS"
    if t == EOS:        return "EOS"
    if t == BAR:        return "BAR"
    if t == TIE:        return "TIE"
    if t == GRACE:      return "GRACE"
    if t == FILL_START: return "FILL_START"
    if t == FILL_END:   return "FILL_END"
    if NOTE_OFF <= t < NOTE_OFF + 128:
        return f"NOTE({m21.pitch.Pitch(midi=t-NOTE_OFF).nameWithOctave})"
    if DUR_OFF <= t < DUR_OFF + len(DURATIONS):
        return f"DUR({DURATIONS[t-DUR_OFF]:.4g})"
    if REST_OFF <= t < REST_OFF + len(DURATIONS):
        return f"REST({DURATIONS[t-REST_OFF]:.4g})"
    if KEY_OFF <= t < KEY_OFF + 13:
        return f"KEY({t-KEY_OFF-6:+d})"
    if TIME_OFF <= t < TIME_OFF + 6:
        return f"TIME({t-TIME_OFF})"
    if TEXT_OFF <= t < TEXT_OFF + 57:
        return f"TEXT({chr(t-TEXT_OFF+32)!r})"
    if PART_OFF <= t < PART_OFF + MAX_PARTS:
        return f"PART_{t-PART_OFF}"
    if STEP_OFF <= t < STEP_OFF + MAX_STEP:
        return f"STEP_{t-STEP_OFF}(+{(t-STEP_OFF)*STEP_SIZE:.4g}ql)"
    return f"UNK({t})"

# Print first 240 tokens
print("\n-- First 240 tokens --")
for i, t in enumerate(tokens[:240]):
    print(f"  [{i:4d}] {t:4d}  {tok_name(t)}")
if len(tokens) > 240:
    print(f"  ... ({len(tokens)-240} more tokens) ...")
print(f"  [{len(tokens)-1:4d}] {tokens[-1]:4d}  {tok_name(tokens[-1])}")

# Token-type histogram
from collections import Counter
hist = Counter()
for t in tokens:
    if t in (BOS, EOS, BAR, PAD):
        hist[tok_name(t)] += 1
    elif t == TIE:                                    hist["TIE"]   += 1
    elif t == GRACE:                                  hist["GRACE"] += 1
    elif NOTE_OFF <= t < NOTE_OFF + 128:              hist["NOTE"]  += 1
    elif DUR_OFF  <= t < DUR_OFF  + len(DURATIONS):  hist["DUR"]   += 1
    elif REST_OFF <= t < REST_OFF + len(DURATIONS):  hist["REST"]  += 1
    elif PART_OFF <= t < PART_OFF + MAX_PARTS:        hist["PART"]  += 1
    elif STEP_OFF <= t < STEP_OFF + MAX_STEP:         hist["STEP"]  += 1
    else: hist["OTHER"] += 1

print("\n-- Token-type histogram --")
for k, v in sorted(hist.items(), key=lambda x: -x[1]):
    print(f"  {k:<8} {v:5d}")

# -- Detokenize ------------------------------------------------------------
print("\nDetokenizing ...")
reconstructed = tokens_to_score(tokens)

out_xml = "roundtrip_output.xml"
reconstructed.write("musicxml", fp=out_xml)
print(f"Reconstructed score written to: {out_xml}")

orig_notes  = len(list(original.flatten().notes))
recon_notes = len(list(reconstructed.flatten().notes))
print(f"\nOriginal note count     : {orig_notes}")
print(f"Reconstructed note count: {recon_notes}")
if orig_notes == recon_notes:
    print("  PASS: Note counts match -- round-trip looks correct.")
else:
    print("  FAIL: Note counts differ -- check tokenizer for dropped/merged events.")


# -- FIM round-trip via InfillDataset --------------------------------------
print("\n" + "="*70)
print("FIM ROUND-TRIP TEST  (using InfillDataset -- exactly what the model sees)")
print("="*70)

from ml.infill_trainer import InfillDataset, _load_measure_sequences, create_fim_example

def fim_roundtrip_test(seed: int | None = None):
    """Use InfillDataset to produce the exact FIM token sequence the transformer
    trains on, then decode it back to a score with the gap measures colored red."""
    if seed is not None:
        random.seed(seed)

    if not paths:
        print("  SKIP: No dataset files found. Run 'Download Dataset' first.")
        return

    # Build the dataset the same way training does
    sequences = _load_measure_sequences(paths, print, max_sequences=100)
    if not sequences:
        print("  SKIP: No sequences parsed.")
        return
    dataset = InfillDataset(sequences)

    # Pick a random item; retry until create_fim_example succeeds
    for _ in range(20):
        idx = random.randrange(len(dataset))
        measure_tokens = dataset.sequences[idx]
        result = create_fim_example(measure_tokens)
        if result is not None:
            break
    else:
        print("  SKIP: Could not create a FIM example after 20 tries.")
        return

    fim_seq, loss_start_idx = result

    # Parse the FIM sequence
    # Layout: [BOS] prefix [FILL_START] suffix [FILL_END] gap [EOS]
    fill_start_pos = fim_seq.index(FILL_START)
    fill_end_pos   = fim_seq.index(FILL_END)

    prefix_tokens = fim_seq[1                 : fill_start_pos]
    suffix_tokens = fim_seq[fill_start_pos + 1: fill_end_pos]
    gap_tokens    = fim_seq[fill_end_pos + 1  : -1]           # strip EOS

    n_prefix_bars        = prefix_tokens.count(BAR)
    n_gap_bars           = gap_tokens.count(BAR)
    decoded_gap_measures = set(range(n_prefix_bars + 1, n_prefix_bars + n_gap_bars + 1))

    print(f"  Dataset index   : {idx}  ({len(measure_tokens)} measures)")
    print(f"  FIM seq length  : {len(fim_seq)} tokens  "
          f"(prefix {len(prefix_tokens)}  |  suffix {len(suffix_tokens)}  |  gap {len(gap_tokens)})")
    print(f"  Loss starts at  : position {loss_start_idx}  (gap begins here)")
    print(f"  Gap bars        : {n_gap_bars}  -> decoded measure nos. {sorted(decoded_gap_measures)}")

    # Print what the transformer sees (first 80 tokens)
    print("\n  -- FIM token sequence (first 80 tokens, what the model sees) --")
    region = ["BOS"]
    for t in prefix_tokens: region.append(f"PFX[{tok_name(t)}]")
    region.append("FILL_START")
    for t in suffix_tokens: region.append(f"SFX[{tok_name(t)}]")
    region.append("FILL_END")
    for t in gap_tokens:    region.append(f"GAP[{tok_name(t)}]")
    region.append("EOS")

    for i, (raw, label) in enumerate(zip(fim_seq[:80], region[:80])):
        marker = "  <- loss starts" if i == loss_start_idx else ""
        print(f"  [{i:4d}] {raw:4d}  {label}{marker}")
    if len(fim_seq) > 80:
        print(f"  ... ({len(fim_seq) - 80} more tokens) ...")

    # Reassemble in natural order and decode to score
    full_tokens = [BOS] + prefix_tokens + gap_tokens + suffix_tokens + [EOS]
    score = tokens_to_score(full_tokens)

    # Color gap measures red
    colored = 0
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            if measure.number in decoded_gap_measures:
                for el in measure.recurse():
                    if isinstance(el, (m21.note.Note, m21.chord.Chord)):
                        el.style.color = 'red'
                        colored += 1

    print(f"\n  Colored {colored} note(s) red in gap measures.")

    out_xml = "fim_roundtrip_output.xml"
    score.write("musicxml", fp=out_xml)
    print(f"  FIM round-trip score written to: {out_xml}")

fim_roundtrip_test()
