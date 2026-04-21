"""
Round-trip test: load the training score → tokenize → detokenize → show token stats
and save both a MusicXML of the reconstructed score and a plain-text token dump.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import music21 as m21
from src.tokenizer import (
    score_to_tokens, tokens_to_score,
    PAD, BOS, EOS, BAR, NOTE_OFF, DUR_OFF, REST_OFF,
    KEY_OFF, TIME_OFF, TEXT_OFF, PART_OFF, STEP_OFF, MAX_STEP, MAX_PARTS,
    VOCAB_SIZE, DURATIONS,
)

SCORE_PATH = r"data\openscore\scores_Abbott,_Jane_Bingham___Just_for_Today_lc6583477.mxl"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading score …")
original = m21.converter.parse(SCORE_PATH)
parts = list(original.parts)
print(f"  {len(parts)} part(s), "
      f"{len(list(original.recurse().getElementsByClass(m21.stream.Measure)))} measures total")

# ── Tokenize ─────────────────────────────────────────────────────────────────
tokens = score_to_tokens(original)
print(f"\nToken sequence length : {len(tokens)}")
print(f"Vocabulary size       : {VOCAB_SIZE}")

def tok_name(t):
    if t == PAD:   return "PAD"
    if t == BOS:   return "BOS"
    if t == EOS:   return "EOS"
    if t == BAR:   return "BAR"
    if NOTE_OFF <= t < NOTE_OFF + 128:
        return f"NOTE({m21.pitch.Pitch(midi=t-NOTE_OFF).nameWithOctave})"
    if DUR_OFF <= t < DUR_OFF + 16:
        return f"DUR({DURATIONS[t-DUR_OFF]})"
    if REST_OFF <= t < REST_OFF + 16:
        return f"REST({DURATIONS[t-REST_OFF]})"
    if KEY_OFF <= t < KEY_OFF + 13:
        return f"KEY({t-KEY_OFF-6:+d})"
    if TIME_OFF <= t < TIME_OFF + 6:
        return f"TIME({t-TIME_OFF})"
    if TEXT_OFF <= t < TEXT_OFF + 56:
        return f"TEXT({chr(t-TEXT_OFF+32)!r})"
    if PART_OFF <= t < PART_OFF + MAX_PARTS:
        return f"PART_{t-PART_OFF}"
    if STEP_OFF <= t < STEP_OFF + MAX_STEP:
        return f"STEP_{t-STEP_OFF}(+{(t-STEP_OFF)*0.125:.3f}ql)"
    return f"UNK({t})"

# Print first 120 tokens to give a clear picture
print("\n-- First 120 tokens --")
for i, t in enumerate(tokens[:240]):
    print(f"  [{i:4d}] {t:4d}  {tok_name(t)}")
if len(tokens) > 240:
    print(f"  … ({len(tokens)-240} more tokens) …")
print(f"  [{len(tokens)-1:4d}] {tokens[-1]:4d}  {tok_name(tokens[-1])}")

# Token-type histogram
from collections import Counter
hist = Counter()
for t in tokens:
    if t in (BOS, EOS, BAR, PAD):
        hist[tok_name(t)] += 1
    elif NOTE_OFF <= t < NOTE_OFF + 128: hist["NOTE"] += 1
    elif DUR_OFF  <= t < DUR_OFF + 16:   hist["DUR"]  += 1
    elif REST_OFF <= t < REST_OFF + 16:  hist["REST"] += 1
    elif PART_OFF <= t < PART_OFF + MAX_PARTS: hist["PART"] += 1
    elif STEP_OFF <= t < STEP_OFF + MAX_STEP: hist["STEP"] += 1
    else: hist["OTHER"] += 1

print("\n-- Token-type histogram --")
for k, v in sorted(hist.items(), key=lambda x: -x[1]):
    print(f"  {k:<8} {v:5d}")

# ── Detokenize ────────────────────────────────────────────────────────────────
print("\nDetokenizing …")
reconstructed = tokens_to_score(tokens)

out_xml = "roundtrip_output.xml"
reconstructed.write("musicxml", fp=out_xml)
print(f"Reconstructed score written to: {out_xml}")

# Quick note-count comparison
orig_notes  = len(list(original.flatten().notes))
recon_notes = len(list(reconstructed.flatten().notes))
print(f"\nOriginal note count     : {orig_notes}")
print(f"Reconstructed note count: {recon_notes}")
if orig_notes == recon_notes:
    print("  PASS: Note counts match -- round-trip looks correct.")
else:
    print("  FAIL: Note counts differ -- check tokenizer for dropped/merged events.")
