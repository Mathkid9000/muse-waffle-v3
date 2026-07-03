"""Round-trip a random cached dataset score through the compound-event codec.

Requires downloaded dataset files (run the dataset cells in
notebooks/training.ipynb first).  Picks a random score, encodes it via the
same path the trainer uses, decodes it back to MusicXML, and reports note
counts plus a MusicDataset tensor sanity check.
"""

from random import randint
from pathlib import Path

import music21 as m21

from ml.dataset import load_cached_paths
from ml.events import F_TYPE, NOTE_T, GRACE_T, PAD_T
from ml.tokenizer import (
    collect_score, emit_measure_events, score_to_header_events, events_to_score,
    MAX_REAL_PARTS,
)
from ml.trainer import MusicDataset

paths = load_cached_paths()
if not paths:
    raise SystemExit("No dataset cached — run 'Download Dataset' first.")

idx = randint(0, len(paths) - 1)
path = paths[idx]
print(f"Analyzing score ({idx}): {path}")

score = m21.converter.parse(path)
header = score_to_header_events(score)
collected = collect_score(score)
measure_events = emit_measure_events(collected)
n_parts = len(list(score.parts)[:MAX_REAL_PARTS])

train_set = MusicDataset([(header, measure_events, collected, n_parts)],
                         max_len=1024, fim_ratio=0, transpose=False)
x, y, mask = train_set[0]
print(f"x {tuple(x.shape)}  y {tuple(y.shape)}  mask {tuple(mask.shape)}")
assert x[1:].equal(y[:-1]), "y is not x shifted by one event"

rows = [r for r in x.tolist() if r[F_TYPE] != PAD_T]
recon = events_to_score(rows)

Path("roundtrip").mkdir(exist_ok=True)
recon.write("musicxml", fp="./roundtrip/roundtrip_in.mxl")

orig_notes  = sum(1 for r in rows if r[F_TYPE] in (NOTE_T, GRACE_T))
recon_notes = len(list(recon.flatten().notes))
print(f"encoded NOTE/GRACE rows: {orig_notes}, reconstructed notes: {recon_notes}")
print("wrote ./roundtrip/roundtrip_in.mxl")
