from ml.trainer import MusicDataset
from ml.trainer import _load_measure_sequences
from ml.dataset import load_cached_paths
from ml.tokenizer import tokens_to_score
import numpy as np
from random import randint

idx=randint(0,1800)
paths = load_cached_paths()[idx:idx+1]
sequences = _load_measure_sequences(paths, print)

max_len = 2048
IDX = 1
train_set = MusicDataset(sequences, max_len, 0, transpose_range=0, fim_example_fn=None)

print('Loaded dataset')

x, y, mask = train_set.__getitem__(0)
score_in = tokens_to_score(x.flatten().tolist())
# score_out = tokens_to_score(y.flatten().tolist())

score_in.write("musicxml", fp="./roundtrip/roundtrip_in.mxl")
# score_out.write("musicxml", fp="./roundtrip/roundtrip_out.mxl")

diffs = [ i for i, x in enumerate(np.absolute(x - y).flatten().tolist()) if x > 0 ]

print(f"Analyzed score ({idx}): {paths[0]}")
print(f"SCORE IN ({len(x):04d} tokens) / SCORE OUT ({len(y):04d} tokens)")
print(f"Diffs found: {len(diffs)}")
print(x.flatten().tolist()[:1000])