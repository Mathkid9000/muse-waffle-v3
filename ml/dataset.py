import math
import torch
from helpers import SEQ_LEN, BATCH_SIZE
from tokenizer import load_vocab_list, transpose_tok, NULL_ROW, TRANSPOSE_INTERVALS
import numpy as np

class QuintupleDataset(torch.utils.data.Dataset):
    """Transpose encoded to every key, concatenate into one (N,5) stream and chunk
    into (x, y) = ((seq_len,5), (seq_len,5)) windows shifted by one event."""

    def __init__(self, encoded, seq_len=SEQ_LEN):
        # scores are stored in their original key only and transposed here instead, so a score and
        # its transpositions can never land on opposite sides of the train/val split. Transposing a
        # row only renames its pitch tokens, so each interval is one id -> id table per field.
        self.seq_len = seq_len
        vocab = load_vocab_list()
        ids = [ { tok: idx for idx, tok in enumerate(head) } for head in vocab ]
        # the vocab's outermost pitches exist only as transposition targets, so they have nowhere
        # further to go; no stored score uses them, so mapping them to themselves is never read
        # int16 rather than the int64 the tables would default to: the widest head
        # is a few hundred entries, and the concatenated stream is every score in
        # all 12 keys, so the dtype is the difference between ~2GB and ~9GB resident
        # (double that at the concatenate, which holds source and destination at once)
        tables = { i: [ np.array([ ids[f].get(transpose_tok(tok, i), idx) for idx, tok in enumerate(vocab[f]) ], dtype=np.int16) for f in range(5) ] for i in TRANSPOSE_INTERVALS }
        # the (score, interval) segments are shuffled (seeded, so every run of the same data sees
        # the same stream) before concatenating. Laid out score-major instead, a 2048-row window
        # regularly held a score followed by ITS OWN transposition a semitone up (the median score
        # is ~1000 rows), and "copy the previous section shifted" is a shortcut that drives train
        # loss down without teaching composition. Shuffling makes adjacent neighbors different
        # scores, and gives evaluate_val's first batches key and score diversity for free.
        segments = [ (s, i) for s in range(len(encoded)) for i in TRANSPOSE_INTERVALS ]
        np.random.default_rng(0).shuffle(segments)
        self.encoded = np.concatenate([
            np.stack([ tables[i][f][encoded[s][:, f]] for f in range(5) ], axis=1)
            for s, i in segments
        ], axis=0)

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]
        # widen at the window boundary, since nn.Embedding and cross_entropy both
        # want int64; one window is 5 * seq_len ids, so the cast costs nothing
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


class InfillDataset(QuintupleDataset):
    """The same windowed stream, read as a canvas instead of a next-row target: a window is
    corrupted in place and only the hidden fields carry loss. Windows are still allowed to
    straddle a score boundary — no row references a position, absolute or otherwise, so the
    EOS/SOG pair ends the context here exactly as it does for the causal model.

    ``seed`` pins each window's corruption to its index, so a val split scores the same task
    every epoch rather than a freshly drawn one; leave it None to train."""

    def __init__(self, encoded, seq_len=SEQ_LEN, seed=None):
        super().__init__(encoded, seq_len)
        self.seed = seed
        vocab = load_vocab_list()
        ids = [ { tok: idx for idx, tok in enumerate(head) } for head in vocab ]
        self.mask_ids = np.array([ ids[f]['MASK'] for f in range(5) ], dtype=np.int16)
        self.null_row = np.array([ ids[f][tok] for f, tok in enumerate(NULL_ROW) ], dtype=np.int16)
        self.deg_ids = np.array([ idx for tok, idx in ids[2].items() if tok.startswith('DEG') ], dtype=np.int16)
        self.voice_ids = np.array([ idx for tok, idx in ids[4].items() if tok.startswith('VOICE') ], dtype=np.int16)

    def __len__(self):
        # no shifted target to leave room for, so a window may start on the last usable row
        return max(0, len(self.encoded) - self.seq_len + 1)

    def canvas(self, idx, rng):
        # splice NULL rows into the window: the model only learns to leave a slot empty if it
        # has seen empty slots, and at generation the caller sizes each bar without knowing
        # how many events it wants, so it can only over-allocate
        rows = self.encoded[idx:idx + self.seq_len - rng.integers(0, self.seq_len // 4)]
        canvas = np.repeat(self.null_row[None], self.seq_len, axis=0)
        canvas[np.sort(rng.choice(self.seq_len, len(rows), replace=False))] = rows
        return canvas

    def corrupt(self, canvas, rng):
        """(T,5) bool of the fields to hide. Rows and fields are drawn independently: the row
        axis is a span, a scatter, or everything outside the rows worth keeping, and the field
        axis is the shape of one injection."""
        # the rate is cosine-distributed rather than uniform, so the mixture spends real time
        # at high rates; a masked model only takes gradient where it is hidden, and a uniform
        # draw spends half its batches supervising a quarter of the canvas
        rate = math.cos(rng.random() * math.pi / 2)
        row = rng.random()
        if row < 0.35:                                       # one contiguous hole
            rows = np.zeros(len(canvas), dtype=bool)
            width = max(1, int(len(canvas) * rate))
            start = rng.integers(0, len(canvas) - width + 1)
            rows[start:start + width] = True
        elif row < 0.70:                                     # injections scattered through a region
            rows = rng.random(len(canvas)) < rate
        elif row < 0.85:                                     # keep the harmonic skeleton
            rows = ~np.isin(canvas[:, 2], self.deg_ids)
        else:                                                # keep one voice, write the others
            present = np.intersect1d(canvas[:, 4], self.voice_ids)
            rows = canvas[:, 4] != rng.choice(present) if len(present) else np.ones(len(canvas), dtype=bool)

        fields = np.ones(5, dtype=bool)
        field = rng.random()
        if field < 0.6:
            fields[0] = False                                # what the caller actually injects
        elif field >= 0.8:
            fields[0] = False                                # one field survives beside the bar:
            fields[rng.integers(1, 5)] = False               # a rhythmic grid, a pitch line, a voice

        hidden = rows[:, None] & fields
        # a spliced NULL slot is always the model's to fill, whichever rows the pattern chose
        hidden[np.all(canvas == self.null_row, axis=1)] = fields
        return hidden

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.seed is None else self.seed + idx)
        y = self.canvas(idx, rng)
        hidden = self.corrupt(y, rng)
        x = np.where(hidden, self.mask_ids, y)
        # widen at the window boundary, since nn.Embedding and cross_entropy both
        # want int64; one window is 5 * seq_len ids, so the cast costs nothing
        return torch.from_numpy(x).long(), torch.from_numpy(y).long(), torch.from_numpy(hidden).float()


def make_dataloader(encoded_events, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True):
    dataset = QuintupleDataset(encoded_events, seq_len=seq_len)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), seq_len // 4))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def make_infill_dataloader(encoded_events, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                           shuffle=True, stride=None, seed=None):
    dataset = InfillDataset(encoded_events, seq_len=seq_len, seed=seed)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), stride or seq_len // 4))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)