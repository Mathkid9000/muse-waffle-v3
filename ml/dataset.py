import torch
from helpers import SEQ_LEN, BATCH_SIZE
from tokenizer import load_vocab_list, transpose_tok, TRANSPOSE_INTERVALS
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
        # score-major, since evaluate_val only reads the first batches of the val stream and would
        # otherwise see a single key
        self.encoded = np.concatenate([
            np.stack([ tables[i][f][seq[:, f]] for f in range(5) ], axis=1)
            for seq in encoded for i in TRANSPOSE_INTERVALS
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


def make_dataloader(encoded_events, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True):
    dataset = QuintupleDataset(encoded_events, seq_len=seq_len)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), seq_len // 4))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)