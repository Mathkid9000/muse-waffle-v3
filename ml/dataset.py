import torch
from helpers import SEQ_LEN, BATCH_SIZE
import numpy as np

class QuintupleDataset(torch.utils.data.Dataset):
    """Concatenate encoded into one (N,5) stream and chunk into
    (x, y) = ((seq_len,5), (seq_len,5)) windows shifted by one event."""

    def __init__(self, encoded, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.encoded = np.concatenate(list(encoded), axis=0)

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def make_dataloader(encoded_events, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True):
    dataset = QuintupleDataset(encoded_events, seq_len=seq_len)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), seq_len // 4))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)