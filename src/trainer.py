"""Trains MusicTransformer on tokenized OpenScore data."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import score_to_tokens, text_to_tokens, BOS, PAD
from src.model import MusicTransformer, WEIGHTS_FILE

CHECKPOINT_DIR = Path("checkpoints")


class MusicDataset(Dataset):
    def __init__(self, sequences: list[list[int]], max_len: int = 512):
        self.seqs    = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx][: self.max_len + 1]
        x   = seq[:-1]
        y   = seq[1:]
        pad = self.max_len - len(x)
        x  += [PAD] * pad
        y  += [PAD] * pad
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def _load_sequences(paths: list[str], log: Callable[[str], None]) -> list[list[int]]:
    import music21 as m21
    sequences = []
    for i, path in enumerate(paths):
        if i % 20 == 0:
            log(f"  Parsing {i}/{len(paths)} scores...")
        try:
            score = m21.converter.parse(path)
            # build text prefix from metadata
            meta  = score.metadata
            title = getattr(meta, "title", "") or ""
            composer = getattr(meta, "composer", "") or ""
            prefix = text_to_tokens((composer + " " + title).strip()[:56])
            music  = score_to_tokens(score)[1:]  # strip leading BOS; prepended below
            sequences.append([BOS] + prefix + music)
        except Exception as e:
            log(f"  Skip {Path(path).name}: {e}")
    return sequences


def train(
    model: MusicTransformer,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float], None],
    stop_flag: list[bool],
):
    log("Parsing scores...")
    sequences = _load_sequences(data_paths, log)
    if not sequences:
        raise RuntimeError("No sequences loaded — run 'Download Dataset' first.")
    log(f"Loaded {len(sequences)} sequences.")

    random.shuffle(sequences)
    dataset    = MusicDataset(sequences, model.cfg["max_seq_len"])
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss(ignore_index=PAD)
    model.to(device).train()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    global_step = 0

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("Training stopped by user.")
            break
        epoch_loss = 0.0
        batches    = 0
        for x, y in loader:
            if stop_flag[0]:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss  += loss.item()
            global_step += 1
            batches     += 1

        avg_loss = epoch_loss / max(batches, 1)
        on_epoch(epoch, avg_loss)
        torch.save(model.state_dict(), WEIGHTS_FILE)
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  (checkpoint saved)")

    log("Training complete.")
