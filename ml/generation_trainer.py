"""Trains MusicTransformer for pure autoregressive music generation.

All training examples use the standard next-token prediction format:
    [BOS] + header + [BAR] + measures... + [EOS]

No fill-in-the-middle restructuring is applied.
"""

from __future__ import annotations
from typing import Callable

from ml.model import MusicTransformer
from ml.trainer import _train_loop


def train_generation(
    name: str,
    model: MusicTransformer,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float, float], None],
    stop_flag: list[bool],
    resume_optimizer: bool = False,
):
    """Train model on pure standard autoregressive generation (no FIM).

    Args:
        model: MusicTransformer model
        data_paths: Paths to MusicXML files
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        log: Logging callback
        on_epoch: Epoch callback (epoch, train_loss, val_loss)
        stop_flag: Stop flag for interruption
        resume_optimizer: Resume optimizer state and val split from last run
    """
    _train_loop(
        name, model, data_paths, epochs, batch_size, lr, device,
        log, on_epoch, stop_flag,
        fim_ratio=0.0,
        fim_example_fn=None,
        resume_optimizer=resume_optimizer,
    )
