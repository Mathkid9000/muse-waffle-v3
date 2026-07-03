"""MLE-pretrains the MMT generator for pure autoregressive generation.

All training examples use the standard next-event prediction format:
    [SOS] + header + measures (bars 0..N-1) + [EOS]

No fill-in-the-middle restructuring is applied — use ml.infill_trainer for
the combined standard+FIM regime (recommended for the product's infill
feature), then ml.gan_trainer for MMT-BERT's adversarial phase.
"""

from __future__ import annotations
from typing import Callable

from ml.trainer import _train_loop


def train_generation(
    name: str,
    model,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float, float], None],
    stop_flag: list[bool],
    resume_optimizer: bool = False,
    cache_workers: int = 1,
):
    """Train model on pure standard autoregressive generation (no FIM).

    Args:
        model: MMTGenerator model
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
        cache_workers=cache_workers,
    )
