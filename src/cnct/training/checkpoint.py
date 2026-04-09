"""Checkpoint I/O for the dual-domain cascade Trainer.

A checkpoint is a plain ``dict`` saved with :func:`torch.save`. Every
dynamic piece of the training loop is included so a resumed run is
bit-indistinguishable from an uninterrupted one:

    * ``epoch`` — last completed epoch.
    * ``model_state_dict`` — network weights.
    * ``optimizer_state_dict`` — AdamW moment estimates + step counters.
    * ``scheduler_state_dict`` — ``ReduceLROnPlateau`` counters.
    * ``scaler_state_dict`` — AMP ``GradScaler`` state (loss scale +
      growth tracker). Omitting this on resume causes the scaler to
      restart with a default scale, which produces NaNs on the first
      restored step.
    * ``best_psnr`` — best validation PSNR seen so far.
    * ``metrics`` — most recent train/val loss+PSNR for human inspection.

Two files per training run live under :attr:`TrainingCfg.checkpoint_dir`:

    * ``last_checkpoint.pytorch`` — overwritten at the end of every epoch.
    * ``best_checkpoint.pytorch`` — overwritten only when val PSNR improves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..utils.io import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointPaths",
    "TrainState",
    "save_checkpoint",
    "load_checkpoint",
]

LAST_NAME = "last_checkpoint.pytorch"
BEST_NAME = "best_checkpoint.pytorch"


@dataclass(frozen=True)
class CheckpointPaths:
    """Resolved filesystem paths for a training run's checkpoints.

    Attributes:
        directory: Root checkpoint directory.
        last: Path to ``last_checkpoint.pytorch``.
        best: Path to ``best_checkpoint.pytorch``.
    """

    directory: Path
    last: Path
    best: Path

    @classmethod
    def from_dir(cls, directory: Path | str) -> "CheckpointPaths":
        """Create a :class:`CheckpointPaths` for a given root directory.

        Args:
            directory: Root checkpoint directory; it is created if absent.

        Returns:
            Resolved checkpoint path bundle.
        """
        root = ensure_dir(directory)
        return cls(directory=root, last=root / LAST_NAME, best=root / BEST_NAME)


@dataclass
class TrainState:
    """Mutable training counters tracked across epochs.

    Attributes:
        start_epoch: Next epoch index to run (1-indexed).
        best_psnr: Best validation PSNR (dB) seen so far.
    """

    start_epoch: int = 1
    best_psnr: float = float("-inf")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    best_psnr: float,
    metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Serialise a complete training snapshot to ``path``.

    Args:
        path: Destination file path. Parent directory must exist.
        epoch: Epoch number that has just completed.
        model: Network whose ``state_dict`` will be stored.
        optimizer: Optimiser whose ``state_dict`` will be stored.
        scheduler: LR scheduler whose ``state_dict`` will be stored.
        scaler: AMP gradient scaler whose ``state_dict`` will be stored.
        best_psnr: Best validation PSNR (dB) observed so far.
        metrics: Latest epoch metrics (e.g. train_loss, val_psnr).
        extra: Optional free-form dict merged into the saved record.
    """
    record: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_psnr": best_psnr,
        "metrics": dict(metrics),
    }
    if extra:
        record.update(extra)

    torch.save(record, path)
    logger.info("Checkpoint saved → %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    map_location: torch.device | str = "cpu",
) -> TrainState:
    """Restore training state from a checkpoint file.

    The model, optimiser, scheduler, and scaler are all updated in place.
    The returned :class:`TrainState` gives the trainer the epoch index to
    resume from and the best PSNR seen so far.

    Args:
        path: Checkpoint file to load.
        model: Target model (must have the same architecture as the one
            that was saved).
        optimizer: Target optimiser.
        scheduler: Target LR scheduler.
        scaler: Target AMP gradient scaler.
        map_location: ``torch.load`` ``map_location`` argument.

    Returns:
        :class:`TrainState` with ``start_epoch = saved_epoch + 1`` and
        ``best_psnr`` populated from the checkpoint.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    state = TrainState(
        start_epoch=int(ckpt["epoch"]) + 1,
        best_psnr=float(ckpt.get("best_psnr", float("-inf"))),
    )
    logger.info(
        "Checkpoint loaded ← %s (resume at epoch %d, best PSNR=%.2f dB)",
        path,
        state.start_epoch,
        state.best_psnr,
    )
    return state
