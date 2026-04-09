"""``cnct-train`` CLI — run dual-domain cascade training from a YAML config.

The entry point is deliberately thin: every configurable parameter lives
in the YAML file loaded by :func:`cnct.config.loader.load_training_cfg`.
The CLI's only job is to stitch config → datasets → Trainer → ``fit()``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.loader import load_training_cfg
from ..data.dataset import DualDomainDataset
from ..training.trainer import DEFAULT_GRAD_CLIP_NORM, Trainer
from ..utils.logging import configure_root_logger

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the train CLI."""
    parser = argparse.ArgumentParser(
        prog="cnct-train",
        description="Train the dual-domain cascade network.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training YAML config (see configs/training.yaml).",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=DEFAULT_GRAD_CLIP_NORM,
        help=(
            "Maximum L2 norm for gradient clipping. Pass 0 to disable "
            "clipping (not recommended — TIGRE adjoint gradients can spike)."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional path to a file log in addition to stdout.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the ``cnct-train`` console script.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = _build_parser().parse_args(argv)
    configure_root_logger(log_file=args.log_file, level=args.log_level)

    cfg = load_training_cfg(args.config)
    logger.info("Loaded training config: %s", args.config)
    logger.info("Checkpoint dir: %s", cfg.checkpoint_dir)

    train_ds = DualDomainDataset(cfg.data, split="train", geometry=cfg.geometry)
    val_ds = DualDomainDataset(cfg.data, split="val", geometry=cfg.geometry)

    if len(train_ds) == 0:
        logger.error(
            "No training cases found. Run `cnct-prepare-data` and the "
            "cnct_dataprep pipeline first."
        )
        return 2

    clip = args.grad_clip_norm if args.grad_clip_norm > 0 else None
    trainer = Trainer(cfg, train_ds, val_ds, grad_clip_norm=clip)

    try:
        trainer.fit()
    except Exception:  # noqa: BLE001 — top-level CLI boundary
        logger.exception("cnct-train failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
