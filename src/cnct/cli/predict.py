"""``cnct-predict`` CLI — run dual-domain cascade inference from a YAML config."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.loader import load_inference_cfg
from ..data.dataset import DualDomainDataset
from ..inference.predictor import Predictor
from ..models import VolumeUNet3D
from ..utils.logging import configure_root_logger

logger = logging.getLogger(__name__)


class _ImageOnlyPredictor(Predictor):
    """Predictor variant that drops Branch A and the DBP, keeping Branch B only.

    Mirrors :class:`cnct.cli.train._ImageOnlyTrainer`: builds a
    :class:`VolumeUNet3D` with ``in_channels=1`` and shims its ``forward`` to
    absorb the sinogram/geometry arguments threaded through the base
    :class:`Predictor` loop.
    """

    def _build_model(self) -> VolumeUNet3D:
        m = self.cfg.model
        net = VolumeUNet3D(
            in_channels=1,
            out_channels=1,
            f_maps=m.volume_f_maps,
            num_groups=m.num_groups,
            use_checkpoint=False,
            use_residual=m.use_fdk,
        )
        base_forward = net.forward

        def _forward(fdk, sinogram=None, geo=None, angles=None):  # noqa: ARG001
            return base_forward(fdk)

        net.forward = _forward  # type: ignore[assignment]
        return net


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the predict CLI."""
    parser = argparse.ArgumentParser(
        prog="cnct-predict",
        description="Run trained dual-domain cascade inference.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an inference YAML config (see configs/inference.yaml).",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--image_only",
        action="store_true",
        help=(
            "Run the image-only baseline: FDK → VolumeUNet3D (Branch B) "
            "→ reconstruction. Skips Branch A and the differentiable "
            "backprojection. Must match the model used at training time."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the ``cnct-predict`` console script.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = _build_parser().parse_args(argv)
    configure_root_logger(log_file=args.log_file, level=args.log_level)

    cfg = load_inference_cfg(args.config)
    logger.info("Loaded inference config: %s", args.config)
    logger.info("Checkpoint: %s  | output_dir: %s", cfg.checkpoint, cfg.output_dir)

    dataset = DualDomainDataset(cfg.data, split=cfg.split, geometry=cfg.geometry)
    if len(dataset) == 0:
        logger.error("No cases discovered for split %r.", cfg.split)
        return 2

    predictor_cls = _ImageOnlyPredictor if args.image_only else Predictor
    if args.image_only:
        logger.info(
            "Image-only baseline: FDK → VolumeUNet3D (use_fdk=%s, no DBP)",
            cfg.model.use_fdk,
        )
    predictor = predictor_cls(cfg, dataset)
    try:
        predictor.run()
    except Exception:  # noqa: BLE001 — top-level CLI boundary
        logger.exception("cnct-predict failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
