"""``cnct-prepare-data`` CLI — build HDF5 train/val/test split directories.

Thin argparse wrapper around :func:`cnct.data.prepare.build_splits`. The
builder is idempotent and safe to re-run: existing ``.h5`` files are left
untouched so interrupted jobs can resume cleanly.

Example:
    cnct-prepare-data --fdk_dir /path/to/fdk60 \\
                      --gt_dir  /path/to/AbdomenCT-1K-Image \\
                      --out_dir /path/to/h5_3dunet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..data.prepare import PrepareConfig, build_splits
from ..utils.logging import configure_root_logger

logger = logging.getLogger(__name__)

_BASE_DIR = Path("/projects/CTdata")  # root data directory (change for a new server)

_DEFAULTS = PrepareConfig(
    fdk_dir=_BASE_DIR / "fdk60",
    gt_dir=_BASE_DIR / "AbdomenCT-1K-Image",
    out_dir=_BASE_DIR / "h5_3dunet",
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the prepare CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="cnct-prepare-data",
        description=(
            "Build gzip-compressed HDF5 train/val/test splits from FDK "
            "reconstructions and ground-truth NIfTI volumes."
        ),
    )
    parser.add_argument("--fdk_dir", type=Path, default=_DEFAULTS.fdk_dir)
    parser.add_argument("--gt_dir", type=Path, default=_DEFAULTS.gt_dir)
    parser.add_argument("--out_dir", type=Path, default=_DEFAULTS.out_dir)
    parser.add_argument("--mu_water", type=float, default=_DEFAULTS.mu_water)
    parser.add_argument(
        "--patch_min",
        type=int,
        default=_DEFAULTS.patch_min,
        help="Skip cases with any spatial dimension < patch_min",
    )
    parser.add_argument("--train_frac", type=float, default=_DEFAULTS.train_frac)
    parser.add_argument("--val_frac", type=float, default=_DEFAULTS.val_frac)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print split sizes without writing any files",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cnct-prepare-data`` console script.

    Args:
        argv: Optional argument list (used by tests). Defaults to
            ``sys.argv[1:]``.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = _build_parser().parse_args(argv)
    configure_root_logger(level=args.log_level)

    cfg = PrepareConfig(
        fdk_dir=args.fdk_dir,
        gt_dir=args.gt_dir,
        out_dir=args.out_dir,
        mu_water=args.mu_water,
        patch_min=args.patch_min,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    try:
        build_splits(cfg, dry_run=args.dry_run)
    except Exception:  # noqa: BLE001 — top-level CLI boundary
        logger.exception("cnct-prepare-data failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
