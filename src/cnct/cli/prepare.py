"""``cnct-prepare-data`` CLI — build HDF5 train/val/test split directories.

Thin argparse wrapper around :func:`cnct.data.prepare.build_splits`. The
builder is idempotent and safe to re-run: existing ``.h5`` files are left
untouched so interrupted jobs can resume cleanly.

When ``--geometry_config`` is supplied the ``n_angles`` value is read from
the geometry YAML and used to resolve ``{n_angles}`` placeholders in the
default ``--fdk_dir`` path, exactly like the data_prep and training configs.

Example:
    cnct-prepare-data --geometry_config configs/geometry.yaml
    cnct-prepare-data --fdk_dir /path/to/fdk30 --gt_dir /path/to/AbdomenCT-1K-Image
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config.loader import load_geometry_cfg
from ..data.prepare import PrepareConfig, build_splits
from ..utils.logging import configure_root_logger

logger = logging.getLogger(__name__)

_BASE_DIR = Path("/projects/CTdata")  # root data directory (change for a new server)


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
    parser.add_argument(
        "--geometry_config",
        type=Path,
        default=None,
        help=(
            "Path to geometry YAML (e.g. configs/geometry.yaml). "
            "n_angles is read from it to resolve {n_angles} in --fdk_dir."
        ),
    )
    parser.add_argument("--fdk_dir", type=Path, default=None)
    parser.add_argument("--gt_dir", type=Path, default=_BASE_DIR / "AbdomenCT-1K-Image")
    parser.add_argument("--out_dir", type=Path, default=_BASE_DIR / "h5_3dunet")
    defaults = PrepareConfig(fdk_dir=Path("."), gt_dir=Path("."), out_dir=Path("."))
    parser.add_argument("--mu_water", type=float, default=defaults.mu_water)
    parser.add_argument(
        "--patch_min",
        type=int,
        default=defaults.patch_min,
        help="Skip cases with any spatial dimension < patch_min",
    )
    parser.add_argument("--train_frac", type=float, default=defaults.train_frac)
    parser.add_argument("--val_frac", type=float, default=defaults.val_frac)
    parser.add_argument("--seed", type=int, default=defaults.seed)
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

    # Resolve fdk_dir: explicit flag > geometry-derived > error.
    fdk_dir = args.fdk_dir
    if fdk_dir is None:
        if args.geometry_config is not None:
            geo = load_geometry_cfg(args.geometry_config)
            fdk_dir = _BASE_DIR / f"fdk{geo.n_angles}"
            logger.info(
                "Resolved fdk_dir from geometry (n_angles=%d): %s",
                geo.n_angles,
                fdk_dir,
            )
        else:
            # Try loading geometry.yaml from the canonical data_prep location.
            default_geo = Path("data_prep/configs/geometry.yaml")
            if default_geo.exists():
                geo = load_geometry_cfg(default_geo)
                fdk_dir = _BASE_DIR / f"fdk{geo.n_angles}"
                logger.info(
                    "Resolved fdk_dir from %s (n_angles=%d): %s",
                    default_geo,
                    geo.n_angles,
                    fdk_dir,
                )
            else:
                logger.error(
                    "No --fdk_dir given and no --geometry_config to derive it from. "
                    "Provide one of: --fdk_dir, --geometry_config, or run from the "
                    "project root (so configs/geometry.yaml is found automatically)."
                )
                return 1

    cfg = PrepareConfig(
        fdk_dir=fdk_dir,
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
