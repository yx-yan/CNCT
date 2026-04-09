"""``cnct-projection`` CLI — run TIGRE forward projection."""
from __future__ import annotations

import sys

from ..config.loader import load_projection_cfg
from ..projection.forward import run_projection
from ..utils.logging import configure_root_logger, get_logger
from ._common import build_stage_parser


def main() -> int:
    """Entry point for ``cnct-projection``.

    Returns:
        ``0`` on success, ``2`` if the config could not be loaded.
    """
    parser = build_stage_parser(
        description="Run TIGRE forward projection for every configured case.",
        config_help="Path to a projection YAML config.",
    )
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file, level=args.log_level)
    logger = get_logger("cnct_dataprep.cli.projection")

    try:
        cfg = load_projection_cfg(args.config)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("Failed to load config %s: %s", args.config, exc)
        return 2

    logger.info("Loaded ProjectionCfg from %s", args.config)
    run_projection(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
