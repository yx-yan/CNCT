"""``cnct-fdk`` CLI — run TIGRE FDK reconstruction."""
from __future__ import annotations

import sys

from ..config.loader import load_fdk_cfg
from ..reconstruction.fdk import run_fdk
from ..utils.logging import configure_root_logger, get_logger
from ._common import build_stage_parser


def main() -> int:
    """Entry point for ``cnct-fdk``.

    Returns:
        ``0`` on success, ``2`` if the config could not be loaded.
    """
    parser = build_stage_parser(
        description="Run TIGRE FDK reconstruction for every configured case.",
        config_help="Path to an FDK YAML config.",
    )
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file, level=args.log_level)
    logger = get_logger("cnct_dataprep.cli.fdk")

    try:
        cfg = load_fdk_cfg(args.config)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("Failed to load config %s: %s", args.config, exc)
        return 2

    logger.info("Loaded FdkCfg from %s", args.config)
    run_fdk(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
