"""``cnct-evaluation`` CLI — evaluate FDK reconstructions against ground truth."""
from __future__ import annotations

import sys

from ..config.loader import load_evaluation_cfg
from ..evaluation.runner import run_evaluation
from ..utils.logging import configure_root_logger, get_logger
from ._common import build_stage_parser


def main() -> int:
    """Entry point for ``cnct-evaluation``.

    Returns:
        ``0`` on success, ``2`` if the config could not be loaded.
    """
    parser = build_stage_parser(
        description="Evaluate FDK reconstructions against ground-truth volumes.",
        config_help="Path to an evaluation YAML config.",
    )
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file, level=args.log_level)
    logger = get_logger("cnct_dataprep.cli.evaluation")

    try:
        cfg = load_evaluation_cfg(args.config)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("Failed to load config %s: %s", args.config, exc)
        return 2

    logger.info("Loaded EvaluationCfg from %s", args.config)
    run_evaluation(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
