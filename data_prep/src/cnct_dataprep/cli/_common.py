"""Shared argument parsing for ``cnct_dataprep`` CLI entry points."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_stage_parser(description: str, config_help: str) -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` shared by every stage CLI.

    Args:
        description: One-line description shown in ``--help``.
        config_help: Help string for the required ``--config`` flag.

    Returns:
        An argparse parser with ``--config``, ``--log-file``, ``--log-level``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=config_help,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path (parent dirs are created).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root logger level (default: INFO).",
    )
    return parser
