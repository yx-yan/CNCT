"""Centralized logging configuration for the ``cnct`` package.

Typical usage from an entry-point script::

    from cnct.utils.logging import configure_root_logger, get_logger

    configure_root_logger(log_file="logs/train.log", level="INFO")
    logger = get_logger(__name__)
    logger.info("Training started")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

PathLike = Union[str, Path]


def configure_root_logger(
    log_file: Optional[PathLike] = None,
    level: Union[int, str] = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    date_fmt: str = _DEFAULT_DATE_FORMAT,
) -> None:
    """Configure the root logger with console (and optional file) handlers.

    Should be called exactly once per process, at the start of an entry-point
    script. Repeated calls are idempotent — existing handlers are removed before
    new ones are installed, so re-configuring is safe.

    Args:
        log_file: Optional filesystem path for a file log. The parent directory
            is created if it does not already exist.
        level: Logging level, either a string (``"INFO"``) or a numeric constant
            (``logging.DEBUG``).
        fmt: Format string passed to :class:`logging.Formatter`.
        date_fmt: Date format string passed to :class:`logging.Formatter`.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` that inherits handlers from the root logger
        configured by :func:`configure_root_logger`.
    """
    return logging.getLogger(name)
