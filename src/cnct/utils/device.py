"""Device resolution helpers for the ``cnct`` package."""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Resolve a :class:`torch.device` from a config string.

    Args:
        requested: One of:

            - ``None`` or ``"auto"`` — use CUDA if available, else CPU.
            - ``"cpu"`` — force CPU.
            - ``"cuda"`` / ``"cuda:0"`` / etc. — explicit CUDA device.

    Returns:
        The resolved :class:`torch.device`.

    Raises:
        RuntimeError: If a CUDA device is explicitly requested but
            :func:`torch.cuda.is_available` returns ``False``.
    """
    if requested is None or requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{requested}' but "
            f"torch.cuda.is_available() is False"
        )

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        logger.info("Using CUDA device %d (%s)", idx, name)
    else:
        logger.info("Using CPU device")

    return device
