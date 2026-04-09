"""Deterministic seeding for reproducible PyTorch runs."""
from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int], deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) random number generators.

    Args:
        seed: Seed value. If ``None``, the function is a no-op and RNG state is
            left untouched (useful for "random but fast" runs).
        deterministic: If ``True``, additionally sets cuDNN to deterministic
            mode (``torch.backends.cudnn.deterministic = True`` and
            ``benchmark = False``). This trades throughput for bit-level
            reproducibility, and is required whenever the math-preservation
            golden-reference check is active.
    """
    if seed is None:
        logger.info("Seeding skipped (seed=None)")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(
        "Seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic
    )
