"""Training metrics for the dual-domain cascade pipeline."""
from __future__ import annotations

import math

import torch

__all__ = ["compute_psnr"]


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB) assuming a ``[-1, 1]`` data range.

    Both ``pred`` and ``target`` are expected to carry normalised
    reconstructions in the ``[-1, 1]`` range used throughout the
    dual-domain pipeline, so the data range is hard-coded to 2 (and
    ``data_range² = 4``).

    Args:
        pred: Prediction tensor of any shape.
        target: Ground-truth tensor with the same shape as ``pred``.

    Returns:
        PSNR in decibels. Numerically saturated to ``100.0`` when the
        mean squared error falls below ``1e-10`` so downstream logging
        never hits ``inf``.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)
