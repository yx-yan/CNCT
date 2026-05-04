"""Training metrics for the dual-domain cascade pipeline."""
from __future__ import annotations

import math

import torch

__all__ = ["compute_psnr", "PSNR_DATA_RANGE"]

# Locked PSNR data range. ``mu_max - mu_min = 0.08 - (-0.02) = 0.1`` is the
# physical span of the fixed-range mu normalisation window used throughout
# the project. Pinning ``data_range`` to this constant — rather than letting
# it drift with ``gt.max() - gt.min()`` on a per-case basis — keeps every
# PSNR figure (training logs, validation, evaluation, thesis plots) on the
# same scale so cross-sample numbers are directly comparable.
PSNR_DATA_RANGE: float = 0.1


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mu_min: float,
    mu_max: float,
) -> float:
    """Peak signal-to-noise ratio (dB) on the mu scale with ``data_range=0.1``.

    The trainer passes tensors in the normalised ``[-1, 1]`` range. This
    function denormalises them back to mu units before computing the
    squared error, then reports PSNR with ``data_range = PSNR_DATA_RANGE``
    so every PSNR number in the project lives on the same physical scale.

    Args:
        pred: Prediction tensor in the normalised ``[-1, 1]`` range.
        target: Ground-truth tensor with the same shape as ``pred``, also in
            the normalised ``[-1, 1]`` range.
        mu_min: Lower bound of the fixed mu normalisation range (e.g. -0.02).
        mu_max: Upper bound of the fixed mu normalisation range (e.g.  0.08).

    Returns:
        PSNR in decibels. Numerically saturated to ``100.0`` when the
        mean squared error falls below ``1e-12`` so downstream logging
        never hits ``inf``.
    """
    half_range = 0.5 * (mu_max - mu_min)
    pred_mu = (pred + 1.0) * half_range + mu_min
    target_mu = (target + 1.0) * half_range + mu_min

    mse = torch.mean((pred_mu - target_mu) ** 2).item()
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10(PSNR_DATA_RANGE * PSNR_DATA_RANGE / mse)
