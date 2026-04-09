"""Volumetric PSNR and mean-per-slice SSIM for CT reconstructions.

This module is shape- and unit-aware: PSNR's ``data_range`` is derived from
the ground-truth max - min so both PSNR and SSIM are consistently scaled.

Ground-truth loading is also here because the conversion pipeline (HU -> mu,
axis reorder) is tightly coupled to the unit convention assumed by the
metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..geometry.conversions import hu_to_mu
from ..utils.io import safe_load_nifti

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_gt_as_mu(
    nii_path: PathLike,
    mu_water: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a ground-truth NIfTI volume and return it in mu units (Z, Y, X).

    The NIfTI axis order (X, Y, Z) is transposed to TIGRE's convention
    (Z, Y, X), Hounsfield values are converted to linear attenuation, and
    voxel spacing is reordered so it matches the volume axes.

    Args:
        nii_path: Path to a ``.nii`` or ``.nii.gz`` file in Hounsfield units.
        mu_water: Linear attenuation of water (mm⁻¹) for HU -> mu conversion.

    Returns:
        A tuple ``(gt, dVoxel)`` where:

            * ``gt`` is a ``(Z, Y, X)`` ``float32`` volume in mu units.
            * ``dVoxel`` is a ``(3,)`` ``float32`` array of voxel spacing in
              (Z, Y, X) order.

    Raises:
        FileNotFoundError: If ``nii_path`` does not exist.
    """
    nii_img = safe_load_nifti(nii_path)
    volume_hu = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(
        nii_img.header.get_zooms()[:3], dtype=np.float32
    )

    # (X, Y, Z) -> (Z, Y, X)
    gt = hu_to_mu(np.transpose(volume_hu, (2, 1, 0)), mu_water)
    dVoxel = np.array(
        [voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]], dtype=np.float32
    )
    return gt, dVoxel


def compute_psnr_ssim(
    gt: np.ndarray,
    recon: np.ndarray,
) -> Tuple[float, float]:
    """Compute volumetric PSNR and mean per-slice SSIM.

    ``data_range`` is derived from ``gt.max() - gt.min()`` and used for both
    metrics so they remain comparable across cases with different intensity
    ranges.

    Args:
        gt: Ground-truth volume in mu units, shape ``(Z, Y, X)``.
        recon: Reconstructed volume in mu units, same shape as ``gt``.

    Returns:
        A tuple ``(psnr_db, mean_ssim)``.

    Raises:
        ValueError: If ``gt`` and ``recon`` have different shapes.
    """
    if gt.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, recon={recon.shape}"
        )

    data_range = float(gt.max() - gt.min())
    psnr = peak_signal_noise_ratio(gt, recon, data_range=data_range)

    ssim_scores = [
        structural_similarity(gt[z], recon[z], data_range=data_range)
        for z in range(gt.shape[0])
    ]
    return float(psnr), float(np.mean(ssim_scores))
