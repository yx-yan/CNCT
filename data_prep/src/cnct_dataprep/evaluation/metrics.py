"""Volumetric PSNR and 3D SSIM for CT reconstructions.

Every metric in this module is pinned to ``data_range = SSIM_DATA_RANGE``
(= 0.1, the physical span of the fixed-range mu normalisation window used
across the project). Locking ``data_range`` — rather than deriving it from
``gt.max() - gt.min()`` on a per-case basis — keeps PSNR/SSIM numbers
directly comparable across cases and across the training / inference /
thesis-figure pipelines.

SSIM is computed as a single 3D measurement with ``win_size = SSIM_WINDOW``
(= 7), matching the cubic window used by the training loss, instead of the
legacy mean-per-slice 2D formulation.

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

SSIM_DATA_RANGE: float = 0.1
SSIM_WINDOW: int = 7


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
) -> Tuple[float, float, float]:
    """Compute volumetric PSNR, 3D SSIM, and RMSE (all in mu units).

    PSNR and SSIM use ``data_range = SSIM_DATA_RANGE`` (= 0.1), the fixed
    physical mu span. SSIM is evaluated as a single 3D measurement with a
    cubic window of side ``SSIM_WINDOW`` (= 7) rather than the legacy
    mean-per-slice 2D average. RMSE is reported in the same mu units so it
    is directly comparable across cases.

    Args:
        gt: Ground-truth volume in mu units, shape ``(Z, Y, X)``.
        recon: Reconstructed volume in mu units, same shape as ``gt``.

    Returns:
        A tuple ``(psnr_db, ssim_3d, rmse)``.

    Raises:
        ValueError: If ``gt`` and ``recon`` have different shapes.
    """
    if gt.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, recon={recon.shape}"
        )

    psnr = peak_signal_noise_ratio(gt, recon, data_range=SSIM_DATA_RANGE)
    ssim = structural_similarity(
        gt, recon, win_size=SSIM_WINDOW, data_range=SSIM_DATA_RANGE
    )
    rmse = float(np.sqrt(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)))
    return float(psnr), float(ssim), rmse
