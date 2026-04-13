"""Comparison PNGs + PSNR/SSIM for dual-domain inference outputs.

Self-contained visualisation helpers for the ``cnct`` package. Mirrors the
layout of :mod:`cnct_dataprep.evaluation.visualize` but is duplicated so the
deep-learning package has no import-time dependency on ``cnct_dataprep``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..geometry.conversions import hu_to_mu
from ..utils.io import safe_load_nifti

logger = logging.getLogger(__name__)


def load_gt_as_mu(
    nii_path: Path, mu_water: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a ground-truth NIfTI volume as ``(Z, Y, X)`` mu + voxel spacing."""
    nii_img = safe_load_nifti(nii_path)
    volume_hu = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    gt = hu_to_mu(np.transpose(volume_hu, (2, 1, 0)), mu_water)
    dVoxel = np.array(
        [voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]], dtype=np.float32
    )
    return gt, dVoxel


def compute_psnr_ssim(
    gt: np.ndarray, recon: np.ndarray
) -> Tuple[float, float]:
    """Volumetric PSNR + mean per-slice SSIM in mu units."""
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


def save_comparison(
    gt: np.ndarray,
    fdk: np.ndarray,
    recon: np.ndarray,
    dVoxel: np.ndarray,
    case_out: Path,
    case_id: str,
    recon_label: str,
    image_dpi: int,
    psnr: float,
    ssim: float,
    fdk_psnr: float | None = None,
    fdk_ssim: float | None = None,
) -> None:
    """Save ``GT | FDK | recon | |diff|`` comparison PNGs for one case."""
    if gt.shape != recon.shape or gt.shape != fdk.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, fdk={fdk.shape}, "
            f"recon={recon.shape}"
        )

    case_out.mkdir(parents=True, exist_ok=True)

    dz, dy, dx = (float(v) for v in dVoxel)
    nz, ny, nx = gt.shape
    vmin, vmax = np.percentile(gt, [1, 99])

    slice_spec = [
        (
            "axial",
            gt[nz // 2, :, :],
            fdk[nz // 2, :, :],
            recon[nz // 2, :, :],
            ny * dy,
            nx * dx,
        ),
        (
            "coronal",
            gt[:, ny // 2, :],
            fdk[:, ny // 2, :],
            recon[:, ny // 2, :],
            nz * dz,
            nx * dx,
        ),
        (
            "sagittal",
            gt[:, :, nx // 2],
            fdk[:, :, nx // 2],
            recon[:, :, nx // 2],
            nz * dz,
            ny * dy,
        ),
    ]

    fdk_title = "FDK Input"
    if fdk_psnr is not None and fdk_ssim is not None:
        fdk_title += f"\nPSNR: {fdk_psnr:.2f} dB | SSIM: {fdk_ssim:.4f}"
    rec_title = recon_label + f"\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"

    for name, gt_img, fdk_img, rec_img, phys_h, phys_w in slice_spec:
        panel_h = 5.0
        panel_w = phys_w / phys_h * panel_h
        fig, axes = plt.subplots(1, 4, figsize=(4 * panel_w, panel_h))

        axes[0].imshow(gt_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(fdk_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title(fdk_title)
        axes[1].axis("off")

        axes[2].imshow(rec_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[2].set_title(rec_title)
        axes[2].axis("off")

        diff = np.abs(gt_img - rec_img)
        im = axes[3].imshow(diff, cmap="hot", aspect="auto")
        axes[3].set_title("|GT - Prediction|")
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        fig.suptitle(f"{case_id} — {name}")
        fig.savefig(
            case_out / f"pred_{name}.png",
            bbox_inches="tight",
            dpi=image_dpi,
        )
        plt.close(fig)
