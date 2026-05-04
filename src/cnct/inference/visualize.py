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

SSIM_DATA_RANGE: float = 0.1
SSIM_WINDOW: int = 7


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
) -> Tuple[float, float, float]:
    """Volumetric PSNR + 3D SSIM + RMSE in mu units.

    PSNR and SSIM are locked to ``data_range = SSIM_DATA_RANGE`` (= 0.1).
    SSIM is a single 3D measurement with a cubic window of side
    :data:`SSIM_WINDOW` (= 7), matching the training loss and the
    data-prep evaluation script. RMSE is reported in the same mu units.
    """
    if gt.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, recon={recon.shape}"
        )
    psnr = peak_signal_noise_ratio(gt, recon, data_range=SSIM_DATA_RANGE)
    ssim = structural_similarity(
        gt, recon, win_size=SSIM_WINDOW, data_range=SSIM_DATA_RANGE
    )
    rmse = float(
        np.sqrt(
            np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
        )
    )
    return float(psnr), float(ssim), rmse


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
    rmse: float | None = None,
    fdk_psnr: float | None = None,
    fdk_ssim: float | None = None,
    fdk_rmse: float | None = None,
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
        if fdk_rmse is not None:
            fdk_title += f" | RMSE: {fdk_rmse:.6f}"
    rec_title = recon_label + f"\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"
    if rmse is not None:
        rec_title += f" | RMSE: {rmse:.6f}"

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


def save_individual_panels(
    gt: np.ndarray,
    fdk: np.ndarray,
    recon: np.ndarray,
    dVoxel: np.ndarray,
    case_out: Path,
    image_dpi: int,
) -> None:
    """Save paper-ready single-panel PNGs (no titles, no axes).

    Emits one PNG per panel (GT / FDK / Pred / |diff|) for each of the
    three slice planes (axial / coronal / sagittal) into
    ``case_out / "individual" /``. Aspect ratio reflects physical voxel
    spacing; window/level matches :func:`save_comparison` so the panels
    are directly comparable.
    """
    if gt.shape != recon.shape or gt.shape != fdk.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, fdk={fdk.shape}, "
            f"recon={recon.shape}"
        )

    out = case_out / "individual"
    out.mkdir(parents=True, exist_ok=True)

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

    for name, gt_img, fdk_img, rec_img, phys_h, phys_w in slice_spec:
        diff = np.abs(gt_img - rec_img)
        panels = (
            ("gt", gt_img, "gray", vmin, vmax),
            ("fdk", fdk_img, "gray", vmin, vmax),
            ("pred", rec_img, "gray", vmin, vmax),
            ("diff", diff, "hot", None, None),
        )
        for kind, img, cmap, vlo, vhi in panels:
            _save_clean_panel(
                img,
                out / f"{name}_{kind}.png",
                cmap=cmap,
                vmin=vlo,
                vmax=vhi,
                phys_h=phys_h,
                phys_w=phys_w,
                image_dpi=image_dpi,
            )


def _save_clean_panel(
    img: np.ndarray,
    out_path: Path,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    phys_h: float,
    phys_w: float,
    image_dpi: int,
) -> None:
    panel_h = 5.0
    panel_w = phys_w / phys_h * panel_h
    fig = plt.figure(figsize=(panel_w, panel_h))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=image_dpi, pad_inches=0)
    plt.close(fig)
