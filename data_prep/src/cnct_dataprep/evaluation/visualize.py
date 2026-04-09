"""Side-by-side comparison images for FDK evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

logger = logging.getLogger(__name__)


def save_comparison(
    gt: np.ndarray,
    recon: np.ndarray,
    dVoxel: np.ndarray,
    case_out: Path,
    case_id: str,
    recon_label: str,
    image_dpi: int,
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
) -> None:
    """Save axial, coronal, and sagittal comparison PNGs for one case.

    Each PNG is a 1×3 grid::

        Ground Truth | <recon_label> | Absolute Difference

    Slice contrast uses the 1st and 99th percentiles of the ground truth so
    windows are consistent across panels and cases. Figure sizes reflect
    physical voxel dimensions in mm, so aspect ratios are correct.

    Args:
        gt: Ground-truth volume, shape ``(Z, Y, X)``, mu units.
        recon: Reconstructed volume, same shape as ``gt``, mu units.
        dVoxel: Voxel spacing in (Z, Y, X) order, shape ``(3,)``.
        case_out: Directory to save PNGs into (created if missing).
        case_id: Case identifier for figure titles.
        recon_label: Label for the reconstruction panel (e.g.
            ``"FDK Reconstruction"``).
        image_dpi: DPI for saved PNGs.
        psnr: Optional PSNR value to display in the suptitle.
        ssim: Optional SSIM value to display in the suptitle.

    Raises:
        ValueError: If ``gt`` and ``recon`` have different shapes.
    """
    if gt.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, recon={recon.shape}"
        )

    case_out.mkdir(parents=True, exist_ok=True)

    dz, dy, dx = (float(v) for v in dVoxel)
    nz, ny, nx = gt.shape
    vmin, vmax = np.percentile(gt, [1, 99])

    slice_spec = [
        ("axial", gt[nz // 2, :, :], recon[nz // 2, :, :], ny * dy, nx * dx),
        ("coronal", gt[:, ny // 2, :], recon[:, ny // 2, :], nz * dz, nx * dx),
        ("sagittal", gt[:, :, nx // 2], recon[:, :, nx // 2], nz * dz, ny * dy),
    ]

    for name, gt_img, rec_img, phys_h, phys_w in slice_spec:
        panel_h = 5.0
        panel_w = phys_w / phys_h * panel_h
        fig, axes = plt.subplots(1, 3, figsize=(3 * panel_w, panel_h))

        axes[0].imshow(gt_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(rec_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title(recon_label)
        axes[1].axis("off")

        diff = np.abs(gt_img - rec_img)
        im = axes[2].imshow(diff, cmap="hot", aspect="auto")
        axes[2].set_title("Absolute Difference")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        title = case_id
        if psnr is not None and ssim is not None:
            title += f" — PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}"
        title += f" — {name}"
        fig.suptitle(title)

        fig.savefig(
            case_out / f"eval_{name}.png",
            bbox_inches="tight",
            dpi=image_dpi,
        )
        plt.close(fig)
