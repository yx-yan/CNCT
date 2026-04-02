"""Shared evaluation utilities for FDK and UNet pipelines.

Provides PSNR/SSIM computation, side-by-side comparison image generation,
and ground-truth loading used by both fdk/evaluation.py and 3dunet/evaluation.py.
"""

import os

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from geometry import hu_to_mu


def load_gt_as_mu(nii_path):
    """Load a ground-truth NIfTI volume and return it in mu units (Z, Y, X).

    Parameters
    ----------
    nii_path : str
        Path to a .nii.gz file in HU units.

    Returns
    -------
    gt : ndarray, shape (Z, Y, X), float32
        Ground truth in linear attenuation (mu) units.
    dVoxel : ndarray, shape (3,), float32
        Voxel spacing in (Z, Y, X) order.
    """
    nii_img = nib.load(nii_path)
    volume_hu = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)

    gt = hu_to_mu(np.transpose(volume_hu, (2, 1, 0)))  # (X,Y,Z) -> (Z,Y,X)
    dVoxel = np.array([voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]])
    return gt, dVoxel


def compute_psnr_ssim(gt, recon):
    """Compute volumetric PSNR and mean per-slice SSIM.

    Parameters
    ----------
    gt : ndarray, shape (Z, Y, X)
        Ground truth in mu units.
    recon : ndarray, shape (Z, Y, X)
        Reconstruction in mu units.

    Returns
    -------
    psnr : float
        Peak signal-to-noise ratio in dB.
    ssim : float
        Mean axial-slice structural similarity index.
    """
    data_range = gt.max() - gt.min()
    psnr = peak_signal_noise_ratio(gt, recon, data_range=data_range)
    ssim_scores = [
        structural_similarity(gt[z], recon[z], data_range=data_range)
        for z in range(gt.shape[0])
    ]
    return psnr, float(np.mean(ssim_scores))


def save_comparison(gt, recon, dVoxel, case_out, case_name, recon_label, image_dpi):
    """Save side-by-side GT vs reconstruction for axial, coronal, sagittal mid-slices.

    Parameters
    ----------
    gt : ndarray, shape (Z, Y, X)
    recon : ndarray, shape (Z, Y, X)
    dVoxel : ndarray, shape (3,)
        Voxel spacing in (Z, Y, X) order.
    case_out : str
        Output directory for this case.
    case_name : str
        Case identifier for the figure title.
    recon_label : str
        Label for the reconstruction panel (e.g. "FDK Reconstruction" or "UNet Prediction").
    image_dpi : int
        DPI for saved PNGs.
    """
    dz, dy, dx = dVoxel
    nz, ny, nx = gt.shape
    vmin, vmax = np.percentile(gt, [1, 99])

    slices = {
        "axial":    (gt[nz // 2],           recon[nz // 2],           ny * dy, nx * dx),
        "coronal":  (gt[:, ny // 2],        recon[:, ny // 2],        nz * dz, nx * dx),
        "sagittal": (gt[:, :, nx // 2],     recon[:, :, nx // 2],     nz * dz, ny * dy),
    }
    for name, (gt_img, rec_img, phys_h, phys_w) in slices.items():
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

        fig.suptitle(f"{case_name} — {name}")
        fig.savefig(os.path.join(case_out, f"eval_{name}.png"),
                    bbox_inches="tight", dpi=image_dpi)
        plt.close(fig)