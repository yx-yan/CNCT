#!/usr/bin/env python3
"""Evaluate FDK reconstructions against ground-truth CT volumes.

Pipeline stage 3/3.  For each case with recon_fdk.npy in FDK_DIR:
  1. Load the ground-truth NIfTI and convert HU to mu (same units as FDK)
  2. Compute volumetric PSNR and mean per-slice SSIM
  3. Optionally save side-by-side comparison images (GT vs FDK vs difference)
  4. Write a CSV summary of all evaluated cases

Input:  FDK_DIR/<case>/recon_fdk.npy  (from fdk.py)
Output: EVAL_DIR/evaluation_results.csv
"""

import os
import glob
import csv

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import DATA_DIR, FDK_DIR, EVAL_DIR, MAX_CASES, IMAGE_DPI, SAVE_PNG
from geometry import hu_to_mu


def compute_psnr_ssim(gt, recon):
    """Compute volumetric PSNR and mean per-slice SSIM.

    Parameters
    ----------
    gt : ndarray, shape (Z, Y, X)
        Ground truth in mu units.
    recon : ndarray, shape (Z, Y, X)
        FDK reconstruction in mu units.

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


def save_comparison(gt, recon, dVoxel, case_out, case_name):
    """Save side-by-side GT vs reconstruction for axial, coronal, sagittal mid-slices."""
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
        axes[1].set_title("FDK Reconstruction")
        axes[1].axis("off")

        diff = np.abs(gt_img - rec_img)
        im = axes[2].imshow(diff, cmap="hot", aspect="auto")
        axes[2].set_title("Absolute Difference")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"{case_name} — {name}")
        fig.savefig(os.path.join(case_out, f"eval_{name}.png"),
                    bbox_inches="tight", dpi=IMAGE_DPI)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES]
print(f"Found {len(cases)} cases\n")

results = []

for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    recon_path = os.path.join(FDK_DIR, case_name, "recon_fdk.npy")
    case_out = os.path.join(EVAL_DIR, case_name)

    if not os.path.exists(recon_path):
        print(f"  Skipping {case_name} — recon_fdk.npy not found")
        continue

    print(f"=== Evaluating {case_name} ===")

    # --- Ground truth: NIfTI HU -> mu (same units as FDK output) ---
    nii_img = nib.load(nii_path)
    volume_hu = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)

    gt = hu_to_mu(np.transpose(volume_hu, (2, 1, 0)))  # (X,Y,Z) -> (Z,Y,X)
    dVoxel = np.array([voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]])

    # --- Load reconstruction ---
    recon = np.load(recon_path).astype(np.float32)
    print(f"  GT shape: {gt.shape}, Recon shape: {recon.shape}")

    if gt.shape != recon.shape:
        print(f"  WARNING: shape mismatch — skipping metrics")
        continue

    # --- Compute metrics ---
    psnr, ssim = compute_psnr_ssim(gt, recon)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    results.append({"case": case_name, "psnr_db": round(psnr, 4), "ssim": round(ssim, 6)})

    # --- Comparison images ---
    if SAVE_PNG:
        os.makedirs(case_out, exist_ok=True)
        save_comparison(gt, recon, dVoxel, case_out, case_name)
        print(f"  Comparison images saved to {case_out}/")

# --- CSV summary ---
if results:
    os.makedirs(EVAL_DIR, exist_ok=True)
    csv_path = os.path.join(EVAL_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "psnr_db", "ssim"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    psnr_vals = [r["psnr_db"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    print(f"\nSummary across {len(results)} cases:")
    print(f"  PSNR — mean: {np.mean(psnr_vals):.2f} dB, std: {np.std(psnr_vals):.2f} dB")
    print(f"  SSIM — mean: {np.mean(ssim_vals):.4f}, std: {np.std(ssim_vals):.4f}")

print("\nEvaluation done.")
