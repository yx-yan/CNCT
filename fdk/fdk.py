#!/usr/bin/env python3
"""Reconstruct CT volumes from projections using Feldkamp-Davis-Kress (FDK).

Pipeline stage 2/3.  For each case with projections.npy in PROJ_DIR:
  1. Load projections and rebuild the identical cone-beam geometry
  2. Run TIGRE's FDK filtered back-projection
  3. Save recon_fdk.npy (always) and optionally .nii.gz + PNG slices

Input:  PROJ_DIR/<case>/projections.npy  (from projection.py)
Output: FDK_DIR/<case>/recon_fdk.npy     (consumed by evaluation.py)
"""

import os
import glob

import numpy as np
import nibabel as nib
import tigre
import tigre.algorithms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, PROJ_DIR, FDK_DIR, N_ANGLES, MAX_CASES,
    CASE_START, CASE_END,
    MU_WATER, IMAGE_DPI, FDK_FILTER, SAVE_PNG, SAVE_NII,
)
from geometry import build_geometry, load_nifti_as_tigre, mu_to_hu

# Must match the angles used in projection.py
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES][CASE_START:CASE_END]
print(f"Found {len(cases)} cases: {[os.path.basename(c) for c in cases]}\n")


def save_recon_slices(recon, geo, case_out, case_name):
    """Save axial, coronal, and sagittal mid-slices with correct physical aspect ratios."""
    vmin, vmax = np.percentile(recon, [1, 99])
    dz, dy, dx = geo.dVoxel
    nz, ny, nx = recon.shape

    slices = {
        "axial":    (recon[nz // 2, :, :],  ny * dy, nx * dx),
        "coronal":  (recon[:, ny // 2, :],  nz * dz, nx * dx),
        "sagittal": (recon[:, :, nx // 2],  nz * dz, ny * dy),
    }
    for name, (img, phys_h, phys_w) in slices.items():
        scale = 6.0 / max(phys_h, phys_w)
        fig, ax = plt.subplots(figsize=(phys_w * scale, phys_h * scale))
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{case_name} — {name}")
        ax.axis("off")
        fig.savefig(os.path.join(case_out, f"recon_{name}.png"),
                    bbox_inches="tight", dpi=IMAGE_DPI)
        plt.close(fig)


for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    proj_path = os.path.join(PROJ_DIR, case_name, "projections.npy")
    case_out = os.path.join(FDK_DIR, case_name)

    if not os.path.exists(proj_path):
        print(f"  Skipping {case_name} — projections.npy not found")
        continue

    print(f"=== Reconstructing {case_name} ===")

    # --- Load projections ---
    projections = np.load(proj_path).astype(np.float32)
    print(f"  Projections shape: {projections.shape}")

    # --- Rebuild geometry from NIfTI header (must match projection.py) ---
    nii_img = nib.load(nii_path)
    nVoxel, voxel_sizes = load_nifti_as_tigre(nii_img)
    geo = build_geometry(nVoxel, voxel_sizes)

    # --- FDK reconstruction ---
    recon = tigre.algorithms.fdk(projections, geo, angles, filter=FDK_FILTER)
    print(f"  Reconstruction shape: {recon.shape}")

    # --- Save outputs ---
    os.makedirs(case_out, exist_ok=True)
    np.save(os.path.join(case_out, "recon_fdk.npy"), recon)

    if SAVE_PNG:
        save_recon_slices(recon, geo, case_out, case_name)

    if SAVE_NII:
        recon_hu = mu_to_hu(recon)
        recon_nii = nib.Nifti1Image(
            np.transpose(recon_hu, (2, 1, 0)),
            affine=nii_img.affine,
            header=nii_img.header,
        )
        nib.save(recon_nii, os.path.join(case_out, "recon_fdk.nii.gz"))

    print(f"  Saved to {case_out}/\n")

print("All cases done.")
