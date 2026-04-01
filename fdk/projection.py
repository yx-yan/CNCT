#!/usr/bin/env python3
"""Forward-project CT volumes into simulated cone-beam X-ray projections.

Pipeline stage 1/3.  For each NIfTI volume in DATA_DIR:
  1. Load and transpose to TIGRE's (Z, Y, X) axis order
  2. Convert HU to linear attenuation (mu)
  3. Build per-case cone-beam geometry from the NIfTI header
  4. Run TIGRE's GPU-accelerated Siddon ray-tracing (tigre.Ax)
  5. Save projections.npy (and optionally PNG previews) to PROJ_DIR/<case>/

Output is consumed by fdk.py (stage 2).
"""

import os
import glob

import numpy as np
import nibabel as nib
import tigre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, PROJ_DIR, N_ANGLES, MAX_CASES,
    CASE_START, CASE_END,
    PROJ_SAVE_EVERY, IMAGE_DPI, SAVE_PNG,
)
from geometry import build_geometry, hu_to_mu

# Projection angles: uniform 360-degree coverage
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES][CASE_START:CASE_END]
print(f"Found {len(cases)} cases: {[os.path.basename(c) for c in cases]}\n")

for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    print(f"=== Processing {case_name} ===")

    # --- Load and preprocess volume ---
    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    print(f"  Volume shape: {volume.shape}, voxel sizes (mm): {voxel_sizes}")

    # Transpose from NIfTI (X, Y, Z) to TIGRE (Z, Y, X)
    volume = np.transpose(volume, (2, 1, 0)).copy()
    nVoxel = np.array(volume.shape, dtype=np.int64)

    # Convert HU to linear attenuation and ensure contiguous memory for GPU
    volume = hu_to_mu(volume)
    volume = np.ascontiguousarray(volume)

    # --- Build geometry and project ---
    geo = build_geometry(nVoxel, voxel_sizes)
    projections = tigre.Ax(volume, geo, angles)
    print(f"  Projections shape: {projections.shape}")

    # --- Save outputs ---
    case_out = os.path.join(PROJ_DIR, case_name)
    os.makedirs(case_out, exist_ok=True)
    np.save(os.path.join(case_out, "projections.npy"), projections)

    if SAVE_PNG:
        # Global contrast range across all angles for consistent visualisation
        vmin_proj, vmax_proj = projections.min(), projections.max()

        # Figure sized to physical detector dimensions (mm)
        phys_det_h = geo.nDetector[0] * geo.dDetector[0]
        phys_det_w = geo.nDetector[1] * geo.dDetector[1]
        det_scale = 6.0 / max(phys_det_h, phys_det_w)

        for i in range(0, N_ANGLES, PROJ_SAVE_EVERY):
            fig, ax = plt.subplots(figsize=(phys_det_w * det_scale, phys_det_h * det_scale))
            ax.imshow(projections[i], cmap="gray", aspect="auto",
                      vmin=vmin_proj, vmax=vmax_proj)
            ax.set_title(f"{case_name} — {np.degrees(angles[i]):.1f}°")
            ax.axis("off")
            fig.savefig(os.path.join(case_out, f"proj_{i:03d}.png"),
                        bbox_inches="tight", dpi=IMAGE_DPI)
            plt.close(fig)

        # Mid axial slice of the input volume
        phys_vol_h = geo.nVoxel[1] * geo.dVoxel[1]
        phys_vol_w = geo.nVoxel[2] * geo.dVoxel[2]
        vol_scale = 6.0 / max(phys_vol_h, phys_vol_w)
        mid_z = volume.shape[0] // 2
        fig, ax = plt.subplots(figsize=(phys_vol_w * vol_scale, phys_vol_h * vol_scale))
        ax.imshow(volume[mid_z], cmap="gray", aspect="auto")
        ax.set_title(f"{case_name} — axial slice z={mid_z}")
        ax.axis("off")
        fig.savefig(os.path.join(case_out, "volume_axial_mid.png"),
                    bbox_inches="tight", dpi=IMAGE_DPI)
        plt.close(fig)

    print(f"  Saved to {case_out}/\n")

print("All cases done.")
