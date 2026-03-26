import os
import glob
import tigre
import tigre.algorithms
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    DATA_DIR, OUTPUT_DIR, N_ANGLES, MAX_CASES,
    DSO_SCALE, DSD_SCALE, DETECTOR_COL_MARGIN,
    ACCURACY, IMAGE_DPI,
)

angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES]
print(f"Found {len(cases)} cases: {[os.path.basename(c) for c in cases]}\n")


def build_geometry(nVoxel, voxel_sizes):
    """Build TIGRE geometry matching projection.py exactly."""
    geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = nVoxel
    geo.dVoxel = np.array([voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]])
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    max_radius = np.sqrt((geo.sVoxel[1] / 2) ** 2 + (geo.sVoxel[2] / 2) ** 2)
    geo.DSO = max_radius * DSO_SCALE
    geo.DSD = geo.DSO * DSD_SCALE

    magnification = geo.DSD / geo.DSO
    geo.nDetector = np.array([nVoxel[0], max(nVoxel[1], nVoxel[2])])
    geo.dDetector = np.array([geo.dVoxel[0] * magnification,
                               geo.dVoxel[2] * magnification * DETECTOR_COL_MARGIN])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])
    geo.accuracy = ACCURACY
    return geo


def save_recon_slices(recon, geo, case_out, case_name):
    """Save axial, coronal, and sagittal mid-slices with correct physical aspect ratios."""
    vmin, vmax = np.percentile(recon, [1, 99])
    dz, dy, dx = geo.dVoxel
    slices = {
        # shape (Y, X) → aspect = dx/dy
        "axial":    (recon[recon.shape[0] // 2, :, :], dx / dy),
        # shape (Z, X) → aspect = dx/dz
        "coronal":  (recon[:, recon.shape[1] // 2, :], dx / dz),
        # shape (Z, Y) → aspect = dy/dz
        "sagittal": (recon[:, :, recon.shape[2] // 2], dy / dz),
    }
    for name, (img, aspect) in slices.items():
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
        ax.set_title(f"{case_name} — {name}")
        ax.axis("off")
        fig.savefig(os.path.join(case_out, f"recon_{name}.png"), bbox_inches="tight", dpi=IMAGE_DPI)
        plt.close(fig)


for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    case_out = os.path.join(OUTPUT_DIR, case_name)
    proj_path = os.path.join(case_out, "projections.npy")

    if not os.path.exists(proj_path):
        print(f"  Skipping {case_name} — projections.npy not found")
        continue

    print(f"=== Reconstructing {case_name} ===")

    # --- Load projections ---
    projections = np.load(proj_path).astype(np.float32)
    print(f"  Projections shape: {projections.shape}")

    # --- Rebuild geometry from NIfTI header (must match projection.py) ---
    nii_img = nib.load(nii_path)
    volume_shape = nii_img.get_fdata().shape  # (X, Y, Z)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    nVoxel = np.array([volume_shape[2], volume_shape[1], volume_shape[0]], dtype=np.int64)

    geo = build_geometry(nVoxel, voxel_sizes)

    # --- FDK reconstruction ---
    recon = tigre.algorithms.fdk(projections, geo, angles)
    print(f"  Reconstruction shape: {recon.shape}")

    # --- Save reconstruction ---
    np.save(os.path.join(case_out, "recon_fdk.npy"), recon)
    save_recon_slices(recon, geo, case_out, case_name)

    print(f"  Saved to {case_out}/\n")

print("All cases done.")
