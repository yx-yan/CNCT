import os
import glob
import tigre
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    DATA_DIR, OUTPUT_DIR, N_ANGLES, MAX_CASES,
    DSO_SCALE, DSD_SCALE, DETECTOR_COL_MARGIN,
    ACCURACY, MU_WATER, PROJ_SAVE_EVERY, IMAGE_DPI,
    SAVE_PNG,
)

angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES]
print(f"Found {len(cases)} cases: {[os.path.basename(c) for c in cases]}\n")

for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    print(f"=== Processing {case_name} ===")

    # --- Load volume ---
    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata().astype(np.float32)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    print(f"  Volume shape: {volume.shape}, voxel sizes (mm): {voxel_sizes}")

    # TIGRE expects (Z, Y, X)
    volume = np.transpose(volume, (2, 1, 0)).copy()
    nVoxel = np.array(volume.shape, dtype=np.int64)

    # Convert HU to linear attenuation coefficients (mu_water ≈ 0.02 mm⁻¹)
    volume = (volume + 1000.0) / 1000.0 * MU_WATER
    volume = np.clip(volume, 0, None)
    volume = np.ascontiguousarray(volume)

    # --- Geometry (per-case, derived from NIfTI header) ---
    geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = nVoxel
    geo.dVoxel = np.array([voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]])
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # DSO scaled to volume diagonal so source is always outside the object
    max_radius = np.sqrt((geo.sVoxel[1] / 2) ** 2 + (geo.sVoxel[2] / 2) ** 2)
    geo.DSO = max_radius * DSO_SCALE
    geo.DSD = geo.DSO * DSD_SCALE

    magnification = geo.DSD / geo.DSO
    # Detector columns cover max(nY, nX) with margin to prevent truncation
    geo.nDetector = np.array([nVoxel[0], max(nVoxel[1], nVoxel[2])])
    geo.dDetector = np.array([geo.dVoxel[0] * magnification,
                               geo.dVoxel[2] * magnification * DETECTOR_COL_MARGIN])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])
    geo.accuracy = ACCURACY

    # --- Forward projection ---
    projections = tigre.Ax(volume, geo, angles)
    print(f"  Projections shape: {projections.shape}")

    # --- Save outputs ---
    case_out = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(case_out, exist_ok=True)

    np.save(os.path.join(case_out, "projections.npy"), projections)

    if SAVE_PNG:
        # Global contrast range across all angles
        vmin_proj, vmax_proj = projections.min(), projections.max()
        # Figure sized to physical detector dimensions (mm), normalised so max dim = 6 inches
        phys_det_h = geo.nDetector[0] * geo.dDetector[0]
        phys_det_w = geo.nDetector[1] * geo.dDetector[1]
        det_scale = 6.0 / max(phys_det_h, phys_det_w)

        # Save every Nth projection angle as PNG
        for i in range(0, N_ANGLES, PROJ_SAVE_EVERY):
            fig, ax = plt.subplots(figsize=(phys_det_w * det_scale, phys_det_h * det_scale))
            ax.imshow(projections[i], cmap="gray", aspect="auto",
                      vmin=vmin_proj, vmax=vmax_proj)
            ax.set_title(f"{case_name} — {np.degrees(angles[i]):.1f}°")
            ax.axis("off")
            fig.savefig(os.path.join(case_out, f"proj_{i:03d}.png"), bbox_inches="tight", dpi=IMAGE_DPI)
            plt.close(fig)

        # Save mid axial slice of volume
        # volume[mid_z] shape is (nY, nX); physical size is (nVoxel[1]*dVoxel[1]) x (nVoxel[2]*dVoxel[2])
        phys_vol_h = geo.nVoxel[1] * geo.dVoxel[1]
        phys_vol_w = geo.nVoxel[2] * geo.dVoxel[2]
        vol_scale = 6.0 / max(phys_vol_h, phys_vol_w)
        mid_z = volume.shape[0] // 2
        fig, ax = plt.subplots(figsize=(phys_vol_w * vol_scale, phys_vol_h * vol_scale))
        ax.imshow(volume[mid_z], cmap="gray", aspect="auto")
        ax.set_title(f"{case_name} — axial slice z={mid_z}")
        ax.axis("off")
        fig.savefig(os.path.join(case_out, "volume_axial_mid.png"), bbox_inches="tight", dpi=IMAGE_DPI)
        plt.close(fig)

    print(f"  Saved to {case_out}/\n")

print("All cases done.")
