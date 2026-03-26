import os
import glob
import tigre
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "output"
N_ANGLES = 1000

angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))
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
    mu_water = 0.02
    volume = (volume + 1000.0) / 1000.0 * mu_water
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
    geo.DSO = max_radius * 5
    geo.DSD = geo.DSO * 1.5

    magnification = geo.DSD / geo.DSO
    # Detector columns cover max(nY, nX) with 1.5× margin to prevent truncation
    geo.nDetector = np.array([nVoxel[0], max(nVoxel[1], nVoxel[2])])
    geo.dDetector = np.array([geo.dVoxel[0] * magnification,
                               geo.dVoxel[2] * magnification * 1.5])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])
    geo.accuracy = 0.5

    # --- Forward projection ---
    projections = tigre.Ax(volume, geo, angles)
    print(f"  Projections shape: {projections.shape}")

    # --- Save outputs ---
    case_out = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(case_out, exist_ok=True)

    np.save(os.path.join(case_out, "projections.npy"), projections)

    # Global contrast range across all angles; aspect preserves physical pixel dimensions
    vmin_proj, vmax_proj = projections.min(), projections.max()
    proj_aspect = geo.dDetector[1] / geo.dDetector[0]

    # Save every 10th projection angle as PNG
    for i in range(0, N_ANGLES, 10):
        fig, ax = plt.subplots()
        ax.imshow(projections[i], cmap="gray", aspect=proj_aspect,
                  vmin=vmin_proj, vmax=vmax_proj)
        ax.set_title(f"{case_name} — {np.degrees(angles[i]):.1f}°")
        ax.axis("off")
        fig.savefig(os.path.join(case_out, f"proj_{i:03d}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Save mid axial slice of volume
    # volume[mid_z] shape is (Y, X); physical pixel sizes are dVoxel[1] x dVoxel[2]
    vol_aspect = geo.dVoxel[2] / geo.dVoxel[1]
    mid_z = volume.shape[0] // 2
    fig, ax = plt.subplots()
    ax.imshow(volume[mid_z], cmap="gray", aspect=vol_aspect)
    ax.set_title(f"{case_name} — axial slice z={mid_z}")
    ax.axis("off")
    fig.savefig(os.path.join(case_out, "volume_axial_mid.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"  Saved to {case_out}/\n")

print("All cases done.")
