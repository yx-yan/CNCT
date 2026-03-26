# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical imaging X-ray projection simulation project. Loads 3D volumetric CT data from NIfTI files (.nii.gz), generates simulated cone-beam CT projections (`projection.py`), then reconstructs with FDK (`fdk.py`), using the TIGRE library (GPU-accelerated CUDA).

## Running

This project runs on an HPC cluster via SLURM. Submit jobs from the `CNCT/` directory:

```bash
# Submit to GPU cluster
sbatch sbatch.sh

# Monitor job
squeue -u $USER

# Check output
cat logs/output_<job_id>.log
cat logs/output_<job_id>.err
```

The SLURM job loads `CUDA/12.3.0` and activates the `tigre` conda environment, then runs `projection.py` → `fdk.py` → `evaluation.py`. Reference TIGRE demos at `~/TIGRE/Python/demos/`.

### Known issue: CUDA architecture mismatch
The `--constraint=gpu_48g` constraint may allocate a Blackwell GPU (RTX PRO 6000), which is incompatible with CUDA 12.3 and causes `Ax:Siddon_projection no kernel image is available for execution on the device`. If this happens, TIGRE silently fails and no `projections.npy` is written, causing `fdk.py` to skip all cases. Fix by recompiling TIGRE for the target GPU architecture or requesting a different node.

## Configuration

All tunable parameters live in `config.py` and are imported by both `projection.py` and `fdk.py`. Edit `config.py` to change scan angles, geometry scaling, detector margins, accuracy, attenuation constants, or output DPI. **`evaluation.py` has its own `MU_WATER` constant that must be kept in sync with `config.py`.**

## Three-script pipeline

`projection.py` and `fdk.py` must use **identical geometry**. `fdk.py` reconstructs from the `.npy` files written by `projection.py` by rebuilding the same geometry from the NIfTI header. If geometry parameters diverge, reconstruction will be mis-registered.

### Data flow
1. `projection.py`: Load NIfTI → transpose `(X,Y,Z)` → `(Z,Y,X)` → convert HU to linear attenuation → build geometry → `tigre.Ax()` → save `projections.npy` + PNGs
2. `fdk.py`: Load `projections.npy` → rebuild same geometry from NIfTI header → `tigre.algorithms.fdk()` → save `recon_fdk.npy` + slice PNGs
3. `evaluation.py`: Load NIfTI + `recon_fdk.npy` → compute PSNR & mean per-slice SSIM → save `eval_*.png` comparison images + `output/evaluation_results.csv`

### Geometry conventions
- TIGRE volumes are always **(Z, Y, X)** — NIfTI `get_fdata()` returns `(X, Y, Z)` and must be transposed with `np.transpose(volume, (2, 1, 0))`.
- NIfTI `get_zooms()` returns voxel sizes in `(X, Y, Z)` order; `geo.dVoxel` must be reordered to `[Z, Y, X]`.
- **DSO/DSD are dynamic**, not hardcoded: `max_radius = sqrt((sVoxel[1]/2)² + (sVoxel[2]/2)²)`, then `DSO = max_radius × DSO_SCALE`, `DSD = DSO × DSD_SCALE` (defaults 5 and 1.5). This ensures the source is outside every volume and eliminates the circular FOV boundary artifact.
- Detector columns use `max(nVoxel[1], nVoxel[2])` with `dDetector[1] *= DETECTOR_COL_MARGIN` to prevent truncation artifacts.
- Geometry must be rebuilt per case — `nVoxel`, `dVoxel`, and therefore DSO/DSD all differ across cases.

### Volume preprocessing
- HU values are converted to linear attenuation coefficients before projection: `mu = clip((HU + 1000) / 1000 × 0.02, 0)`.
- Volume must be `np.ascontiguousarray` after conversion for GPU efficiency.

### Visualisation conventions
- Always `matplotlib.use("Agg")` before importing pyplot — the cluster is headless.
- `imshow` aspect ratio must reflect physical pixel dimensions: `aspect = col_physical_size / row_physical_size`. For projections: `dDetector[1] / dDetector[0]`; for volume axial slices: `dVoxel[2] / dVoxel[1]`.
- Projection images use a **global** `vmin/vmax` across all angles so contrast is consistent.
- Reconstruction slice images use per-volume `np.percentile(recon, [1, 99])` to avoid outlier-dominated contrast.

## Output structure

```
output/
  Case_00001/
    projections.npy         # shape (N_ANGLES, nDetZ, nDetX), float32, linear attenuation units
    proj_000.png            # every PROJ_SAVE_EVERY-th angle, consistent contrast
    volume_axial_mid.png    # mid-slice sanity check (mu values)
    recon_fdk.npy           # FDK reconstruction, shape (Z, Y, X)
    recon_axial.png
    recon_coronal.png
    recon_sagittal.png
    eval_axial.png          # GT | FDK | diff (written by evaluation.py)
    eval_coronal.png
    eval_sagittal.png
  Case_00002/
    ...
  evaluation_results.csv    # per-case PSNR (dB) and mean SSIM
```

## Data

Five CT volumes in `data/` (Case_00001–Case_00005), `.nii.gz` format (~590 MB total). Volumes vary significantly in Z depth (61–834 slices) and voxel spacing, which is why geometry is rebuilt per case.
