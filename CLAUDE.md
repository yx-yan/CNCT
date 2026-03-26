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
Certain GPU constraints (e.g. `--constraint=gpu_48g`) may allocate a Blackwell GPU (RTX PRO 6000), which is incompatible with CUDA 12.3 and causes `Ax:Siddon_projection no kernel image is available for execution on the device`. If this happens, TIGRE silently fails and no `projections.npy` is written, causing `fdk.py` to skip all cases. Fix by recompiling TIGRE for the target GPU architecture or requesting a different node (current `sbatch.sh` uses `--constraint=gpu` to avoid this).

## Configuration

All tunable parameters live in `config.py` and are imported by `projection.py`, `fdk.py`, and `evaluation.py`. The log header printed by `sbatch.sh` echoes all key config values at job start, making it easy to compare runs.

Key parameters and their effects on quality:

| Parameter | Default | Effect |
|---|---|---|
| `N_ANGLES` | 360 | More angles → fewer streak artifacts, longer runtime |
| `ACCURACY` | 0.5 | Ray-integration step size in voxels — **lower = finer steps = more accurate** (comment direction is: lower is more accurate) |
| `FDK_FILTER` | `"shepp_logan"` | FDK backprojection filter: `"ram_lak"` \| `"shepp_logan"` \| `"cosine"` \| `"hamming"` \| `"hann"` |
| `DSO_SCALE` | 5 | Source distance = `max_radius × DSO_SCALE` |
| `DSD_SCALE` | 1.5 | Detector distance = `DSO × DSD_SCALE` (magnification) |
| `DETECTOR_COL_MARGIN` | 1.5 | Extra detector width factor to prevent truncation artifacts |
| `SAVE_PNG` | `True` | Save projection and reconstruction slice PNGs |
| `SAVE_NII` | `True` | Save `recon_fdk.nii.gz` (large; disable to save disk space) |

`recon_fdk.npy` is always written regardless of `SAVE_PNG`/`SAVE_NII` since `evaluation.py` depends on it.

## Three-script pipeline

`projection.py` and `fdk.py` must use **identical geometry**. `fdk.py` reconstructs from the `.npy` files written by `projection.py` by rebuilding the same geometry from the NIfTI header. If geometry parameters diverge, reconstruction will be mis-registered.

### Data flow
1. `projection.py`: Load NIfTI → transpose `(X,Y,Z)` → `(Z,Y,X)` → convert HU to linear attenuation → build geometry → `tigre.Ax()` → save `projections.npy` + (optionally) PNGs
2. `fdk.py`: Load `projections.npy` → rebuild same geometry from NIfTI header → `tigre.algorithms.fdk(..., filter=FDK_FILTER)` → save `recon_fdk.npy` + (optionally) `recon_fdk.nii.gz` + slice PNGs
3. `evaluation.py`: Load NIfTI + `recon_fdk.npy` → compute PSNR & mean per-slice SSIM → save `eval_*.png` comparison images + `output/evaluation_results.csv`

### Geometry code duplication
`fdk.py` encapsulates geometry construction in a `build_geometry(nVoxel, voxel_sizes)` function. `projection.py` inlines the equivalent logic. If you change geometry parameters, **both files must be updated identically**. Consider extracting `build_geometry` to `config.py` or a shared module if the pipeline grows.

### Geometry conventions
- TIGRE volumes are always **(Z, Y, X)** — NIfTI `get_fdata()` returns `(X, Y, Z)` and must be transposed with `np.transpose(volume, (2, 1, 0))`.
- NIfTI `get_zooms()` returns voxel sizes in `(X, Y, Z)` order; `geo.dVoxel` must be reordered to `[Z, Y, X]`.
- **DSO/DSD are dynamic**, not hardcoded: `max_radius = sqrt((sVoxel[1]/2)² + (sVoxel[2]/2)²)`, then `DSO = max_radius × DSO_SCALE`, `DSD = DSO × DSD_SCALE`. This ensures the source is outside every volume and eliminates the circular FOV boundary artifact.
- Detector columns use `max(nVoxel[1], nVoxel[2])` with `dDetector[1] *= DETECTOR_COL_MARGIN` to prevent truncation artifacts.
- Geometry must be rebuilt per case — `nVoxel`, `dVoxel`, and therefore DSO/DSD all differ across cases.

### Volume preprocessing
- HU values are converted to linear attenuation coefficients before projection: `mu = clip((HU + 1000) / 1000 × MU_WATER, 0)`.
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
    proj_000.png            # every PROJ_SAVE_EVERY-th angle — only if SAVE_PNG=True
    volume_axial_mid.png    # mid-slice sanity check (mu values) — only if SAVE_PNG=True
    recon_fdk.npy           # FDK reconstruction, shape (Z, Y, X) — always written
    recon_fdk.nii.gz        # same, converted back to HU and NIfTI (X, Y, Z) — only if SAVE_NII=True
    recon_axial.png         # only if SAVE_PNG=True
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

Dataset: AbdomenCT-1K, available in three parts at `/projects/_hdd/CTdata/AbdomenCT-1K-ImagePart1` (and Part2, Part3). `config.py` points to Part1 by default. Volumes are `.nii.gz` files named `<CaseID>_0000.nii.gz`. Volumes vary significantly in Z depth (61–834 slices) and voxel spacing, which is why geometry is rebuilt per case.
