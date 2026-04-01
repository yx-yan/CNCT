# CNCT — Cone-Beam CT Simulation and Deep Learning Enhancement

Simulates cone-beam CT (CBCT) X-ray projections from 3D volumetric CT data,
reconstructs with the FDK algorithm, and trains a 3D UNet to enhance the
coarse reconstructions back towards diagnostic quality.

Built on [TIGRE](https://github.com/CERN/TIGRE) (GPU-accelerated CUDA) and
[pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).

## Project structure

```
CNCT/
  config.py                  # All tunable parameters (paths, geometry, output flags)
  geometry.py                # Shared CBCT geometry builder and HU/mu conversions
  projection.py              # Stage 1: NIfTI -> simulated cone-beam projections
  fdk.py                     # Stage 2: Projections -> FDK reconstruction
  evaluation.py              # Stage 3: PSNR/SSIM evaluation against ground truth
  prepare_data.py            # Convert FDK + GT pairs to HDF5 for 3D UNet training
  run_train.py               # Training entry-point with clean logging
  postprocess_predictions.py # Convert 3D UNet predictions back to numpy
  train_config.yaml          # pytorch-3dunet training configuration
  test_config.yaml           # pytorch-3dunet inference configuration
  sbatch.sh                  # SLURM job: projection -> FDK -> evaluation
  sbatch_train.sh            # SLURM job: data prep + 3D UNet training
  sbatch_predict.sh          # SLURM job: 3D UNet inference + postprocessing
```

## Requirements

- **Hardware**: NVIDIA GPU (tested on A100/V100; see known issues below)
- **Software**: CUDA 12.3, conda environment `fyp` with:
  - [TIGRE](https://github.com/CERN/TIGRE) (compiled for target GPU architecture)
  - PyTorch, nibabel, scikit-image, h5py, matplotlib
  - [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) (at `~/pytorch-3dunet`)

## Dataset

[AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) — 1K+ abdominal CT
volumes as NIfTI files (`<CaseID>_0000.nii.gz`).  Located at
`/projects/CTdata/AbdomenCT-1K-ImagePart{1,2,3}` on the cluster.

## Quick start

All jobs are submitted from the `CNCT/` directory on the HPC cluster.

### Pipeline 1: CBCT simulation and evaluation

```bash
# Runs projection.py -> fdk.py -> evaluation.py
sbatch sbatch.sh

# Monitor
squeue -u $USER
cat logs/output_<job_id>.log
```

**Data flow:**

1. **projection.py** — Load NIfTI, convert HU to linear attenuation, build
   per-case cone-beam geometry, run GPU ray-tracing, save `projections.npy`
2. **fdk.py** — Load projections, rebuild identical geometry from NIfTI
   header, run FDK filtered back-projection, save `recon_fdk.npy`
3. **evaluation.py** — Compare FDK reconstruction against ground truth,
   compute PSNR and SSIM, write `evaluation_results.csv`

### Pipeline 2: 3D UNet enhancement

```bash
# Step 1: Build HDF5 dataset (run once; existing files are skipped)
python prepare_data.py          # add --dry_run to preview the split

# Step 2: Train
sbatch sbatch_train.sh          # logs -> logs/train_<job_id>.log

# Step 3: Predict and postprocess
sbatch sbatch_predict.sh        # logs -> logs/predict_<job_id>.log
```

## Configuration

All parameters are in [`config.py`](config.py). Key settings:

| Parameter | Effect |
|---|---|
| `N_ANGLES` | Number of projection angles (more = fewer streak artifacts) |
| `ACCURACY` | Ray-integration step size in voxels (lower = finer, slower) |
| `FDK_FILTER` | FDK filter: `ram_lak`, `shepp_logan`, `cosine`, `hamming`, `hann` |
| `DSO_SCALE` | Source distance = max_radius * DSO_SCALE |
| `DSD_SCALE` | Detector distance = DSO * DSD_SCALE (magnification) |
| `DETECTOR_COL_MARGIN` | Extra detector width to prevent truncation artifacts |
| `SAVE_PNG` | Toggle PNG output in all three pipeline scripts |
| `SAVE_NII` | Toggle NIfTI output in fdk.py |

## Architecture notes

### Geometry consistency

`projection.py` and `fdk.py` both call `geometry.build_geometry()` from the
shared [`geometry.py`](geometry.py) module.  This guarantees identical geometry
for forward projection and reconstruction.  The geometry is rebuilt per case
because volumes vary in size and voxel spacing.

### Axis conventions

- **TIGRE** volumes are always `(Z, Y, X)`
- **NIfTI** `get_fdata()` returns `(X, Y, Z)` — transposed with
  `np.transpose(volume, (2, 1, 0))`
- Voxel sizes from NIfTI are `(X, Y, Z)` — reordered to `(Z, Y, X)` for TIGRE

### Unit conventions

- Volumes in the pipeline are in **linear attenuation (mu, mm-1)**
- HU-to-mu conversion: `mu = clip((HU + 1000) / 1000 * MU_WATER, 0)`
- `geometry.hu_to_mu()` and `geometry.mu_to_hu()` handle conversions

## Known issues

**CUDA architecture mismatch**: `--constraint=gpu_48g` may allocate a Blackwell
GPU (RTX PRO 6000) incompatible with CUDA 12.3.  TIGRE silently fails with no
`projections.npy` output.  Use `--constraint=gpu` for TIGRE jobs or recompile
TIGRE for the target architecture.
