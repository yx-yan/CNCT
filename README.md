# CNCT — Dual-Domain Cascaded Network for Sparse-View Cone-Beam CT

End-to-end pipeline for sparse-view cone-beam CT reconstruction enhancement.
Forward-projects 3D volumetric CT data with a TIGRE-backed cone-beam model,
reconstructs with FDK, and then refines the coarse reconstructions with a
dual-domain cascaded neural network that operates on both the sinogram and
image domains through a differentiable backprojection bridge.

Built on [TIGRE](https://github.com/CERN/TIGRE) (GPU-accelerated CUDA) and
PyTorch.

---

## Project Structure

The repository is split into **two fully independent Python packages** so
the data-preparation stage (TIGRE-heavy, no PyTorch) and the deep-learning
stage (PyTorch + TIGRE) can be installed and shipped separately:

```
CNCT/
├── data_prep/                       # Package 1: cnct_dataprep
│   ├── pyproject.toml               # installable as `cnct-dataprep`
│   ├── configs/
│   │   ├── geometry.yaml            # shared cone-beam geometry
│   │   ├── projection.yaml          # forward-projection stage config
│   │   ├── fdk.yaml                 # FDK reconstruction stage config
│   │   └── evaluation.yaml          # PSNR/SSIM evaluation config
│   ├── scripts/                     # thin wrappers around the package CLI
│   └── src/cnct_dataprep/
│       ├── config/                  # typed dataclasses + YAML loaders
│       ├── geometry/                # HU↔mu conversion + per-case TIGRE geometry
│       ├── projection/              # tigre.Ax forward projection
│       ├── reconstruction/          # tigre.algorithms.fdk
│       ├── evaluation/              # PSNR / SSIM + comparison PNGs
│       ├── utils/                   # io, paths, logging, seeding, device
│       └── cli/                     # projection / fdk / evaluation entry points
│
├── src/cnct/                        # Package 2: cnct (deep-learning)
│   ├── config/                      # typed dataclasses + YAML loaders
│   ├── geometry/                    # HU↔mu + per-case geometry (duplicated from data_prep)
│   ├── data/                        # HPC-friendly lazy dataset + HDF5 split builder
│   │   ├── normalization.py         # pure fn helpers (mu↔[-1,1], sino→[0,1])
│   │   ├── dataset.py               # DualDomainDataset (fork-safe mmap I/O)
│   │   └── prepare.py               # HDF5 split-builder (0.80/0.10/0.10, seed=42)
│   ├── models/                      # verbatim-ported dual-domain cascade
│   │   ├── blocks.py                # ConvBlock / Encoder / Bottleneck / Decoder
│   │   ├── unet3d.py                # ResidualUNet3D (single-domain baseline)
│   │   └── dual_domain.py           # SinogramUNet3D + DBP + VolumeUNet3D + cascade
│   ├── training/                    # Trainer, losses, metrics, checkpointing
│   ├── inference/                   # Predictor
│   ├── utils/                       # io, paths, logging, seeding, device
│   └── cli/                         # train / predict / prepare-data entry points
│
├── configs/                         # cnct YAML configs
│   ├── geometry.yaml
│   ├── training.yaml
│   └── inference.yaml
│
├── slurm/                           # SLURM templates
│   ├── data_prep/sbatch_dataprep.sh         # stage 1+2+3 chained
│   └── dual_domain/
│       ├── sbatch_train.sh                  # cnct-train
│       └── sbatch_predict.sh                # cnct-predict
│
├── pyproject.toml                   # cnct package manifest
├── README.md                        # this file
└── CLAUDE.md                        # developer notes (architecture + gotchas)
```

### Why two packages?

`cnct_dataprep` has a narrow dependency footprint (numpy, nibabel, tigre,
scikit-image, matplotlib) and can run on any TIGRE-capable node without
PyTorch. `cnct` adds torch + h5py on top and reuses the dataset produced by
`cnct_dataprep`. The geometry module is **duplicated** between the two
packages rather than hoisted into a third shared package, so each package
can be installed, tested, and shipped independently.

### Data flow

```
NIfTI (HU)  ─┐
             ├─▶ cnct_dataprep.projection   ─▶ projections.npy  (sinograms)
             │
             ├─▶ cnct_dataprep.reconstruction ─▶ recon_fdk.npy  (coarse FDK)
             │
             └─▶ cnct.data.prepare            ─▶ h5_3dunet/{train,val,test}/*.h5
                                                  │
                                                  ▼
                                        cnct.training.Trainer
                                                  │
                                                  ▼
                                       best_checkpoint.pytorch
                                                  │
                                                  ▼
                                       cnct.inference.Predictor
                                                  │
                                                  ▼
                                        <case>_recon.npy   (enhanced)
                                                  │
                                                  ▼
                                   cnct_dataprep.evaluation  (PSNR/SSIM)
```

---

## Installation

Both packages use the `src/` layout and can be installed with
`pip install -e .`. You will typically use a single conda environment
(`fyp`) and install both packages into it, but they can live in separate
environments if needed.

### 1. Prerequisites

- Python ≥ 3.10
- CUDA 12.3 toolkit (the SLURM scripts load `CUDA/12.3.0`)
- [TIGRE](https://github.com/CERN/TIGRE) built against the node's GPU
  architecture — see the CUDA architecture note in `CLAUDE.md` before
  running on a Blackwell node
- A conda environment named `fyp` (used by the SLURM scripts)

### 2. Data-preparation package

```bash
cd CNCT/data_prep
pip install -e .
```

This registers three console scripts:

- `cnct-projection`  — simulate cone-beam projections from NIfTI volumes
- `cnct-fdk`         — reconstruct FDK volumes from projections
- `cnct-evaluation`  — PSNR/SSIM of FDK (or any .npy recon) vs GT

### 3. Deep-learning package

```bash
cd CNCT
pip install -e .
```

This registers three console scripts:

- `cnct-prepare-data` — build the HDF5 train/val/test splits
- `cnct-train`        — train the dual-domain cascade
- `cnct-predict`      — run trained inference

Both packages pick up `tigre` from the active conda environment; it is
**not** declared as a PyPI dependency because it must be built from source
for each node's GPU architecture.

---

## Quick Start

All paths and tunables live in YAML files under `data_prep/configs/` and
`configs/`. The typical workflow edits those files once and then submits
SLURM jobs.

### Step 1. Configure geometry and paths

Edit `data_prep/configs/geometry.yaml` (cone-beam geometry, `mu_water`) and
each stage config (`projection.yaml`, `fdk.yaml`, `evaluation.yaml`) to
point at your dataset. `configs/geometry.yaml` in the cnct package mirrors
the same schema and must agree with the data-prep geometry.

`configs/training.yaml` and `configs/inference.yaml` configure the cnct
pipeline. Key blocks:

```yaml
# configs/training.yaml  (excerpt)
geometry_config: geometry.yaml
data:
  data_dir: /projects/CTdata/AbdomenCT-1K-ImagePart1
  proj_dir: /projects/CTdata/projection60
  fdk_dir:  /projects/CTdata/fdk60
  h5_root:  /projects/CTdata/h5_3dunet
  spatial_scale: 0.5
  mu_min: -0.02
  mu_max:  0.08
model:
  sinogram_out_features: 4
  sinogram_f_maps: [8, 16, 32, 64]
  volume_f_maps:   [8, 16, 32, 64, 128]
  num_groups: 8
  use_checkpoint: true
loss:      {alpha: 1.0, beta: 0.2, gamma: 1.0, ssim_window: 7}
optimizer: {lr: 2.0e-4, weight_decay: 1.0e-5}
scheduler: {mode: max, factor: 0.5, patience: 10}
epochs: 200
seed: 42
device: auto
checkpoint_dir: /projects/CTdata/dual_domain_checkpoints
amp: true
```

### Step 2. Run data preparation (TIGRE)

```bash
cd CNCT
sbatch slurm/data_prep/sbatch_dataprep.sh
```

This chains projection → FDK → evaluation for every NIfTI case in
`data_dir`. Outputs:

- `projections.npy` in `PROJ_DIR/<case>/`
- `recon_fdk.npy`   in `FDK_DIR/<case>/`
- PNG comparisons + `evaluation_results.csv` in `EVAL_DIR/`

### Step 3. Build HDF5 splits

```bash
cnct-prepare-data                                 # full run with defaults
cnct-prepare-data --dry_run                       # preview split sizes
cnct-prepare-data --out_dir /tmp/h5 --seed 7      # override defaults
```

Produces `<h5_root>/{train,val,test}/<case>.h5`. The builder is
idempotent — existing files are skipped, so re-runs after interrupted
jobs are safe.

### Step 4. Train the dual-domain cascade

```bash
sbatch slurm/dual_domain/sbatch_train.sh                         # default config
sbatch slurm/dual_domain/sbatch_train.sh configs/my_training.yaml
```

Each epoch writes `last_checkpoint.pytorch`; whenever validation PSNR
improves, `best_checkpoint.pytorch` is also refreshed. Both files live
under `checkpoint_dir` from the training YAML. Resume by setting
`resume: /path/to/last_checkpoint.pytorch` in the YAML (or run
`cnct-train --config ... ` directly).

### Step 5. Run inference and evaluate

```bash
sbatch slurm/dual_domain/sbatch_predict.sh                         # default config
sbatch slurm/dual_domain/sbatch_predict.sh configs/my_inference.yaml
```

Outputs `<case>_recon.npy` files in the inference `output_dir`. Pipe those
through `cnct-evaluation` to get PSNR/SSIM vs the NIfTI ground truth.

### Running commands directly (outside SLURM)

Every SLURM script is a thin wrapper over a module-style invocation, so
you can run the same commands on a GPU login shell:

```bash
python -m cnct.cli.train   --config configs/training.yaml
python -m cnct.cli.predict --config configs/inference.yaml
cnct-prepare-data --dry_run
```

---

## Key Design Notes

- **Math preservation**: the dual-domain cascade (`cnct.models`) is a
  verbatim port of the legacy model. Bit-exact forward + backward parity
  with the pre-refactor implementation was confirmed on CPU (all
  CPU-testable blocks) and on GPU including the TIGRE differentiable
  backprojection, under `torch.backends.cudnn.deterministic = True`.
- **AMP + gradient clipping**: the Trainer runs forward and loss under
  `torch.amp.autocast("cuda")` with a `GradScaler`, unscales gradients
  before clipping at `||g||₂ ≤ 1.0` (default), then steps the optimiser.
  Clipping is essential because TIGRE adjoint gradients can spike.
- **HPC-friendly I/O**: `DualDomainDataset` opens no file handles in
  `__init__`. Every call to `__getitem__` memory-maps the sinogram and
  FDK `.npy` files and releases the handle on return, so the dataset is
  fork-safe with `num_workers > 0`.
- **Checkpoint completeness**: model, optimiser, scheduler, **and AMP
  scaler** states are all persisted so resumed runs continue at the same
  loss scale — skipping the scaler state causes NaNs on the first
  restored step.

See `CLAUDE.md` for deeper architectural notes and the CUDA-architecture
warning about Blackwell nodes.
