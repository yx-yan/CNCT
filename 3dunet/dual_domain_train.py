#!/usr/bin/env python3
"""Training script for the Dual-Domain Cascaded Network.

This is a standalone training loop (not using pytorch-3dunet) because the
dual-domain model requires:
  - Two inputs per sample (sinogram + FDK volume) plus case-specific geometry
  - Full-volume processing (no patch-based SliceBuilder)
  - Variable-size inputs (batch_size=1, no collation across cases)

Features:
  - AMP (Automatic Mixed Precision) with GradScaler for memory-efficient training
  - Gradient checkpointing in both Branch A and Branch B
  - ReduceLROnPlateau scheduling on validation PSNR
  - Deterministic seeding for reproducibility
  - Checkpoint save/resume
  - Reuses the existing train/val/test split from h5_3dunet/

Usage:
  python 3dunet/dual_domain_train.py [--epochs 200] [--lr 2e-4] [--resume PATH]
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# Project root on PYTHONPATH (set by sbatch script)
from config import (
    DATA_DIR, PROJ_DIR, FDK_DIR, MU_WATER,
)
from geometry import build_geometry, hu_to_mu, load_nifti_as_tigre

# Model (imported from same directory)
sys.path.insert(0, os.path.dirname(__file__))
from dual_domain_model import DualDomainCascadeNet

logging.basicConfig(
    format="%(asctime)s  %(levelname)-5s  %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("dual_domain_train")


# =============================================================================
# Normalisation helpers
# =============================================================================

# Fixed range for mu values -- same as train_config.yaml Normalize transform.
# Ensures residual learning consistency between FDK input and GT label.
MU_MIN = -0.02
MU_MAX = 0.08


def normalize_mu(x: np.ndarray) -> np.ndarray:
    """Map mu values to [-1, 1] using fixed range."""
    return (2.0 * (x - MU_MIN) / (MU_MAX - MU_MIN) - 1.0).astype(np.float32)


def denormalize_mu(x: np.ndarray) -> np.ndarray:
    """Inverse of normalize_mu."""
    return ((x + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN).astype(np.float32)


def normalize_sinogram(sino: np.ndarray) -> np.ndarray:
    """Per-sample normalisation of sinogram to [0, 1].

    Sinogram values are line integrals of mu (unitless) and vary with
    volume size and path length, so per-sample normalisation is needed
    unlike the fixed-range mu normalisation for volumes.
    """
    smin, smax = sino.min(), sino.max()
    if smax - smin < 1e-8:
        return np.zeros_like(sino, dtype=np.float32)
    return ((sino - smin) / (smax - smin)).astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================


class DualDomainDataset(torch.utils.data.Dataset):
    """Loads matched (sinogram, FDK, GT) triplets with per-case geometry.

    Discovers cases from the existing HDF5 split directories (h5_3dunet/
    {train,val,test}/) to guarantee the same data split as the single-domain
    pipeline.  For each case, loads:
      - sinogram:   PROJ_DIR/<case>/projections.npy
      - fdk_volume: FDK_DIR/<case>/recon_fdk.npy
      - gt_volume:  DATA_DIR/<case>_0000.nii.gz  (HU -> mu conversion)
      - geometry:   rebuilt from the NIfTI header

    Volumes are optionally spatially downsampled in Y,X (matching the
    SpatialDownsample transform in the existing pipeline) to reduce memory
    for full-volume processing.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.
    h5_root : str
        Root of the HDF5 split directories (to discover case IDs).
    spatial_scale : float
        Downsample factor for Y,X dimensions (1.0 = no downsampling,
        0.5 = half resolution).  Z is left unchanged.
    """

    def __init__(
        self,
        split: str,
        h5_root: str = "/projects/CTdata/h5_3dunet",
        spatial_scale: float = 0.5,
    ):
        super().__init__()
        self.split = split
        self.spatial_scale = spatial_scale

        # Discover case IDs from the existing HDF5 split
        h5_dir = Path(h5_root) / split
        if not h5_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {h5_dir}")

        all_case_ids = sorted(p.stem for p in h5_dir.glob("*.h5"))

        # Filter to cases that have both projections.npy and recon_fdk.npy
        self.cases = []
        for case_id in all_case_ids:
            sino_path = Path(PROJ_DIR) / case_id / "projections.npy"
            fdk_path = Path(FDK_DIR) / case_id / "recon_fdk.npy"
            nii_path = Path(DATA_DIR) / f"{case_id}_0000.nii.gz"
            if sino_path.exists() and fdk_path.exists() and nii_path.exists():
                self.cases.append((case_id, str(sino_path), str(fdk_path), str(nii_path)))

        logger.info(
            f"[{split}] Found {len(self.cases)}/{len(all_case_ids)} cases "
            f"with sinogram + FDK + GT"
        )

    def __len__(self) -> int:
        return len(self.cases)

    def _spatial_downsample(self, vol: np.ndarray, mode: str = "trilinear") -> np.ndarray:
        """Downsample Y,X dimensions of a (Z, Y, X) volume."""
        if self.spatial_scale >= 1.0:
            return vol
        t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        align = mode != "nearest"
        t = torch.nn.functional.interpolate(
            t,
            scale_factor=(1.0, self.spatial_scale, self.spatial_scale),
            mode=mode,
            align_corners=align if mode != "nearest" else None,
        )
        return t.squeeze(0).squeeze(0).numpy()

    def __getitem__(self, idx: int) -> dict:
        case_id, sino_path, fdk_path, nii_path = self.cases[idx]

        # --- Load raw data ---
        # Sinogram is kept at ORIGINAL resolution -- it was generated by
        # projection.py with the original geometry, and the DBP layer needs
        # nDetector to match the sinogram's detector dimensions exactly.
        sinogram = np.load(sino_path).astype(np.float32)    # [N_angles, det_r, det_c]
        fdk_vol = np.load(fdk_path).astype(np.float32)      # [Z, Y, X] in mu units

        nii_img = nib.load(nii_path)
        gt_hu = nii_img.get_fdata(dtype=np.float32)          # [X, Y, Z]
        gt_vol = hu_to_mu(np.transpose(gt_hu, (2, 1, 0)))    # [Z, Y, X] in mu units

        # --- Spatial downsampling of VOLUMES only ---
        fdk_vol = self._spatial_downsample(fdk_vol, mode="trilinear")
        gt_vol = self._spatial_downsample(gt_vol, mode="nearest")

        # --- Build DBP-compatible geometry ---
        # Start from the ORIGINAL NIfTI header so nDetector and DSO/DSD
        # match the sinogram that projection.py generated.  Then override
        # nVoxel/dVoxel to target the (possibly downsampled) volume grid.
        # Physical volume size (sVoxel) is preserved, so the geometry
        # remains physically consistent.
        nVoxel_orig, voxel_sizes_orig = load_nifti_as_tigre(nii_img)
        geo = build_geometry(nVoxel_orig, voxel_sizes_orig)

        # Override voxel grid for the downsampled target volume
        nVoxel_ds = np.array(fdk_vol.shape, dtype=np.int64)
        sVoxel_orig = geo.sVoxel.copy()          # physical size stays the same
        geo.nVoxel = nVoxel_ds
        geo.dVoxel = sVoxel_orig / nVoxel_ds.astype(np.float64)
        # sVoxel = nVoxel * dVoxel = sVoxel_orig  (unchanged)

        # Ensure nDetector is integer (TIGRE requires this)
        geo.nDetector = geo.nDetector.astype(np.int64)

        # --- Derive angle count from the actual sinogram ---
        n_angles = sinogram.shape[0]
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)

        # --- Normalise ---
        fdk_norm = normalize_mu(fdk_vol)
        gt_norm = normalize_mu(gt_vol)
        sino_norm = normalize_sinogram(sinogram)

        return {
            "case_id": case_id,
            "sinogram": torch.from_numpy(sino_norm).unsqueeze(0),     # [1, Nang, Dr, Dc]
            "fdk_volume": torch.from_numpy(fdk_norm).unsqueeze(0),    # [1, Z, Y, X]
            "gt_volume": torch.from_numpy(gt_norm).unsqueeze(0),      # [1, Z, Y, X]
            "geo": geo,
            "angles": angles,
        }


# =============================================================================
# Metrics
# =============================================================================


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB between two tensors (normalised to [-1, 1], so data_range=2)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)  # data_range^2 = 2^2 = 4


# =============================================================================
# Training loop
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    dataset: DualDomainDataset,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.  Returns (avg_loss, avg_psnr)."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    # Shuffle case order each epoch
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for step, idx in enumerate(indices):
        sample = dataset[idx]
        sino = sample["sinogram"].unsqueeze(0).to(device)       # [1, 1, Nang, Dr, Dc]
        fdk = sample["fdk_volume"].unsqueeze(0).to(device)      # [1, 1, Z, Y, X]
        gt = sample["gt_volume"].unsqueeze(0).to(device)        # [1, 1, Z, Y, X]
        geo = sample["geo"]
        angles = sample["angles"]

        optimizer.zero_grad(set_to_none=True)

        # Forward under AMP autocast
        with torch.amp.autocast("cuda"):
            pred = model(fdk, sino, geo, angles)
            loss = criterion(pred, gt)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        psnr_val = compute_psnr(pred.detach().float(), gt.float())
        total_loss += loss_val
        total_psnr += psnr_val

        if (step + 1) % 5 == 0 or step == 0:
            logger.info(
                f"  Epoch {epoch} [{step+1}/{len(indices)}] "
                f"loss={loss_val:.6f}  PSNR={psnr_val:.2f} dB  "
                f"case={sample['case_id']}"
            )

        # Free GPU memory between cases (variable-size volumes)
        del sino, fdk, gt, pred, loss
        torch.cuda.empty_cache()

    n = len(indices)
    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(
    model: nn.Module,
    dataset: DualDomainDataset,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate over all cases.  Returns (avg_loss, avg_psnr)."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        sino = sample["sinogram"].unsqueeze(0).to(device)
        fdk = sample["fdk_volume"].unsqueeze(0).to(device)
        gt = sample["gt_volume"].unsqueeze(0).to(device)
        geo = sample["geo"]
        angles = sample["angles"]

        with torch.amp.autocast("cuda"):
            pred = model(fdk, sino, geo, angles)
            loss = criterion(pred, gt)

        total_loss += loss.item()
        total_psnr += compute_psnr(pred.float(), gt.float())

        del sino, fdk, gt, pred, loss
        torch.cuda.empty_cache()

    n = len(dataset)
    return total_loss / n, total_psnr / n


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Domain Cascade Network")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--sino_features", type=int, default=4,
                        help="Branch A output feature channels")
    parser.add_argument("--sino_f_maps", type=int, nargs="+", default=[8, 16, 32],
                        help="Branch A U-Net feature maps")
    parser.add_argument("--vol_f_maps", type=int, nargs="+", default=[8, 16, 32, 64, 128],
                        help="Branch B U-Net feature maps")

    # Data
    parser.add_argument("--spatial_scale", type=float, default=0.5,
                        help="Y,X downsample factor (0.5 = half resolution)")
    parser.add_argument("--h5_root", default="/projects/CTdata/h5_3dunet",
                        help="Root of existing HDF5 splits (for case discovery)")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", default="/projects/CTdata/dual_domain_checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # --- Deterministic seeding ---
    logger.info(f"Seeding all RNGs with {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Device ---
    device = torch.device("cuda")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")

    # --- Datasets ---
    train_ds = DualDomainDataset("train", h5_root=args.h5_root, spatial_scale=args.spatial_scale)
    val_ds = DualDomainDataset("val", h5_root=args.h5_root, spatial_scale=args.spatial_scale)

    if len(train_ds) == 0:
        logger.error("No training cases found. Run prepare_data.py and the FDK pipeline first.")
        sys.exit(1)

    # --- Model ---
    model = DualDomainCascadeNet(
        sinogram_out_features=args.sino_features,
        sinogram_f_maps=tuple(args.sino_f_maps),
        volume_f_maps=tuple(args.vol_f_maps),
        num_groups=8,
        use_checkpoint=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Branch A f_maps: {args.sino_f_maps}, out_features: {args.sino_features}")
    logger.info(f"Branch B f_maps: {args.vol_f_maps}")

    # --- Optimiser, loss, scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )
    scaler = torch.amp.GradScaler("cuda")

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_psnr = -float("inf")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, best PSNR={best_psnr:.2f} dB")

    # --- Training loop ---
    logger.info(f"Starting training: epochs={args.epochs}, lr={args.lr}, "
                f"spatial_scale={args.spatial_scale}")
    logger.info(f"Train cases: {len(train_ds)}, Val cases: {len(val_ds)}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_psnr = train_one_epoch(
            model, train_ds, optimizer, scaler, criterion, device, epoch,
        )

        val_loss, val_psnr = validate(model, val_ds, criterion, device)

        scheduler.step(val_psnr)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) -- "
            f"train_loss={train_loss:.6f} train_PSNR={train_psnr:.2f} dB | "
            f"val_loss={val_loss:.6f} val_PSNR={val_psnr:.2f} dB | "
            f"lr={current_lr:.2e}"
        )

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "best_psnr": best_psnr,
            "args": vars(args),
        }

        # Always save latest
        torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "last_checkpoint.pytorch"))

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt_data["best_psnr"] = best_psnr
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "best_checkpoint.pytorch"))
            logger.info(f"  >> New best val PSNR: {best_psnr:.2f} dB")

    logger.info(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
