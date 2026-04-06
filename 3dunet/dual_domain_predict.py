#!/usr/bin/env python3
"""Prediction script for the Dual-Domain Cascaded Network.

Standalone inference -- loads test cases (sinogram + FDK + geometry),
runs the trained DualDomainCascadeNet, denormalizes output back to mu,
and saves _recon.npy files ready for evaluation.py.

Usage:
    python 3dunet/dual_domain_predict.py [--checkpoint PATH] [--output_dir PATH]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# Project root on PYTHONPATH (set by sbatch script)
from config import (
    DATA_DIR, PROJ_DIR, FDK_DIR,
    DUAL_DOMAIN_CHECKPOINT_DIR, SINO_OUT_FEATURES, SINO_F_MAPS, VOLUME_F_MAPS,
)
from geometry import build_geometry, hu_to_mu, load_nifti_as_tigre

sys.path.insert(0, os.path.dirname(__file__))
from dual_domain_model import DualDomainCascadeNet

logging.basicConfig(
    format="%(asctime)s  %(levelname)-5s  %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("dual_domain_predict")


# =============================================================================
# Normalisation (must match dual_domain_train.py exactly)
# =============================================================================

MU_MIN = -0.02
MU_MAX = 0.08


def normalize_mu(x: np.ndarray) -> np.ndarray:
    """Map mu values to [-1, 1] using fixed range."""
    return (2.0 * (x - MU_MIN) / (MU_MAX - MU_MIN) - 1.0).astype(np.float32)


def denormalize_mu(x: np.ndarray) -> np.ndarray:
    """Map from [-1, 1] back to mu units."""
    return ((x + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN).astype(np.float32)


def normalize_sinogram(sino: np.ndarray) -> np.ndarray:
    """Per-sample normalisation of sinogram to [0, 1]."""
    smin, smax = sino.min(), sino.max()
    if smax - smin < 1e-8:
        return np.zeros_like(sino, dtype=np.float32)
    return ((sino - smin) / (smax - smin)).astype(np.float32)


def spatial_downsample(vol: np.ndarray, scale: float, mode: str = "trilinear") -> np.ndarray:
    """Downsample Y,X dimensions of a (Z, Y, X) volume."""
    if scale >= 1.0:
        return vol
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    align = mode != "nearest"
    t = torch.nn.functional.interpolate(
        t,
        scale_factor=(1.0, scale, scale),
        mode=mode,
        align_corners=align if mode != "nearest" else None,
    )
    return t.squeeze(0).squeeze(0).numpy()


def spatial_upsample(vol: np.ndarray, target_shape: tuple, mode: str = "trilinear") -> np.ndarray:
    """Upsample back to original (Z, Y, X) shape."""
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    align = mode != "nearest"
    t = torch.nn.functional.interpolate(
        t,
        size=target_shape,
        mode=mode,
        align_corners=align if mode != "nearest" else None,
    )
    return t.squeeze(0).squeeze(0).numpy()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Dual-Domain Cascade Network prediction")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: best_checkpoint.pytorch in checkpoint_dir)")
    parser.add_argument("--checkpoint_dir", default=DUAL_DOMAIN_CHECKPOINT_DIR)
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for _recon.npy files")
    parser.add_argument("--test_dir", default="/projects/CTdata/h5_3dunet/test",
                        help="Test HDF5 directory (for case discovery)")
    parser.add_argument("--spatial_scale", type=float, default=None,
                        help="Y,X downsample factor (default: from checkpoint args)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load checkpoint ---
    checkpoint_path = args.checkpoint or os.path.join(args.checkpoint_dir, "best_checkpoint.pytorch")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Recover training args from checkpoint
    train_args = ckpt.get("args", {})
    spatial_scale = args.spatial_scale or train_args.get("spatial_scale", 0.5)
    sino_features = train_args.get("sino_features", SINO_OUT_FEATURES)
    sino_f_maps = train_args.get("sino_f_maps", SINO_F_MAPS)
    vol_f_maps = train_args.get("vol_f_maps", VOLUME_F_MAPS)

    output_dir = Path(args.output_dir or os.path.join(args.checkpoint_dir, "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir: {output_dir}")
    logger.info(f"spatial_scale: {spatial_scale}")
    logger.info(f"sino_features: {sino_features}, sino_f_maps: {sino_f_maps}, vol_f_maps: {vol_f_maps}")
    logger.info(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, "
                f"best val PSNR: {ckpt.get('best_psnr', '?')}")

    # --- Build model ---
    model = DualDomainCascadeNet(
        sinogram_out_features=sino_features,
        sinogram_f_maps=tuple(sino_f_maps),
        volume_f_maps=tuple(vol_f_maps),
        num_groups=8,
        use_checkpoint=False,  # no checkpointing at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # --- Discover test cases ---
    test_dir = Path(args.test_dir)
    case_ids = sorted(p.stem for p in test_dir.glob("*.h5"))
    logger.info(f"Found {len(case_ids)} case(s) in {test_dir}")

    # Filter to cases with all required inputs
    cases = []
    for case_id in case_ids:
        sino_path = Path(PROJ_DIR) / case_id / "projections.npy"
        fdk_path = Path(FDK_DIR) / case_id / "recon_fdk.npy"
        nii_path = Path(DATA_DIR) / f"{case_id}_0000.nii.gz"
        if sino_path.exists() and fdk_path.exists() and nii_path.exists():
            cases.append((case_id, str(sino_path), str(fdk_path), str(nii_path)))
        else:
            logger.warning(f"  Skipping {case_id} — missing sinogram/FDK/NIfTI")

    logger.info(f"Predicting {len(cases)} case(s) with complete inputs")

    # --- Predict ---
    t0 = time.time()
    with torch.no_grad():
        for i, (case_id, sino_path, fdk_path, nii_path) in enumerate(cases):
            out_path = output_dir / f"{case_id}_recon.npy"
            if out_path.exists():
                logger.info(f"  [{i+1}/{len(cases)}] {case_id} — skip (exists)")
                continue

            logger.info(f"  [{i+1}/{len(cases)}] {case_id}")

            # Load raw data
            sinogram = np.load(sino_path).astype(np.float32)   # [N_angles, det_r, det_c]
            fdk_vol = np.load(fdk_path).astype(np.float32)     # [Z, Y, X] in mu
            orig_shape = fdk_vol.shape

            nii_img = nib.load(nii_path)

            # Spatial downsampling of volumes only
            fdk_ds = spatial_downsample(fdk_vol, spatial_scale)
            logger.info(f"    orig={orig_shape} -> downsampled={fdk_ds.shape}, "
                        f"sino={sinogram.shape}")

            # Build DBP geometry (from original NIfTI, then override for downsampled grid)
            nVoxel_orig, voxel_sizes_orig = load_nifti_as_tigre(nii_img)
            geo = build_geometry(nVoxel_orig, voxel_sizes_orig)

            nVoxel_ds = np.array(fdk_ds.shape, dtype=np.int64)
            sVoxel_orig = geo.sVoxel.copy()
            geo.nVoxel = nVoxel_ds
            geo.dVoxel = sVoxel_orig / nVoxel_ds.astype(np.float64)
            geo.nDetector = geo.nDetector.astype(np.int64)

            n_angles = sinogram.shape[0]
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)

            # Normalise
            fdk_norm = normalize_mu(fdk_ds)
            sino_norm = normalize_sinogram(sinogram)

            # To tensors
            sino_t = torch.from_numpy(sino_norm).unsqueeze(0).unsqueeze(0).to(device)
            fdk_t = torch.from_numpy(fdk_norm).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass
            with torch.amp.autocast("cuda"):
                pred_norm = model(fdk_t, sino_t, geo, angles)

            # Denormalize to mu
            pred_ds = denormalize_mu(pred_norm.squeeze().cpu().numpy())

            # Upsample back to original shape
            pred_mu = spatial_upsample(pred_ds, orig_shape)

            np.save(out_path, pred_mu)
            logger.info(f"    Saved {out_path.name}  "
                        f"range=[{pred_mu.min():.4f}, {pred_mu.max():.4f}]")

            # Free GPU memory between cases
            del sino_t, fdk_t, pred_norm
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info(f"Prediction complete: {len(cases)} case(s) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
