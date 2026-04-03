"""Full-volume prediction script for ResidualUNet3D.

Standalone inference — no dependency on pytorch-3dunet.  Loads test HDF5
files, resizes to the training target_shape, runs the model, denormalizes,
resizes back to the original shape, and saves _recon.npy files ready for
evaluation.py.

Usage:
    python 3dunet/run_fullvol_predict.py --config 3dunet/fullvol_train_config.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)

logger = logging.getLogger("fullvol_predict")

# ── Imports that depend on PYTHONPATH ─────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from unet3d_model import ResidualUNet3D  # noqa: E402


def load_config():
    parser = argparse.ArgumentParser(description="Full-volume 3D UNet prediction")
    parser.add_argument("--config", required=True, help="Path to YAML config (same as training)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (overrides config checkpoint_dir)")
    parser.add_argument("--test_dir", default=None,
                        help="Test HDF5 directory (overrides config test_dir)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for _recon.npy files")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    return config, args


def build_model(model_config, device):
    valid_keys = {"in_channels", "out_channels", "f_maps", "num_groups", "use_checkpoint"}
    kwargs = {k: v for k, v in model_config.items() if k in valid_keys}
    if "f_maps" in kwargs and isinstance(kwargs["f_maps"], list):
        kwargs["f_maps"] = tuple(kwargs["f_maps"])
    # Disable gradient checkpointing at inference (not needed, avoids overhead)
    kwargs["use_checkpoint"] = False
    model = ResidualUNet3D(**kwargs)
    return model.to(device)


def normalize(x, norm_min, norm_max):
    """Map from [norm_min, norm_max] to [-1, 1]."""
    x = (x - norm_min) / (norm_max - norm_min + 1e-10)
    return torch.clamp(2.0 * x - 1.0, -1.0, 1.0)


def denormalize(x, norm_min, norm_max):
    """Map from [-1, 1] back to [norm_min, norm_max]."""
    return (x + 1.0) / 2.0 * (norm_max - norm_min) + norm_min


def main():
    config, args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Resolve paths
    training_cfg = config["training"]
    data_cfg = config["data"]

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_path = args.checkpoint or str(checkpoint_dir / "best_checkpoint.pytorch")
    test_dir = Path(args.test_dir or data_cfg.get("test_dir", "/projects/CTdata/h5_3dunet/test"))
    output_dir = Path(args.output_dir or str(checkpoint_dir / "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    target_shape = tuple(training_cfg["target_shape"])
    norm_min = data_cfg.get("norm_min", -0.02)
    norm_max = data_cfg.get("norm_max", 0.08)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test dir:   {test_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Target shape: {target_shape}, norm range: [{norm_min}, {norm_max}]")

    # Build and load model
    model = build_model(config["model"], device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    epoch = state.get("num_epochs", "?")
    best_score = state.get("best_eval_score", "?")
    logger.info(f"Loaded checkpoint: epoch={epoch}, best_score={best_score}")

    # Discover test files
    test_files = sorted(test_dir.glob("*.h5"))
    if not test_files:
        logger.error(f"No .h5 files found in {test_dir}")
        return
    logger.info(f"Found {len(test_files)} test volume(s)")

    # Predict
    t0 = time.time()
    with torch.no_grad():
        for i, h5_path in enumerate(test_files):
            case_id = h5_path.stem
            out_path = output_dir / f"{case_id}_recon.npy"

            if out_path.exists():
                logger.info(f"  [{i+1}/{len(test_files)}] {case_id} — skip (exists)")
                continue

            # Load raw volume
            with h5py.File(h5_path, "r") as f:
                raw = f["raw"][:].astype(np.float32)  # (Z, Y, X), μ units
            orig_shape = raw.shape
            logger.info(f"  [{i+1}/{len(test_files)}] {case_id}  shape={orig_shape}")

            # To tensor: (1, 1, Z, Y, X)
            raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)

            # Resize to target_shape
            raw_resized = F.interpolate(raw_t, size=target_shape, mode="trilinear",
                                        align_corners=False)

            # Normalize to [-1, 1]
            raw_norm = normalize(raw_resized, norm_min, norm_max)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=True):
                output_norm = model(raw_norm)

            # Denormalize back to μ
            output_mu = denormalize(output_norm, norm_min, norm_max)

            # Resize back to original shape
            output_orig = F.interpolate(output_mu, size=orig_shape, mode="trilinear",
                                        align_corners=False)

            # Save as numpy
            pred_np = output_orig.squeeze().cpu().numpy().astype(np.float32)
            np.save(out_path, pred_np)
            logger.info(f"    Saved {out_path.name}  range=[{pred_np.min():.4f}, {pred_np.max():.4f}]")

    elapsed = time.time() - t0
    logger.info(f"Prediction complete: {len(test_files)} volumes in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
