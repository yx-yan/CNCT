"""Full-volume training script for ResidualUNet3D.

Standalone trainer with AMP, gradient checkpointing, and full-volume data
loading.  Produces checkpoints compatible with the existing prediction
pipeline (run_predict.py).

Key differences from the patch-based run_train.py:
  1. Loads entire volumes resized to a fixed target_shape (no SliceBuilder)
  2. Mixed precision (AMP) via torch.amp for memory efficiency
  3. Gradient checkpointing to reduce activation memory
  4. Correct loss computation: loss(output, target), not loss(artifacts, target)
  5. No dependency on the pytorch-3dunet library for training

Usage:
    python 3dunet/run_fullvol_train.py --config 3dunet/fullvol_train_config.yaml
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Logging ───────────────────────────────────────────────────────────────────

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)

logger = logging.getLogger("fullvol_train")

# ── Imports that depend on PYTHONPATH ─────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from unet3d_model import ResidualUNet3D  # noqa: E402
from fullvol_dataset import FullVolumeHDF5Dataset  # noqa: E402


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_psnr(prediction, target, max_val=2.0):
    """Compute PSNR between prediction and target.

    Both tensors are in [-1, 1] (normalized), so the dynamic range is 2.0.
    """
    mse = torch.mean((prediction - target) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 10.0 * torch.log10(max_val ** 2 / mse)


class RunningAverage:
    """Track a running mean."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config():
    parser = argparse.ArgumentParser(description="Full-volume 3D UNet training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config, args.config


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_everything(seed):
    if seed is None:
        return
    logger.info(f"Seeding all RNGs with {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning("CuDNN deterministic mode enabled — training may be slower")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(model_config, device):
    valid_keys = {"in_channels", "out_channels", "f_maps", "num_groups", "use_checkpoint"}
    kwargs = {k: v for k, v in model_config.items() if k in valid_keys}
    if "f_maps" in kwargs and isinstance(kwargs["f_maps"], list):
        kwargs["f_maps"] = tuple(kwargs["f_maps"])
    model = ResidualUNet3D(**kwargs)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ResidualUNet3D: {num_params:,} parameters, use_checkpoint={kwargs.get('use_checkpoint', False)}")
    return model.to(device)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(config):
    data_cfg = config["data"]
    training_cfg = config["training"]
    aug_cfg = config.get("augmentation", None)
    target_shape = tuple(training_cfg["target_shape"])
    subset = data_cfg.get("sample_subset_size", None)

    train_ds = FullVolumeHDF5Dataset(
        h5_dir=data_cfg["train_dir"],
        target_shape=target_shape,
        phase="train",
        norm_min=data_cfg.get("norm_min", -0.02),
        norm_max=data_cfg.get("norm_max", 0.08),
        augment_config=aug_cfg,
        subset_size=subset,
    )
    val_ds = FullVolumeHDF5Dataset(
        h5_dir=data_cfg["val_dir"],
        target_shape=target_shape,
        phase="val",
        norm_min=data_cfg.get("norm_min", -0.02),
        norm_max=data_cfg.get("norm_max", 0.08),
        augment_config=None,
        subset_size=subset,
    )

    num_workers = data_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scaler, epoch, iteration, best_score,
                    checkpoint_dir, is_best, training_config):
    """Save checkpoint in a format compatible with pytorch-3dunet's load_checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "num_epochs": epoch + 1,
        "num_iterations": iteration,
        "model_state_dict": model.state_dict(),
        "best_eval_score": best_score,
        "optimizer_state_dict": optimizer.state_dict(),
        # Extra metadata for full-volume inference (ignored by pytorch-3dunet)
        "training_mode": "full_volume",
        "target_shape": list(training_config["target_shape"]),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }

    last_path = checkpoint_dir / "last_checkpoint.pytorch"
    torch.save(state, last_path)
    logger.info(f"Saved checkpoint: epoch={epoch+1}, iter={iteration}, best={best_score:.4f}")

    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pytorch"
        shutil.copy(str(last_path), str(best_path))
        logger.info(f"New best checkpoint (score={best_score:.4f})")


def load_checkpoint_if_exists(config, model, optimizer, scaler):
    """Resume from checkpoint or load pre-trained weights."""
    training_cfg = config["training"]
    start_epoch = 0
    start_iter = 0
    best_score = float("-inf") if training_cfg.get("eval_score_higher_is_better", True) else float("inf")

    resume_path = training_cfg.get("resume")
    pre_trained_path = training_cfg.get("pre_trained")

    path = resume_path or pre_trained_path
    if path is None:
        return start_epoch, start_iter, best_score

    logger.info(f"Loading checkpoint from {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if resume_path:
        # Full resume: restore optimizer, scaler, counters
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state.get("num_epochs", 0)
        start_iter = state.get("num_iterations", 0)
        best_score = state.get("best_eval_score", best_score)
        logger.info(f"Resumed: epoch={start_epoch}, iter={start_iter}, best={best_score:.4f}")
    else:
        logger.info("Loaded pre-trained weights (optimizer NOT restored)")

    return start_epoch, start_iter, best_score


# ── Training & Validation ─────────────────────────────────────────────────────

def validate(model, val_loader, loss_fn, device, amp_enabled):
    """Run validation and return average PSNR."""
    model.eval()
    val_loss = RunningAverage()
    val_psnr = RunningAverage()

    with torch.no_grad():
        for raw, label in val_loader:
            raw = raw.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(raw)
                loss = loss_fn(output, label)

            val_loss.update(loss.item())
            val_psnr.update(compute_psnr(output, label).item())

    model.train()
    return val_psnr.avg, val_loss.avg


def main():
    config, config_path = load_config()

    # Seeding
    seed_everything(config.get("manual_seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model = build_model(config["model"], device)

    # Data
    train_loader, val_loader = build_dataloaders(config)

    # Loss
    loss_name = config["loss"]["name"]
    loss_fn = getattr(nn, loss_name)()
    logger.info(f"Loss: {loss_name}")

    # Optimizer
    opt_cfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get("learning_rate", 2e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )

    # AMP scaler
    training_cfg = config["training"]
    amp_enabled = training_cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    logger.info(f"AMP: {'enabled' if amp_enabled else 'disabled'}")

    # LR scheduler
    sched_cfg = config.get("lr_scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "max"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 10),
    )

    # Resume
    start_epoch, num_iterations, best_score = load_checkpoint_if_exists(
        config, model, optimizer, scaler
    )
    higher_is_better = training_cfg.get("eval_score_higher_is_better", True)

    # TensorBoard
    tb_dir = Path(training_cfg["checkpoint_dir"]) / "logs" / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs: {tb_dir}")

    # Training params
    max_epochs = training_cfg.get("max_num_epochs", 200)
    max_iters = training_cfg.get("max_num_iterations", 150000)
    validate_every = training_cfg.get("validate_after_iters", 500)
    log_every = training_cfg.get("log_after_iters", 100)

    logger.info(f"Training: max_epochs={max_epochs}, max_iters={max_iters}, "
                f"target_shape={training_cfg['target_shape']}")

    # ── Training loop ─────────────────────────────────────────────────────────

    model.train()
    for epoch in range(start_epoch, max_epochs):
        epoch_loss = RunningAverage()
        epoch_psnr = RunningAverage()

        for raw, label in train_loader:
            raw = raw.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(raw)
                loss = loss_fn(output, label)

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.update(loss.item())

            # Periodic logging
            if num_iterations % log_every == 0:
                psnr = compute_psnr(output.detach(), label).item()
                epoch_psnr.update(psnr)
                writer.add_scalar("train/loss", loss.item(), num_iterations)
                writer.add_scalar("train/psnr", psnr, num_iterations)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], num_iterations)
                logger.info(
                    f"Epoch {epoch+1} iter {num_iterations}: "
                    f"loss={loss.item():.6f}  PSNR={psnr:.2f} dB  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # Periodic validation
            if num_iterations > 0 and num_iterations % validate_every == 0:
                val_psnr, val_loss = validate(model, val_loader, loss_fn, device, amp_enabled)
                writer.add_scalar("val/loss", val_loss, num_iterations)
                writer.add_scalar("val/psnr", val_psnr, num_iterations)
                logger.info(
                    f"  Validation: loss={val_loss:.6f}  PSNR={val_psnr:.2f} dB"
                )

                # LR scheduling
                scheduler.step(val_psnr)

                # Checkpoint
                if higher_is_better:
                    is_best = val_psnr > best_score
                else:
                    is_best = val_psnr < best_score
                if is_best:
                    best_score = val_psnr

                save_checkpoint(
                    model, optimizer, scaler, epoch, num_iterations,
                    best_score, training_cfg["checkpoint_dir"], is_best, training_cfg
                )

                model.train()

            num_iterations += 1

            # Check stopping criteria
            if num_iterations >= max_iters:
                logger.info(f"Reached max iterations ({max_iters})")
                writer.close()
                return

            if optimizer.param_groups[0]["lr"] < 1e-6:
                logger.info("Learning rate below 1e-6 — stopping")
                writer.close()
                return

        logger.info(f"Epoch {epoch+1} complete: avg_loss={epoch_loss.avg:.6f}")

    logger.info("Training complete.")
    writer.close()


if __name__ == "__main__":
    main()
