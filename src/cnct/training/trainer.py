"""Dual-domain cascade Trainer.

Encapsulates the training loop that was previously a flat ``main()`` in
``3dunet/dual_domain_train.py``. Responsibilities:

    * Per-case loading, AMP forward + backward, gradient clipping, and
      optimiser step with a correct :class:`torch.amp.GradScaler` dance.
    * ``ReduceLROnPlateau`` scheduling on validation PSNR.
    * Deterministic case shuffling per epoch (via a seeded
      :class:`torch.Generator`).
    * Robust checkpointing: the latest checkpoint is written every epoch,
      and a separate "best" checkpoint tracks the highest validation PSNR.
    * Resume support — optimiser, scheduler, and AMP scaler states are
      all restored so an interrupted run continues with the correct loss
      scale and learning-rate history.

Stability notes:

    * **AMP**: forward+loss run under ``torch.amp.autocast("cuda")``. The
      DBP layer inside the cascade forces float32 internally for TIGRE.
    * **Gradient clipping**: essential because TIGRE adjoint gradients
      can spike when the sinogram domain has high dynamic range. We
      ``scaler.unscale_(optimizer)`` to get gradients back in the true
      scale, then ``clip_grad_norm_`` with a configurable threshold,
      then ``scaler.step(optimizer)``. Skipping ``unscale_`` would clip
      scaled gradients and produce inconsistent updates.
    * **Memory**: per-case tensors are aggressively deleted and
      ``torch.cuda.empty_cache()`` is invoked after each step to keep
      TIGRE and PyTorch from fighting over the allocator.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config.schema import TrainingCfg
from ..models import DualDomainCascadeNet
from ..utils.device import resolve_device
from ..utils.seeding import set_seed
from .checkpoint import (
    CheckpointPaths,
    TrainState,
    load_checkpoint,
    save_checkpoint,
)
from .losses import HybridLoss, build_loss
from .metrics import compute_psnr

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "DEFAULT_GRAD_CLIP_NORM"]

DEFAULT_GRAD_CLIP_NORM = 1.0


class Trainer:
    """End-to-end training driver for :class:`DualDomainCascadeNet`.

    Args:
        cfg: Validated top-level training configuration.
        train_dataset: Streaming dataset for the ``train`` split.
        val_dataset: Streaming dataset for the ``val`` split.
        grad_clip_norm: Maximum L2 norm for gradient clipping. ``None``
            disables clipping, but this is strongly discouraged for the
            dual-domain cascade because TIGRE adjoint gradients can spike.
    """

    def __init__(
        self,
        cfg: TrainingCfg,
        train_dataset: Dataset,
        val_dataset: Dataset,
        grad_clip_norm: Optional[float] = DEFAULT_GRAD_CLIP_NORM,
    ) -> None:
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.grad_clip_norm = grad_clip_norm

        set_seed(cfg.seed, deterministic=True)
        self.device = resolve_device(cfg.device)
        logger.info("Trainer device: %s", self.device)

        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        self.criterion: HybridLoss = build_loss(cfg.loss).to(self.device)

        self.paths = CheckpointPaths.from_dir(cfg.checkpoint_dir)
        self.state = TrainState()

        # Dedicated generator so shuffling is reproducible independently
        # of any global RNG state consumed by the model or dataset.
        self._shuffle_gen = torch.Generator()
        self._shuffle_gen.manual_seed(cfg.seed)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model: %s params | AMP=%s | grad_clip=%s",
            f"{n_params:,}",
            cfg.amp,
            self.grad_clip_norm,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> DualDomainCascadeNet:
        """Instantiate the dual-domain cascade from :attr:`cfg.model`."""
        m = self.cfg.model
        return DualDomainCascadeNet(
            sinogram_out_features=m.sinogram_out_features,
            sinogram_f_maps=m.sinogram_f_maps,
            volume_f_maps=m.volume_f_maps,
            num_groups=m.num_groups,
            use_checkpoint=m.use_checkpoint,
            use_fdk=m.use_fdk,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW from :attr:`cfg.optimizer`."""
        o = self.cfg.optimizer
        return torch.optim.AdamW(
            self.model.parameters(), lr=o.lr, weight_decay=o.weight_decay
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Configure ``ReduceLROnPlateau`` from :attr:`cfg.scheduler`."""
        s = self.cfg.scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=s.mode,
            factor=s.factor,
            patience=s.patience,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_resume(self) -> None:
        """Restore state from :attr:`cfg.resume` if it is set."""
        if self.cfg.resume is None:
            return
        self.state = load_checkpoint(
            Path(self.cfg.resume),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )

    def fit(self) -> float:
        """Run the full training loop from ``state.start_epoch`` to ``cfg.epochs``.

        Returns:
            The best validation PSNR observed during the run.
        """
        self.maybe_resume()
        patience = self.cfg.early_stopping_patience
        logger.info(
            "Starting training: epochs=%d, lr=%.2e, train=%d, val=%d, "
            "early_stopping_patience=%d",
            self.cfg.epochs,
            self.cfg.optimizer.lr,
            len(self.train_dataset),
            len(self.val_dataset),
            patience,
        )

        for epoch in range(self.state.start_epoch, self.cfg.epochs + 1):
            t0 = time.time()

            train_loss, train_psnr = self._train_one_epoch(epoch)
            val_loss, val_psnr = self._validate()
            self.scheduler.step(val_psnr)

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d (%.0fs) — train_loss=%.6f train_PSNR=%.2f dB | "
                "val_loss=%.6f val_PSNR=%.2f dB | lr=%.2e",
                epoch,
                self.cfg.epochs,
                elapsed,
                train_loss,
                train_psnr,
                val_loss,
                val_psnr,
                lr,
            )

            metrics = {
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "lr": lr,
            }

            self._save_epoch_checkpoints(epoch, metrics, val_psnr)

            if self.state.epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping: val PSNR has not improved for %d "
                    "consecutive epoch(s) (best=%.2f dB). Stopping at epoch %d.",
                    self.state.epochs_without_improvement,
                    self.state.best_psnr,
                    epoch,
                )
                break
            elif self.state.epochs_without_improvement > 0:
                logger.info(
                    "  No improvement for %d/%d epoch(s)",
                    self.state.epochs_without_improvement,
                    patience,
                )

        logger.info(
            "Training complete. Best val PSNR: %.2f dB", self.state.best_psnr
        )
        return self.state.best_psnr

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one training epoch over every case in the train dataset.

        Args:
            epoch: 1-indexed epoch number for logging.

        Returns:
            Tuple ``(avg_loss, avg_psnr)`` over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0

        indices = self._shuffled_indices(len(self.train_dataset))
        n_steps = len(indices)

        for step, idx in enumerate(indices, start=1):
            sample = self.train_dataset[idx]
            loss_val, psnr_val = self._train_step(sample)
            total_loss += loss_val
            total_psnr += psnr_val

            if step == 1 or step % self.cfg.log_interval == 0:
                logger.info(
                    "  Epoch %d [%d/%d] loss=%.6f  PSNR=%.2f dB  case=%s",
                    epoch,
                    step,
                    n_steps,
                    loss_val,
                    psnr_val,
                    sample["case_id"],
                )

        return total_loss / n_steps, total_psnr / n_steps

    def _train_step(self, sample: Dict[str, object]) -> tuple[float, float]:
        """Forward + backward + optimiser step for a single case.

        Args:
            sample: One dict yielded by :class:`DualDomainDataset`.

        Returns:
            Tuple ``(loss, psnr)`` as Python floats.
        """
        sino = sample["sinogram"].unsqueeze(0).to(self.device, non_blocking=True)
        fdk = sample["fdk_volume"].unsqueeze(0).to(self.device, non_blocking=True)
        gt = sample["gt_volume"].unsqueeze(0).to(self.device, non_blocking=True)
        geo = sample["geo"]
        angles = sample["angles"]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.cfg.amp):
            pred = self.model(fdk, sino, geo, angles)
            loss = self.criterion(pred, gt)

        # Backward with optional gradient scaling
        self.scaler.scale(loss).backward()

        # Unscale before clipping so the clip threshold refers to the
        # true (un-scaled) gradient norm. Clipping is essential because
        # the TIGRE adjoint pair occasionally produces large spikes
        # in sinogram-domain gradients for edge cases (thin slabs,
        # tight DSO/DSD ratios).
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_val = float(loss.detach().item())
        psnr_val = compute_psnr(
            pred.detach().float(),
            gt.float(),
            mu_min=self.cfg.data.mu_min,
            mu_max=self.cfg.data.mu_max,
        )

        del sino, fdk, gt, pred, loss
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return loss_val, psnr_val

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Evaluate the model on the validation set.

        Returns:
            Tuple ``(avg_loss, avg_psnr)`` averaged over every val case.
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        n = len(self.val_dataset)
        if n == 0:
            return float("nan"), float("nan")

        for idx in range(n):
            sample = self.val_dataset[idx]
            sino = sample["sinogram"].unsqueeze(0).to(self.device, non_blocking=True)
            fdk = sample["fdk_volume"].unsqueeze(0).to(self.device, non_blocking=True)
            gt = sample["gt_volume"].unsqueeze(0).to(self.device, non_blocking=True)
            geo = sample["geo"]
            angles = sample["angles"]

            with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                pred = self.model(fdk, sino, geo, angles)
                loss = self.criterion(pred, gt)

            total_loss += float(loss.item())
            total_psnr += compute_psnr(
                pred.float(),
                gt.float(),
                mu_min=self.cfg.data.mu_min,
                mu_max=self.cfg.data.mu_max,
            )

            del sino, fdk, gt, pred, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return total_loss / n, total_psnr / n

    # ------------------------------------------------------------------
    # Checkpointing + utilities
    # ------------------------------------------------------------------

    def _save_epoch_checkpoints(
        self, epoch: int, metrics: Dict[str, float], val_psnr: float
    ) -> None:
        """Save ``last`` and (conditionally) ``best`` checkpoints.

        Args:
            epoch: Epoch number that just completed.
            metrics: Latest epoch metrics dict.
            val_psnr: Current validation PSNR for best-tracking.
        """
        if val_psnr > self.state.best_psnr:
            self.state.best_psnr = val_psnr
            self.state.epochs_without_improvement = 0
            improved = True
        else:
            self.state.epochs_without_improvement += 1
            improved = False

        extra = {
            "epochs_without_improvement": self.state.epochs_without_improvement
        }

        save_checkpoint(
            self.paths.last,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            best_psnr=self.state.best_psnr,
            metrics=metrics,
            extra=dict(extra),
        )

        if improved:
            save_checkpoint(
                self.paths.best,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                best_psnr=self.state.best_psnr,
                metrics=metrics,
                extra=dict(extra),
            )
            logger.info("  >> New best val PSNR: %.2f dB", self.state.best_psnr)

    def _shuffled_indices(self, n: int) -> List[int]:
        """Reproducibly permuted case indices for one training epoch.

        Args:
            n: Number of cases in the training dataset.

        Returns:
            A permuted list of integers in ``[0, n)``.
        """
        return torch.randperm(n, generator=self._shuffle_gen).tolist()
