"""Inference driver for the dual-domain cascade.

Mirrors the Trainer's structure: load a validated
:class:`~cnct.config.schema.InferenceCfg`, build the model, restore a
checkpoint, stream cases from a :class:`~cnct.data.DualDomainDataset`,
and write denormalised predictions back to disk as ``<case>_recon.npy``
files ready for the evaluation pipeline.

The Predictor preserves the legacy behaviour exactly: predictions are
computed on the (possibly downsampled) target grid, then upsampled back
to the original FDK shape so the evaluation script can compare them
voxelwise against the ground truth.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.schema import InferenceCfg
from ..data import DualDomainDataset, unit_range_to_mu
from ..models import DualDomainCascadeNet
from ..utils.device import resolve_device
from ..utils.io import ensure_dir, safe_load_npy

logger = logging.getLogger(__name__)

__all__ = ["Predictor"]


class Predictor:
    """Run a trained dual-domain cascade over a dataset split.

    Args:
        cfg: Validated inference configuration.
        dataset: Streaming dataset for the chosen split. The dataset
            already applies the same downsampling the trainer used, so
            the Predictor only has to reverse the downsampling after
            the forward pass.
    """

    def __init__(self, cfg: InferenceCfg, dataset: DualDomainDataset) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.device = resolve_device(cfg.device)
        logger.info("Predictor device: %s", self.device)

        self.model = self._build_model().to(self.device)
        self._load_checkpoint(cfg.checkpoint)
        self.model.eval()

        self.output_dir = ensure_dir(cfg.output_dir)
        logger.info("Output directory: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_model(self) -> DualDomainCascadeNet:
        """Instantiate the cascade with checkpointing disabled for inference."""
        m = self.cfg.model
        return DualDomainCascadeNet(
            sinogram_out_features=m.sinogram_out_features,
            sinogram_f_maps=m.sinogram_f_maps,
            volume_f_maps=m.volume_f_maps,
            num_groups=m.num_groups,
            use_checkpoint=False,
        )

    def _load_checkpoint(self, path: Path) -> None:
        """Restore model weights from a saved checkpoint.

        Args:
            path: Path to a ``best_checkpoint.pytorch``-style file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint: %s (epoch=%s, best_psnr=%s)",
            path,
            ckpt.get("epoch", "?"),
            ckpt.get("best_psnr", "?"),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> Dict[str, Path]:
        """Run inference over every case in :attr:`dataset`.

        Existing output files are skipped so the run can be resumed
        safely after an interrupted job.

        Returns:
            Dict mapping each processed case ID to the path of its
            ``<case>_recon.npy`` output file.
        """
        results: Dict[str, Path] = {}
        t0 = time.time()
        n = len(self.dataset)

        for idx in range(n):
            sample = self.dataset[idx]
            case_id = sample["case_id"]
            out_path = self.output_dir / f"{case_id}_recon.npy"
            if out_path.is_file():
                logger.info("[%d/%d] %s — skip (exists)", idx + 1, n, case_id)
                results[case_id] = out_path
                continue

            logger.info("[%d/%d] %s", idx + 1, n, case_id)
            pred_mu = self._predict_case(sample)
            np.save(out_path, pred_mu)
            logger.info(
                "    saved %s  shape=%s  range=[%.4f, %.4f]",
                out_path.name,
                pred_mu.shape,
                float(pred_mu.min()),
                float(pred_mu.max()),
            )
            results[case_id] = out_path

        logger.info(
            "Prediction complete: %d case(s) in %.1fs",
            len(results),
            time.time() - t0,
        )
        return results

    # ------------------------------------------------------------------
    # Per-case helpers
    # ------------------------------------------------------------------

    def _predict_case(self, sample: Dict[str, object]) -> np.ndarray:
        """Forward pass + denormalisation + upsample for a single sample.

        Args:
            sample: One dict from :class:`DualDomainDataset`.

        Returns:
            ``float32`` array in mu units with the **original** (pre-
            downsampling) ``(Z, Y, X)`` shape of the FDK input.
        """
        sino = sample["sinogram"].unsqueeze(0).to(self.device, non_blocking=True)
        fdk = sample["fdk_volume"].unsqueeze(0).to(self.device, non_blocking=True)
        geo = sample["geo"]
        angles = sample["angles"]

        with torch.amp.autocast("cuda", enabled=self.cfg.amp):
            pred_norm = self.model(fdk, sino, geo, angles)

        pred_np = pred_norm.squeeze(0).squeeze(0).detach().float().cpu().numpy()
        pred_mu = unit_range_to_mu(
            pred_np, self.cfg.data.mu_min, self.cfg.data.mu_max
        )

        target_shape = self._original_fdk_shape(sample["case_id"])
        pred_mu = self._spatial_upsample(pred_mu, target_shape)

        del sino, fdk, pred_norm
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pred_mu

    def _original_fdk_shape(self, case_id: str) -> Tuple[int, int, int]:
        """Peek at the FDK ``.npy`` header to recover the pre-downsample shape.

        ``np.load(..., mmap_mode="r")`` parses only the header, so this
        costs one disk seek and does not read the voxel data into RAM.

        Args:
            case_id: Case identifier.

        Returns:
            Tuple ``(Z, Y, X)`` giving the original FDK volume shape.
        """
        from ..utils.paths import case_fdk_path

        fdk_path = case_fdk_path(self.cfg.data.fdk_dir, case_id)
        arr = safe_load_npy(fdk_path)  # header-only via np.load (eager, small)
        shape = tuple(int(s) for s in arr.shape)
        return shape  # type: ignore[return-value]

    @staticmethod
    def _spatial_upsample(
        vol: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Trilinearly upsample a ``(Z, Y, X)`` volume to ``target_shape``.

        Args:
            vol: Downsampled prediction volume in ``[Z, Y, X]`` order.
            target_shape: Desired output shape.

        Returns:
            ``float32`` array with the requested shape.
        """
        if tuple(vol.shape) == tuple(target_shape):
            return vol.astype(np.float32, copy=False)
        t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(
            t, size=target_shape, mode="trilinear", align_corners=True
        )
        return t.squeeze(0).squeeze(0).numpy()
