"""Full-volume HDF5 dataset for 3D UNet training.

Loads entire CT volumes from HDF5 files and resizes them to a uniform
target shape, enabling full-volume (or large sub-volume) training that
avoids the patch-boundary artifacts of traditional patch-based approaches.

Each HDF5 file contains:
    raw   : float32 (Z, Y, X) — FDK reconstruction in μ units
    label : float32 (Z, Y, X) — ground-truth CT in μ units

The dataset resizes both raw and label to `target_shape` using trilinear
interpolation, applies fixed-range normalization to [-1, 1], and
optionally applies spatial augmentations during training.
"""

import logging
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FullVolumeHDF5Dataset(Dataset):
    """Dataset that loads full HDF5 volumes resized to a fixed target shape.

    Parameters
    ----------
    h5_dir : str or Path
        Directory containing .h5 files (each with 'raw' and 'label' datasets).
    target_shape : tuple[int, int, int]
        Target (D, H, W) after resize. E.g. (128, 256, 256).
    phase : str
        'train' or 'val'. Controls whether augmentations are applied.
    norm_min : float
        Lower bound of the fixed normalization range (μ units).
    norm_max : float
        Upper bound of the fixed normalization range (μ units).
    augment_config : dict or None
        Augmentation settings. Keys: random_flip (bool), random_rotate90
        (bool), random_rotate (dict with 'enabled', 'angle_spectrum').
    subset_size : int or None
        If set, only use the first N files (sorted alphabetically).
    """

    def __init__(
        self,
        h5_dir,
        target_shape,
        phase="train",
        norm_min=-0.02,
        norm_max=0.08,
        augment_config=None,
        subset_size=None,
    ):
        self.h5_dir = Path(h5_dir)
        self.target_shape = tuple(target_shape)  # (D, H, W)
        self.phase = phase
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.augment = phase == "train" and augment_config is not None
        self.augment_config = augment_config or {}

        # Discover HDF5 files
        self.file_paths = sorted(self.h5_dir.glob("*.h5"))
        if subset_size is not None and subset_size > 0:
            self.file_paths = self.file_paths[:subset_size]

        if not self.file_paths:
            raise FileNotFoundError(f"No .h5 files found in {self.h5_dir}")

        logger.info(
            f"FullVolumeHDF5Dataset ({phase}): {len(self.file_paths)} volumes, "
            f"target_shape={self.target_shape}"
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load raw and label from HDF5
        with h5py.File(path, "r") as f:
            raw = f["raw"][:].astype(np.float32)      # (Z, Y, X)
            label = f["label"][:].astype(np.float32)   # (Z, Y, X)

        # Convert to tensors: (1, 1, Z, Y, X) for F.interpolate
        raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)
        label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)

        # Resize to target_shape
        raw_t = F.interpolate(raw_t, size=self.target_shape, mode="trilinear", align_corners=False)
        label_t = F.interpolate(label_t, size=self.target_shape, mode="trilinear", align_corners=False)

        # Squeeze back to (1, D, H, W)
        raw_t = raw_t.squeeze(0)    # (1, D, H, W)
        label_t = label_t.squeeze(0)

        # Fixed-range normalization: μ → [-1, 1]
        raw_t = self._normalize(raw_t)
        label_t = self._normalize(label_t)

        # Augmentations (training only)
        if self.augment:
            raw_t, label_t = self._apply_augmentations(raw_t, label_t)

        return raw_t, label_t

    def _normalize(self, x):
        """Map from [norm_min, norm_max] to [-1, 1]."""
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-10)
        x = torch.clamp(2.0 * x - 1.0, -1.0, 1.0)
        return x

    def _apply_augmentations(self, raw, label):
        """Apply random spatial augmentations identically to raw and label.

        All operations are on tensors of shape (1, D, H, W).
        """
        # Random flip along each spatial axis
        if self.augment_config.get("random_flip", True):
            for dim in [1, 2, 3]:  # D, H, W
                if random.random() < 0.5:
                    raw = torch.flip(raw, [dim])
                    label = torch.flip(label, [dim])

        # Random 90-degree rotation in the H-W plane
        if self.augment_config.get("random_rotate90", True):
            k = random.randint(0, 3)
            if k > 0:
                raw = torch.rot90(raw, k, [2, 3])
                label = torch.rot90(label, k, [2, 3])

        # Random small-angle rotation in the ZY (D-H) plane
        rot_cfg = self.augment_config.get("random_rotate", {})
        if rot_cfg.get("enabled", False):
            angle_spectrum = rot_cfg.get("angle_spectrum", 20)
            angle = random.uniform(-angle_spectrum, angle_spectrum)
            if abs(angle) > 0.5:
                raw = self._rotate_zy(raw, angle)
                label = self._rotate_zy(label, angle)

        return raw, label

    @staticmethod
    def _rotate_zy(vol, angle_deg):
        """Rotate a (1, D, H, W) volume by `angle_deg` in the D-H plane.

        Uses affine_grid + grid_sample for differentiable rotation.
        """
        angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Build 3D affine matrix that rotates in the D-H plane (dims 0,1 of DHW)
        # Grid sample expects (N, C, D, H, W) and a (N, D, H, W, 3) grid
        theta = torch.tensor([
            [1.0,  0.0,    0.0,   0.0],
            [0.0,  cos_a, -sin_a, 0.0],
            [0.0,  sin_a,  cos_a, 0.0],
        ], dtype=vol.dtype).unsqueeze(0)  # (1, 3, 4)

        # vol is (1, D, H, W), need (1, 1, D, H, W) for grid_sample
        vol_5d = vol.unsqueeze(0)
        grid = F.affine_grid(theta, vol_5d.shape, align_corners=False)
        rotated = F.grid_sample(vol_5d, grid, mode="bilinear", padding_mode="reflection",
                                align_corners=False)
        return rotated.squeeze(0)  # back to (1, D, H, W)