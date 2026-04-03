"""Training entry-point with clean log output.

Configures logging before pytorch3dunet imports to:
  - Eliminate duplicate log lines (pytorch3dunet adds its own handlers)
  - Suppress per-iteration spam (only log every log_after_iters)
  - Hide DEBUG noise (h5py converters, etc.)
  - Suppress the giant config-dict dump

Also adds three features on top of the base pytorch3dunet trainer:
  1. Full deterministic seeding (torch + numpy + python random + cudnn)
  2. sample_subset_size: limit number of H5 volumes loaded per phase
  3. SpatialDownsample transform: trilinear/nearest resize for VRAM savings
"""
import logging
import re
import sys

import numpy as np
import torch

# ── Logging setup (must precede any pytorch3dunet import) ────────────────────

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)


class _IterationFilter(logging.Filter):
    """Only pass 'Training iteration [N/M]' lines every 500 iterations."""

    _pat = re.compile(r"Training iteration \[(\d+)/")

    def filter(self, record):
        m = self._pat.search(record.getMessage())
        if m:
            return int(m.group(1)) % 500 == 0
        # Suppress the huge config dict dump
        if record.getMessage().startswith("{") and "model" in record.getMessage():
            return False
        return True


_handler.addFilter(_IterationFilter())


# ── Import and patch pytorch3dunet logging ───────────────────────────────────

import random  # noqa: E402

import pytorch3dunet.augment.transforms as aug_transforms  # noqa: E402
import pytorch3dunet.datasets.hdf5 as hdf5_module  # noqa: E402
from pytorch3dunet.unet3d import utils as _u3d_utils  # noqa: E402
from pytorch3dunet.unet3d.config import copy_config, load_config  # noqa: E402
from pytorch3dunet.unet3d import model as _model_module  # noqa: E402
from pytorch3dunet.unet3d import trainer as _trainer_module  # noqa: E402
from pytorch3dunet.unet3d.trainer import create_trainer  # noqa: E402

# ── Custom model override ─────────────────────────────────────────────────────
# Import our ResidualUNet3D and patch get_model so the library instantiates it
# instead of its own UNet3D when the config says name: ResidualUNet3D.
import sys, os  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from unet3d_model import ResidualUNet3D  # noqa: E402

_original_get_model = _model_module.get_model


def _patched_get_model(model_config):
    model_config = dict(model_config)
    name = model_config.pop("name")
    if name == "ResidualUNet3D":
        valid_keys = {"in_channels", "out_channels", "f_maps", "num_groups", "use_checkpoint"}
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        logging.getLogger("run_train").info(f"Using custom ResidualUNet3D with {filtered}")
        return ResidualUNet3D(**filtered)
    return _original_get_model({"name": name, **model_config})


_model_module.get_model = _patched_get_model
_trainer_module.get_model = _patched_get_model

# Remove per-logger handlers added by get_logger() — root handler is enough.
# This prevents every message from appearing twice.
for _lg in _u3d_utils.loggers.values():
    _lg.handlers.clear()

logger = logging.getLogger("run_train")


# ── Feature 3: SpatialDownsample transform ──────────────────────────────────
# Registered into pytorch3dunet.augment.transforms so the YAML config
# can reference it by name just like any built-in transform.

class SpatialDownsample:
    """Downsample spatial (Y, X) dimensions by a given factor.

    Uses torch.nn.functional.interpolate under the hood.
    - For raw data:   trilinear interpolation (smooth, preserves intensities)
    - For label data:  nearest-neighbor (preserves exact values)

    Only the last two axes (Y, X) are resized; Z (depth) is left unchanged
    because Z spacing is typically already coarser in CT volumes.

    Config example (in train_config.yaml transformer section):
        - name: SpatialDownsample
          scale_factor: 0.5       # 512×512 → 256×256
          mode: trilinear          # or "nearest" for labels
    """

    def __init__(self, scale_factor=0.5, mode="trilinear", **kwargs):
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, m):
        # m is a numpy array with shape (Z, Y, X)
        # interpolate needs 5D: (N, C, D, H, W)
        t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        align = False if self.mode == "nearest" else True
        t = torch.nn.functional.interpolate(
            t,
            scale_factor=(1.0, self.scale_factor, self.scale_factor),
            mode=self.mode,
            align_corners=align if self.mode != "nearest" else None,
        )
        return t.squeeze(0).squeeze(0).numpy()


# Register the transform so Transformer._transformer_class("SpatialDownsample") finds it.
aug_transforms.SpatialDownsample = SpatialDownsample


# ── Feature 2: sample_subset_size — limit volumes per phase ─────────────────

_original_traverse_h5_paths = hdf5_module.traverse_h5_paths
_sample_subset_size = None  # set from config before trainer creation


def _patched_traverse_h5_paths(file_paths):
    """Wraps the original traverse_h5_paths to limit the number of files."""
    paths = _original_traverse_h5_paths(file_paths)
    paths = sorted(paths)  # deterministic ordering before slicing
    if _sample_subset_size is not None and _sample_subset_size > 0:
        paths = paths[:_sample_subset_size]
        logger.info(f"sample_subset_size={_sample_subset_size}: using {len(paths)} volumes")
    return paths


hdf5_module.traverse_h5_paths = _patched_traverse_h5_paths


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _sample_subset_size

    config, config_path = load_config()

    # Feature 1: deterministic seeding (extends built-in manual_seed support)
    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        logger.info(f"Seeding all RNGs with {manual_seed}")
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.warning("CuDNN deterministic mode enabled — training may be slower")

    # Feature 2: read sample_subset_size from loaders config
    _sample_subset_size = config.get("loaders", {}).get("sample_subset_size", None)
    if _sample_subset_size is not None:
        logger.info(f"Subset mode: loading at most {_sample_subset_size} volumes per phase")

    # Create trainer (this triggers data loading, which will use the patched traverse)
    trainer = create_trainer(config)
    copy_config(config, config_path)
    trainer.fit()


if __name__ == "__main__":
    main()
