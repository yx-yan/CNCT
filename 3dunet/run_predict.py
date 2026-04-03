"""Prediction entry-point that loads the custom ResidualUNet3D.

The base pytorch3dunet predict module uses the library's own model classes,
which have different state_dict keys than our custom ResidualUNet3D.  This
wrapper patches get_model (same approach as run_train.py) so the checkpoint
loads correctly.

The model was trained on spatially downsampled patches (0.5x in Y,X).  At
inference time the predictor must see full-resolution patches in and out so
that its spatial indexing is consistent.  We wrap the model in a
DownsampleWrapper that downsamples the input, runs the model, and upsamples
the output — so SpatialDownsample is NOT needed in test_config.yaml.
"""
import logging
import sys
import os

import numpy as np
import torch
import torch.nn as nn

# ── Logging setup ──────────────────────────────────────────────────────────────
_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)

# ── Import pytorch3dunet internals ─────────────────────────────────────────────
from pytorch3dunet.unet3d import model as _model_module  # noqa: E402
from pytorch3dunet.unet3d import utils as _u3d_utils  # noqa: E402
from pytorch3dunet.predict import main as _predict_main  # noqa: E402

# ── Custom model override ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from unet3d_model import ResidualUNet3D  # noqa: E402

logger = logging.getLogger("run_predict")

_original_get_model = _model_module.get_model


class DownsampleWrapper(nn.Module):
    """Wraps a model trained on 0.5x downsampled Y,X patches.

    At inference: downsample input → run model → upsample output back to
    the original spatial size.  This makes the predictor's spatial indexing
    work correctly without needing SpatialDownsample in the test transforms.
    """

    def __init__(self, model, scale_factor=0.5):
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor

    def load_state_dict(self, state_dict, **kwargs):
        """Load checkpoint into the inner model (keys have no 'model.' prefix)."""
        self.model.load_state_dict(state_dict, **kwargs)

    def forward(self, x):
        # x: (N, C, D, H, W) — full resolution
        orig_shape = x.shape[2:]  # (D, H, W)
        # Downsample Y,X only (keep D unchanged)
        x_down = torch.nn.functional.interpolate(
            x,
            scale_factor=(1.0, self.scale_factor, self.scale_factor),
            mode="trilinear",
            align_corners=False,
        )
        out_down = self.model(x_down)
        # Upsample back to original spatial size
        out = torch.nn.functional.interpolate(
            out_down,
            size=orig_shape,
            mode="trilinear",
            align_corners=False,
        )
        return out


class FullVolumeWrapper(nn.Module):
    """Wraps a model trained on full volumes resized to a fixed target shape.

    At inference: resize input to target_shape → run model → resize output
    back to the original spatial dimensions.
    """

    def __init__(self, model, target_shape):
        super().__init__()
        self.model = model
        self.target_shape = tuple(target_shape)  # (D, H, W)

    def load_state_dict(self, state_dict, **kwargs):
        """Load checkpoint into the inner model (keys have no 'model.' prefix)."""
        self.model.load_state_dict(state_dict, **kwargs)

    def forward(self, x):
        # x: (N, C, D, H, W) — original resolution
        orig_shape = x.shape[2:]  # (D, H, W)
        x_resized = torch.nn.functional.interpolate(
            x, size=self.target_shape, mode="trilinear", align_corners=False,
        )
        out_resized = self.model(x_resized)
        out = torch.nn.functional.interpolate(
            out_resized, size=orig_shape, mode="trilinear", align_corners=False,
        )
        return out


def _detect_training_mode(model_path):
    """Peek at a checkpoint to detect whether it was trained in full-volume mode.

    Returns (mode, target_shape) where mode is 'full_volume' or 'patch' and
    target_shape is a tuple (D, H, W) or None.
    """
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if state.get("training_mode") == "full_volume":
            target_shape = tuple(state["target_shape"])
            return "full_volume", target_shape
    except Exception:
        pass
    return "patch", None


def _patched_get_model(model_config):
    model_config = dict(model_config)
    name = model_config.pop("name")
    if name == "ResidualUNet3D":
        valid_keys = {"in_channels", "out_channels", "f_maps", "num_groups"}
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        logger.info(f"Using custom ResidualUNet3D with {filtered}")
        model = ResidualUNet3D(**filtered)

        # Detect training mode from checkpoint
        model_path = model_config.get("model_path") or _find_model_path()
        mode, target_shape = _detect_training_mode(model_path)

        if mode == "full_volume":
            logger.info(f"Wrapping model with FullVolumeWrapper (target_shape={target_shape})")
            return FullVolumeWrapper(model, target_shape)
        else:
            logger.info("Wrapping model with DownsampleWrapper (0.5x Y,X)")
            return DownsampleWrapper(model)
    return _original_get_model({"name": name, **model_config})


def _find_model_path():
    """Extract model_path from sys.argv (--config) for checkpoint detection."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            try:
                import yaml
                with open(sys.argv[i + 1]) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("model_path")
            except Exception:
                pass
    return None


_model_module.get_model = _patched_get_model

# Also patch the predict module's local import of get_model
import pytorch3dunet.predict as _predict_module  # noqa: E402
_predict_module.get_model = _patched_get_model

# Remove duplicate handlers
for _lg in _u3d_utils.loggers.values():
    _lg.handlers.clear()


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _predict_main()
