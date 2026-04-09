"""Neural network architectures for the dual-domain cascade.

Public API:

    - :class:`ResidualUNet3D` — single-domain image-only refinement net.
    - :class:`SinogramUNet3D`, :class:`VolumeUNet3D` — the two branches of
      the dual-domain cascade.
    - :class:`DifferentiableBackprojection` — the TIGRE-backed bridge.
    - :class:`DualDomainCascadeNet` — the end-to-end model.
"""
from .blocks import Bottleneck, ConvBlock, Decoder, Encoder
from .dual_domain import (
    DifferentiableBackprojection,
    DualDomainCascadeNet,
    SinogramUNet3D,
    VolumeUNet3D,
)
from .unet3d import ResidualUNet3D

__all__ = [
    "Bottleneck",
    "ConvBlock",
    "Decoder",
    "Encoder",
    "DifferentiableBackprojection",
    "DualDomainCascadeNet",
    "ResidualUNet3D",
    "SinogramUNet3D",
    "VolumeUNet3D",
]
