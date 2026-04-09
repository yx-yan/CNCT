"""Residual 3D U-Net for sparse-view CT reconstruction enhancement.

This module defines :class:`ResidualUNet3D`, the single-domain image-only
refinement network. It is retained because :class:`cnct.models.dual_domain`'s
``VolumeUNet3D`` is a near-duplicate that only differs in accepting a
multi-channel input and restricting the residual shortcut to channel 0.

The implementation is ported verbatim from the legacy
``3dunet/unet3d_model.py`` module. The math-preservation requirement means
this file intentionally mirrors every layer order, kernel size, and residual
connection of the original. Cosmetic changes (type hints, docstrings,
imports) are the only deltas.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from .blocks import Bottleneck, Decoder, Encoder


class ResidualUNet3D(nn.Module):
    """3D U-Net with residual (artifact-predicting) learning.

    The network predicts the artifact component of the input rather than the
    clean image directly::

        artifacts = Network(Input_FDK)
        Clean_Image = Input_FDK - artifacts

    The output has **no activation function** — this is a pure regression
    network outputting continuous values (linear attenuation coefficients).

    Args:
        in_channels: Number of input channels (1 for single-energy CT). Must
            equal ``out_channels`` for the residual subtraction to be valid.
        out_channels: Number of output channels.
        f_maps: Feature map counts at each level. The first ``N-1`` entries
            define the encoder stages; the last entry defines the bottleneck
            depth.
        num_groups: GroupNorm group count. Must evenly divide every entry in
            ``f_maps``.
        use_checkpoint: If ``True``, enable gradient checkpointing in every
            :class:`ConvBlock` and disable ``inplace`` ReLU.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        f_maps: Tuple[int, ...] = (16, 32, 64, 128, 256),
        num_groups: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        assert in_channels == out_channels, (
            "Residual learning requires in_channels == out_channels "
            f"(got {in_channels} and {out_channels})"
        )
        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} does not evenly divide f_maps "
                f"entry {f}. Every feature map size must be divisible by "
                f"num_groups for GroupNorm to work."
            )

        encoder_maps = list(f_maps[:-1])
        bottleneck_in = f_maps[-2]
        bottleneck_out = f_maps[-1]

        self.encoder = Encoder(
            in_channels, encoder_maps, num_groups, use_checkpoint
        )
        self.bottleneck = Bottleneck(
            bottleneck_in, bottleneck_out, num_groups, use_checkpoint
        )
        self.decoder = Decoder(
            encoder_maps, bottleneck_out, num_groups, use_checkpoint
        )
        self.final_conv = nn.Conv3d(
            encoder_maps[0], out_channels, kernel_size=1
        )

        if use_checkpoint:
            for m in self.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass: predict artifacts and subtract from the input.

        Args:
            x: Coarse FDK reconstruction, shape ``[B, in_channels, D, H, W]``.
            **kwargs: If ``return_logits=True`` is passed, also return the raw
                artifact prediction alongside the refined output.

        Returns:
            Enhanced volume of shape ``[B, out_channels, D, H, W]``. If
            ``return_logits=True``, a tuple ``(output, artifacts)``.
        """
        residual = x
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        artifacts = self.final_conv(x)
        output = residual - artifacts

        if kwargs.get("return_logits", False):
            return output, artifacts
        return output
