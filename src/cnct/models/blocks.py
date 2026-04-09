"""Shared building blocks for the 3D U-Nets used in the dual-domain cascade.

This module defines the four structural primitives that both
:class:`cnct.models.unet3d.ResidualUNet3D` and the branches of
:class:`cnct.models.dual_domain.DualDomainCascadeNet` are built from:

    - :class:`ConvBlock`: double Conv3D + GroupNorm + ReLU.
    - :class:`Encoder`: contracting path with per-stage skip connections.
    - :class:`Bottleneck`: the deepest ConvBlock (VGGT insertion point).
    - :class:`Decoder`: expanding path fusing bottleneck + skip connections.

The implementations are ported verbatim from the legacy ``3dunet/unet3d_model.py``
module — the math-preservation requirement means this file intentionally
mirrors every layer order, kernel size, padding value, and activation choice
of the original. Cosmetic changes (type hints, docstrings, imports) are the
only deltas.

Tensor conventions:
    All tensors are 5D ``[B, C, D, H, W]`` where the three spatial axes are
    ``(Depth, Height, Width)`` — i.e. ``(Z, Y, X)`` for volumes and
    ``(N_angles, det_rows, det_cols)`` for sinograms.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class ConvBlock(nn.Module):
    """Two consecutive ``Conv3d → GroupNorm → ReLU`` layers.

    This is the fundamental feature-extraction unit used at every level of
    both the encoder and decoder. Two convolutions per level (rather than one)
    give each stage enough capacity to learn meaningful features before the
    spatial resolution changes.

    GroupNorm is used instead of BatchNorm because 3D medical volumes force
    a batch size of 1-2 patches, which would make per-batch statistics too
    noisy for stable training. GroupNorm normalises within fixed channel
    groups per sample and is therefore batch-size-independent.

    Args:
        in_channels: Number of input feature maps.
        out_channels: Number of output feature maps.
        num_groups: GroupNorm group count. Must evenly divide ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block.

        Args:
            x: Input tensor of shape ``[B, in_channels, D, H, W]``.

        Returns:
            Output tensor of shape ``[B, out_channels, D, H, W]`` — spatial
            dimensions are preserved.
        """
        return self.block(x)


class Encoder(nn.Module):
    """The contracting (downsampling) path of the U-Net.

    Each encoder stage:

        1. Applies a :class:`ConvBlock` at the current resolution.
        2. Stores the output as a skip connection for the decoder.
        3. Applies 2×2×2 max-pooling to halve all spatial dimensions.

    Args:
        in_channels: Channels of the very first input (1 for grayscale CT).
        feature_maps: Feature map counts for each encoder stage; the length
            determines the number of downsampling levels.
        num_groups: GroupNorm group count passed to every :class:`ConvBlock`.
        use_checkpoint: If ``True``, wrap each stage's :class:`ConvBlock` in
            ``torch.utils.checkpoint`` during training to save activation
            memory at the cost of one extra forward pass per block.
    """

    def __init__(
        self,
        in_channels: int,
        feature_maps: List[int],
        num_groups: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_channels = in_channels
        for num_features in feature_maps:
            self.stages.append(
                ConvBlock(current_channels, num_features, num_groups)
            )
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = num_features

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the input through every encoder stage.

        Args:
            x: Input tensor of shape ``[B, in_channels, D, H, W]``.

        Returns:
            A tuple ``(x, skip_connections)`` where:

                * ``x`` is the downsampled tensor after the final pool, ready
                  for the bottleneck.
                * ``skip_connections`` is a list of pre-pool stage outputs
                  ordered from highest to lowest resolution.
        """
        skip_connections: List[torch.Tensor] = []
        for stage, pool in zip(self.stages, self.pools):
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections


class Bottleneck(nn.Module):
    """The deepest :class:`ConvBlock`, connecting encoder to decoder.

    The commented ``### [INSERT VGGT MODULE HERE IN THE FUTURE] ###`` markers
    flag the intended insertion point for a future Vision Graph Guiding
    Transformer. The bottleneck is chosen because spatial dimensions are
    smallest there (so self-attention is feasible) and global context is
    most valuable (the encoder has already extracted local features).

    Args:
        in_channels: Channels arriving from the last encoder pool.
        out_channels: Channels fed to the first decoder upconv.
        num_groups: GroupNorm group count for the inner :class:`ConvBlock`.
        use_checkpoint: Enable gradient checkpointing during training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        ### [INSERT VGGT MODULE HERE IN THE FUTURE] ###
        self.block = ConvBlock(in_channels, out_channels, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the lowest-resolution feature map.

        Args:
            x: Shape ``[B, in_channels, D_min, H_min, W_min]``.

        Returns:
            Tensor of shape ``[B, out_channels, D_min, H_min, W_min]``.
        """
        ### [INSERT VGGT MODULE HERE IN THE FUTURE] ###
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


class Decoder(nn.Module):
    """The expanding (upsampling) path of the U-Net.

    Each decoder stage:

        1. Upsamples by 2× with ``ConvTranspose3d``.
        2. Zero-pads the upsampled tensor on the right/bottom/back if the
           skip connection is larger (off-by-one caused by odd spatial sizes
           being floored by MaxPool3d).
        3. Concatenates the upsampled tensor with the corresponding encoder
           skip connection along the channel dimension.
        4. Applies a :class:`ConvBlock` to fuse the combined information.

    Skip connections are consumed in reverse order (deepest first).

    Args:
        feature_maps: Encoder feature maps in original order; reversed
            internally to drive the upward path.
        bottleneck_channels: Number of channels coming out of the bottleneck.
        num_groups: GroupNorm group count for every :class:`ConvBlock`.
        use_checkpoint: Enable gradient checkpointing during training.
    """

    def __init__(
        self,
        feature_maps: List[int],
        bottleneck_channels: int,
        num_groups: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.upconvs = nn.ModuleList()
        self.stages = nn.ModuleList()

        reversed_maps = list(reversed(feature_maps))

        current_channels = bottleneck_channels
        for num_features in reversed_maps:
            self.upconvs.append(
                nn.ConvTranspose3d(
                    current_channels,
                    num_features,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.stages.append(
                ConvBlock(num_features * 2, num_features, num_groups)
            )
            current_channels = num_features

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor],
    ) -> torch.Tensor:
        """Upsample through all decoder stages, fusing with skip connections.

        Args:
            x: Bottleneck output, shape
                ``[B, bottleneck_channels, D_min, H_min, W_min]``.
            skip_connections: Encoder skip tensors ordered from highest to
                lowest resolution. Consumed in reverse.

        Returns:
            Decoded feature map at the original input resolution,
            ``[B, feature_maps[0], D, H, W]``.
        """
        for upconv, stage, skip in zip(
            self.upconvs, self.stages, reversed(skip_connections)
        ):
            x = upconv(x)

            if x.shape != skip.shape:
                x = nn.functional.pad(
                    x,
                    [
                        0, skip.shape[4] - x.shape[4],  # Width  pad
                        0, skip.shape[3] - x.shape[3],  # Height pad
                        0, skip.shape[2] - x.shape[2],  # Depth  pad
                    ],
                )

            x = torch.cat([skip, x], dim=1)

            if self.use_checkpoint and self.training:
                x = grad_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)

        return x
