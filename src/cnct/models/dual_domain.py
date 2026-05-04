"""Dual-domain cascaded network for sparse-view CT reconstruction.

Architecture::

    Sinogram ──▶ Branch A (SinogramUNet3D) ──▶ sino_features [B,C,Nang,Dr,Dc]
                                                      │
                                                      ▼
                                       DBP (per-channel TIGRE Atb)
                                                      │
                                                      ▼
                                        vol_features  [B,C,Z,Y,X]
                                                      │
    FDK volume ─────────────────▶ cat([fdk, vol_features]) [B,1+C,Z,Y,X]
                                                      │
                                                      ▼
                                       Branch B (VolumeUNet3D)
                                                      │
                                                      ▼
                                              output [B,1,Z,Y,X]

The module is a verbatim port of ``3dunet/dual_domain_model.py``. The
math-preservation requirement means every ``del``, ``empty_cache``, and
``autocast`` call in :meth:`DualDomainCascadeNet.forward` is preserved
exactly — these are load-bearing for correctness (TIGRE CUDA interop)
and for VRAM budget under gradient checkpointing.

Tensor conventions:
    - Sinogram domain: ``[B, C, N_angles, det_rows, det_cols]``
    - Volume domain:   ``[B, C, Z, Y, X]``
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import tigre

from .blocks import Bottleneck, Decoder, Encoder


class SinogramUNet3D(nn.Module):
    """3D U-Net feature extractor on the projection (sinogram) domain.

    Processes a raw sparse sinogram ``[B, 1, N_angles, det_rows, det_cols]``
    and outputs multi-channel feature maps at the same spatial resolution.
    Pure feature extractor — no residual learning, no output activation.

    Args:
        in_channels: Input channel count (1 for a raw sinogram).
        out_features: Output feature channel count. Each channel is
            independently backprojected by the DBP layer, so more features
            give richer volume representation at the cost of more TIGRE
            backprojection calls.
        f_maps: Feature map counts at each U-Net level. The last entry is
            the bottleneck.
        num_groups: GroupNorm group count.
        use_checkpoint: Enable gradient checkpointing in encoder, bottleneck,
            and decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 8,
        f_maps: Tuple[int, ...] = (16, 32, 64),
        num_groups: int = 8,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} must divide f_maps entry {f}"
            )
        assert len(f_maps) >= 2, (
            "f_maps needs at least 2 entries (encoder + bottleneck)"
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
            encoder_maps[0], out_features, kernel_size=1
        )

        if use_checkpoint:
            for m in self.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-channel features from the sinogram.

        Args:
            x: Shape ``[B, 1, N_angles, det_rows, det_cols]``.

        Returns:
            Shape ``[B, out_features, N_angles, det_rows, det_cols]``.
        """
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.final_conv(x)


class _TIGREBackprojectFn(torch.autograd.Function):
    """Autograd function wrapping TIGRE's adjoint pair ``(Atb, Ax)``.

    Forward uses matched backprojection ``tigre.Atb`` (sinogram → volume);
    backward uses forward projection ``tigre.Ax`` (the mathematical adjoint
    of ``Atb``). The adjoint relationship guarantees that for all volumes
    ``x`` and sinograms ``p``::

        <Ax, p> = <x, Atb(p)>

    TIGRE operates on numpy arrays via its own CUDA kernels, outside
    PyTorch's autograd tape. ``torch.cuda.synchronize`` and
    ``torch.cuda.empty_cache`` calls ensure correct ordering and memory
    sharing between PyTorch and TIGRE.
    """

    @staticmethod
    def forward(ctx, sinogram, geo, angles):  # type: ignore[override]
        """Backproject a single-channel sinogram into volume space.

        Args:
            sinogram: Torch tensor of shape ``[N_angles, det_rows, det_cols]``.
            geo: :class:`tigre.utilities.geometry.Geometry` for the case.
            angles: NumPy array of shape ``[N_angles]`` in radians.

        Returns:
            Torch tensor of shape ``[Z, Y, X]`` on the same device as
            ``sinogram``.
        """
        ctx.geo = geo
        ctx.angles = angles
        device = sinogram.device

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        sino_np = np.ascontiguousarray(
            sinogram.detach().cpu().numpy().astype(np.float32)
        )

        vol_np = tigre.Atb(sino_np, geo, angles).astype(np.float32)

        result = torch.from_numpy(np.ascontiguousarray(vol_np)).to(device)
        torch.cuda.synchronize()
        return result

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        """Forward-project the volume gradient to sinogram space (adjoint of Atb)."""
        device = grad_output.device

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        grad_np = np.ascontiguousarray(
            grad_output.detach().cpu().numpy().astype(np.float32)
        )

        sino_grad_np = tigre.Ax(grad_np, ctx.geo, ctx.angles).astype(np.float32)

        sino_grad = torch.from_numpy(np.ascontiguousarray(sino_grad_np)).to(device)
        torch.cuda.synchronize()
        return sino_grad, None, None


class DifferentiableBackprojection(nn.Module):
    """Per-channel differentiable backprojection from sinogram to volume.

    Applies TIGRE's matched backprojection ``Atb`` independently to each
    feature channel, mapping multi-channel sinogram features into a
    multi-channel volume::

        [B, C, N_angles, det_rows, det_cols]  →  [B, C, Z, Y, X]

    This is **not** filtered backprojection — the ramp filter is omitted.
    The unfiltered backprojection serves as a learned geometric mapping;
    Branch B compensates for any resulting blur.

    Geometry and angles are passed at forward time since they are
    case-specific (volumes differ in voxel grid size and spacing).
    """

    def forward(
        self,
        sinogram_features: torch.Tensor,
        geo: "tigre.utilities.geometry.Geometry",
        angles: np.ndarray,
    ) -> torch.Tensor:
        """Run per-channel backprojection.

        Args:
            sinogram_features: Shape ``[B, C, N_angles, det_rows, det_cols]``.
            geo: Case-specific TIGRE geometry.
            angles: NumPy array of shape ``[N_angles]`` in radians.

        Returns:
            Shape ``[B, C, Z, Y, X]`` on the same device as ``sinogram_features``.
        """
        B, C = sinogram_features.shape[:2]
        channels_out: List[torch.Tensor] = []

        for b in range(B):
            for c in range(C):
                vol = _TIGREBackprojectFn.apply(
                    sinogram_features[b, c], geo, angles,
                )
                channels_out.append(vol)

        stacked = torch.stack(channels_out, dim=0)
        return stacked.view(B, C, *stacked.shape[1:])


class VolumeUNet3D(nn.Module):
    """Residual 3D U-Net accepting multi-channel input for volume-domain fusion.

    Channel 0 of the input is the FDK reconstruction; channels ``1..C`` are
    the backprojected sinogram features from the DBP layer. The residual
    shortcut connects only to channel 0::

        output = fdk_channel - predicted_artifacts

    Args:
        in_channels: Total input channels: ``1`` (FDK) + ``C`` (backprojected
            features).
        out_channels: Output channel count (1 for single-energy CT).
        f_maps: Feature map counts at each level.
        num_groups: GroupNorm group count.
        use_checkpoint: Enable gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        f_maps: Tuple[int, ...] = (16, 32, 64, 128, 256),
        num_groups: int = 8,
        use_checkpoint: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} must divide f_maps entry {f}"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual shortcut on channel 0.

        Args:
            x: Shape ``[B, in_channels, Z, Y, X]``. Channel 0 must be the
                FDK reconstruction; channels ``1..C`` are backprojected
                sinogram features.

        Returns:
            Shape ``[B, 1, Z, Y, X]``.
        """
        # [use_fdk 控制点 3] 残差连接——是否从输入中提取 FDK 作为残差基底
        # use_fdk=True  → use_residual=True  → 取 channel 0 (FDK) 存为 residual
        # use_fdk=False → use_residual=False → 不提取，residual=None
        residual = x[:, 0:1] if self.use_residual else None

        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        out = self.final_conv(x)

        if self.use_residual:
            # use_fdk=True: 网络学习伪影，最终输出 = FDK 原图 − 预测伪影
            return residual - out
        # use_fdk=False: 网络直接回归重建结果，无残差
        return out


class DualDomainCascadeNet(nn.Module):
    """End-to-end dual-domain cascaded network for sparse-view CT.

    Combines sinogram-domain feature extraction (Branch A), differentiable
    backprojection (DBP), and volume-domain residual refinement (Branch B)
    into a single trainable model.

    Args:
        sinogram_out_features: Branch A output feature channels.
        sinogram_f_maps: Branch A U-Net feature map sizes.
        volume_f_maps: Branch B U-Net feature map sizes.
        num_groups: GroupNorm group count shared by both branches.
        use_checkpoint: Enable gradient checkpointing in both branches.
    """

    def __init__(
        self,
        sinogram_out_features: int = 4,
        sinogram_f_maps: Tuple[int, ...] = (8, 16, 32),
        volume_f_maps: Tuple[int, ...] = (8, 16, 32, 64, 128),
        num_groups: int = 8,
        use_checkpoint: bool = True,
        use_fdk: bool = True,
    ) -> None:
        super().__init__()

        self.sinogram_out_features = sinogram_out_features
        self.use_checkpoint = use_checkpoint
        self.use_fdk = use_fdk

        self.branch_a = SinogramUNet3D(
            in_channels=1,
            out_features=sinogram_out_features,
            f_maps=sinogram_f_maps,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
        )

        self.dbp = DifferentiableBackprojection()

        # [use_fdk 控制点 1] Branch B 输入通道数
        # use_fdk=True:  1 (FDK, channel 0) + C (DBP特征) = 1+C 个输入通道
        # use_fdk=False: 只有 C 个 DBP 特征通道，不包含 FDK
        branch_b_in = (1 + sinogram_out_features) if use_fdk else sinogram_out_features
        # [use_fdk 控制点 2] Branch B 残差连接开关
        # use_fdk=True:  use_residual=True  → output = FDK − predicted_artifacts
        # use_fdk=False: use_residual=False → output = 直接回归，无残差
        self.branch_b = VolumeUNet3D(
            in_channels=branch_b_in,
            out_channels=1,
            f_maps=volume_f_maps,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
            use_residual=use_fdk,
        )

    def forward(
        self,
        fdk_volume: torch.Tensor | None,
        sinogram: torch.Tensor,
        geo: "tigre.utilities.geometry.Geometry",
        angles: np.ndarray,
    ) -> torch.Tensor:
        """Full dual-domain forward pass.

        Intended to be called under ``torch.amp.autocast("cuda")``. The DBP
        layer internally forces float32 for TIGRE compatibility.

        Args:
            fdk_volume: Shape ``[B, 1, Z, Y, X]``, normalised to ``[-1, 1]``.
            sinogram: Shape ``[B, 1, N_angles, det_rows, det_cols]``,
                normalised to ``[0, 1]``.
            geo: Case-specific cone-beam geometry.
            angles: NumPy array of shape ``[N_angles]`` in radians.

        Returns:
            Shape ``[B, 1, Z, Y, X]`` — the enhanced reconstruction in the
            same normalised range as ``fdk_volume``.
        """
        # --- Branch A: sinogram feature extraction ---
        if self.use_checkpoint and self.training:
            sino_features = grad_checkpoint(
                self.branch_a, sinogram, use_reentrant=False,
            )
        else:
            sino_features = self.branch_a(sinogram)

        # --- DBP: backproject to volume space ---
        del sinogram
        torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=False):
            vol_features = self.dbp(
                sino_features.float(), geo, angles,
            )

        del sino_features
        torch.cuda.empty_cache()

        # [use_fdk 控制点 4] 前向传播中的 FDK 融合分支
        if self.use_fdk:
            # use_fdk=True: 将 FDK [B,1,Z,Y,X] 与 DBP特征 [B,C,Z,Y,X]
            # 在 channel 维拼接 → [B, 1+C, Z, Y, X]，FDK 固定在 channel 0
            if fdk_volume is None:
                raise ValueError("fdk_volume is required when use_fdk=True")
            fused = torch.cat([fdk_volume, vol_features], dim=1)
            del vol_features
            torch.cuda.empty_cache()
            # Branch B 内部取 channel 0 做残差: output = FDK − artifacts
            output = self.branch_b(fused)
        else:
            # use_fdk=False: fdk_volume 参数被完全忽略，不参与任何计算
            # 仅从 DBP 特征 [B,C,Z,Y,X] 直接回归重建结果
            output = self.branch_b(vol_features)
            del vol_features
            torch.cuda.empty_cache()
        return output
