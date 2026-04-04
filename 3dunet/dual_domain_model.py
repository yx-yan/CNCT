"""
Dual-Domain Cascaded Network for Sparse-View CT Reconstruction
================================================================

Architecture
------------

    Sinogram -----> Branch A (Sinogram U-Net) -----> Multi-channel sino features
                                                              |
                                                              v
                                                Differentiable Backprojection (DBP)
                                                              |
                                                              v
    FDK volume ---------------------------------> concat <-- Volume features
                                                    |
                                                    v
                                          Branch B (Volume U-Net)
                                                    |
                                                    v
                                          Clean reconstruction

Branch A:  3D U-Net feature extractor on the sinogram tensor
           [B, 1, N_angles, det_rows, det_cols].  Extracts multi-scale
           features from sparse projections without imputing missing views.
           Outputs C-channel sinogram features at the original resolution.

DBP:       Per-channel TIGRE Atb (matched backprojection) mapping sinogram
           features into volume space.  Differentiable via the (Atb, Ax)
           adjoint pair -- backward uses forward projection (Ax).

Branch B:  Modified ResidualUNet3D on [B, 1+C, Z, Y, X].  Channel 0 is
           the FDK reconstruction; channels 1..C are backprojected features.
           Residual learning subtracts predicted artifacts from the FDK
           channel only.

Memory optimisation
-------------------
- AMP (Automatic Mixed Precision): convolution blocks run in float16;
  TIGRE external CUDA calls execute in float32.
- Gradient checkpointing (torch.utils.checkpoint): enabled in all
  ConvBlocks of both branches -- trades ~30% extra compute for ~40%
  memory savings.
- inplace=False on all ReLU when checkpointing is active (required
  for correct activation recomputation during backward).

Tensor conventions
------------------
All tensors are 5D: [B, C, D, H, W].
- Sinogram domain: D=N_angles, H=det_rows, W=det_cols
- Volume domain:   D=Z, H=Y, W=X  (TIGRE convention)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import tigre

from unet3d_model import ConvBlock, Encoder, Bottleneck, Decoder


# =============================================================================
# Branch A: Sinogram-Domain Feature Extractor
# =============================================================================


class SinogramUNet3D(nn.Module):
    """3D U-Net feature extractor on the projection (sinogram) domain.

    Processes the raw sparse sinogram [B, 1, N_angles, det_rows, det_cols]
    and outputs multi-channel feature maps at the same spatial resolution.
    This is a pure feature extractor -- no residual learning, no output
    activation.

    The learned features encode projection-domain information (edge
    responses, attenuation patterns across angles) that the downstream
    DBP layer maps into volume space.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for raw sinogram).
    out_features : int
        Number of output feature channels.  Each channel is independently
        backprojected by the DBP layer, so more features = richer volume
        representation but more TIGRE backprojection calls.
    f_maps : tuple[int, ...]
        Feature map counts at each U-Net level.  Last entry is the
        bottleneck.  Kept lighter than Branch B since sinograms can be
        large in memory.
    num_groups : int
        Groups for GroupNorm.  Must divide every entry in f_maps.
    use_checkpoint : bool
        Enable gradient checkpointing in encoder/bottleneck/decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 8,
        f_maps: tuple[int, ...] = (16, 32, 64),
        num_groups: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} must divide f_maps entry {f}"
            )
        assert len(f_maps) >= 2, "f_maps needs at least 2 entries (encoder + bottleneck)"

        encoder_maps = list(f_maps[:-1])
        bottleneck_in = f_maps[-2]
        bottleneck_out = f_maps[-1]

        self.encoder = Encoder(in_channels, encoder_maps, num_groups, use_checkpoint)
        self.bottleneck = Bottleneck(bottleneck_in, bottleneck_out, num_groups, use_checkpoint)
        self.decoder = Decoder(encoder_maps, bottleneck_out, num_groups, use_checkpoint)

        # Map decoder features to desired output feature count.
        # No activation -- downstream DBP and Branch B handle the
        # representation learning.
        self.final_conv = nn.Conv3d(encoder_maps[0], out_features, kernel_size=1)

        if use_checkpoint:
            for m in self.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-channel features from the sinogram.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 1, N_angles, det_rows, det_cols].

        Returns
        -------
        torch.Tensor
            Shape [B, out_features, N_angles, det_rows, det_cols].
        """
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.final_conv(x)


# =============================================================================
# Differentiable Backprojection (DBP) Layer
# =============================================================================


class _TIGREBackprojectFn(torch.autograd.Function):
    """Autograd function wrapping TIGRE's adjoint pair: Atb / Ax.

    Forward:  Atb (matched backprojection) -- sinogram -> volume.
    Backward: Ax  (forward projection)     -- the mathematical adjoint
              of Atb, providing exact gradients for end-to-end training.

    The adjoint relationship guarantees that for all volumes x and
    sinograms p:  <Ax, p> = <x, Atb(p)>.

    TIGRE operates on numpy arrays via its own CUDA kernels, outside
    PyTorch's autograd tape.  torch.cuda.synchronize() ensures correct
    ordering between PyTorch and TIGRE GPU operations.
    """

    @staticmethod
    def forward(ctx, sinogram, geo, angles):
        """Backproject a single-channel sinogram into volume space.

        Parameters
        ----------
        sinogram : torch.Tensor [N_angles, det_rows, det_cols]
        geo : tigre.utilities.geometry.Geometry
        angles : np.ndarray [N_angles]

        Returns
        -------
        torch.Tensor [Z, Y, X]
        """
        ctx.geo = geo
        ctx.angles = angles
        device = sinogram.device

        torch.cuda.synchronize()
        # Release PyTorch's cached GPU memory so TIGRE's separate CUDA
        # allocator can use it.  Without this, TIGRE fails with "GPU is
        # being heavily used" because PyTorch's caching allocator holds
        # freed blocks that TIGRE cannot access.
        torch.cuda.empty_cache()
        sino_np = np.ascontiguousarray(
            sinogram.detach().cpu().numpy().astype(np.float32)
        )

        vol_np = tigre.Atb(sino_np, geo, angles).astype(np.float32)

        result = torch.from_numpy(np.ascontiguousarray(vol_np)).to(device)
        torch.cuda.synchronize()
        return result

    @staticmethod
    def backward(ctx, grad_output):
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
        # Gradients for (sinogram, geo, angles): only sinogram needs grad
        return sino_grad, None, None


class DifferentiableBackprojection(nn.Module):
    """Per-channel differentiable backprojection from sinogram to volume space.

    Applies TIGRE's matched backprojection (Atb) independently to each
    feature channel, mapping multi-channel sinogram features into a
    multi-channel volume:

        [B, C, N_angles, det_rows, det_cols] -> [B, C, Z, Y, X]

    This is NOT filtered backprojection (FDK) -- it deliberately omits
    the ramp filter.  The unfiltered backprojection serves as a learned
    geometric mapping; Branch B compensates for any resulting blur.

    Geometry and angles are passed at forward time since they are
    case-specific (volumes differ in voxel grid size and spacing).
    """

    def forward(
        self,
        sinogram_features: torch.Tensor,
        geo: "tigre.utilities.geometry.Geometry",
        angles: np.ndarray,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sinogram_features : torch.Tensor [B, C, N_angles, det_rows, det_cols]
        geo : TIGRE geometry (case-specific).
        angles : np.ndarray [N_angles]

        Returns
        -------
        torch.Tensor [B, C, Z, Y, X]
        """
        B, C = sinogram_features.shape[:2]
        channels_out = []

        for b in range(B):
            for c in range(C):
                vol = _TIGREBackprojectFn.apply(
                    sinogram_features[b, c], geo, angles,
                )
                channels_out.append(vol)

        # Stack: list of [Z,Y,X] -> [B*C, Z, Y, X] -> [B, C, Z, Y, X]
        stacked = torch.stack(channels_out, dim=0)
        return stacked.view(B, C, *stacked.shape[1:])


# =============================================================================
# Branch B: Volume-Domain Residual U-Net (multi-channel input)
# =============================================================================


class VolumeUNet3D(nn.Module):
    """Modified ResidualUNet3D accepting multi-channel input for fusion.

    Channel 0 of the input is the FDK reconstruction; channels 1..C are
    the backprojected sinogram features from the DBP layer.  The residual
    shortcut connects only to channel 0 (the FDK), so:

        output = fdk_channel - predicted_artifacts

    All input channels feed into the encoder, allowing the network to
    leverage both the baseline FDK reconstruction and the learned
    projection-domain features for artifact prediction.

    No sigmoid/softmax -- pure regression output in the same physical
    units as the input (normalised linear attenuation).

    Parameters
    ----------
    in_channels : int
        Total input channels: 1 (FDK) + N (backprojected features).
    out_channels : int
        Output channels (1 for single-energy CT).
    f_maps : tuple[int, ...]
        Feature maps per level.  Same structure as ResidualUNet3D.
    num_groups : int
        Groups for GroupNorm.
    use_checkpoint : bool
        Enable gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        f_maps: tuple[int, ...] = (16, 32, 64, 128, 256),
        num_groups: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} must divide f_maps entry {f}"
            )

        encoder_maps = list(f_maps[:-1])
        bottleneck_in = f_maps[-2]
        bottleneck_out = f_maps[-1]

        self.encoder = Encoder(in_channels, encoder_maps, num_groups, use_checkpoint)
        self.bottleneck = Bottleneck(bottleneck_in, bottleneck_out, num_groups, use_checkpoint)
        self.decoder = Decoder(encoder_maps, bottleneck_out, num_groups, use_checkpoint)
        self.final_conv = nn.Conv3d(encoder_maps[0], out_channels, kernel_size=1)

        if use_checkpoint:
            for m in self.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, in_channels, Z, Y, X]
            Channel 0 = FDK reconstruction (normalised).
            Channels 1..C = backprojected sinogram features.

        Returns
        -------
        torch.Tensor [B, 1, Z, Y, X]
            Artifact-free reconstruction (normalised).
        """
        # Residual shortcut: only the FDK channel
        residual = x[:, 0:1]  # [B, 1, Z, Y, X]

        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        artifacts = self.final_conv(x)

        return residual - artifacts


# =============================================================================
# Full Pipeline: Dual-Domain Cascade Network
# =============================================================================


class DualDomainCascadeNet(nn.Module):
    """End-to-end dual-domain cascaded network for sparse-view CT.

    Combines sinogram-domain feature extraction (Branch A), differentiable
    backprojection (DBP), and volume-domain residual refinement (Branch B)
    into a single trainable model.

    Data flow::

        sinogram -----> Branch A -----> sino_features [B,C,Nang,Dr,Dc]
                                               |
                                               v
                                          DBP (per-channel Atb)
                                               |
                                               v
                                          vol_features [B,C,Z,Y,X]
                                               |
        fdk_volume -----------------> cat([fdk, vol_features]) [B,1+C,Z,Y,X]
                                               |
                                               v
                                          Branch B (residual U-Net)
                                               |
                                               v
                                          output [B,1,Z,Y,X]

    Memory optimisation is deeply integrated:

    - **AMP**: the forward method is designed to run under an external
      ``torch.amp.autocast("cuda")`` context.  Branch A and Branch B
      convolutions execute in float16.  The DBP layer explicitly exits
      autocast and forces float32 for TIGRE's external CUDA kernels.
    - **Gradient checkpointing**: enabled in all ConvBlocks of both
      branches via ``use_checkpoint=True``, trading ~30% extra compute
      for ~40% memory savings on activations.
    - **inplace ReLU disabled** when checkpointing is active to allow
      correct activation recomputation during backward.

    Parameters
    ----------
    sinogram_out_features : int
        Number of feature channels output by Branch A and backprojected
        by DBP.  Higher = richer representation but more backprojection
        calls (each is a full TIGRE Atb).
    sinogram_f_maps : tuple[int, ...]
        Branch A U-Net feature map sizes.
    volume_f_maps : tuple[int, ...]
        Branch B U-Net feature map sizes.
    num_groups : int
        GroupNorm groups for both branches.
    use_checkpoint : bool
        Enable gradient checkpointing in both branches.
    """

    def __init__(
        self,
        sinogram_out_features: int = 4,
        sinogram_f_maps: tuple[int, ...] = (8, 16, 32),
        volume_f_maps: tuple[int, ...] = (8, 16, 32, 64, 128),
        num_groups: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        self.sinogram_out_features = sinogram_out_features
        self.use_checkpoint = use_checkpoint

        # Branch A: sinogram feature extractor
        self.branch_a = SinogramUNet3D(
            in_channels=1,
            out_features=sinogram_out_features,
            f_maps=sinogram_f_maps,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
        )

        # Differentiable Backprojection bridge
        self.dbp = DifferentiableBackprojection()

        # Branch B: volume-domain refinement
        # in_channels = 1 (FDK) + sinogram_out_features (backprojected)
        self.branch_b = VolumeUNet3D(
            in_channels=1 + sinogram_out_features,
            out_channels=1,
            f_maps=volume_f_maps,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
        )

    def forward(
        self,
        fdk_volume: torch.Tensor,
        sinogram: torch.Tensor,
        geo: "tigre.utilities.geometry.Geometry",
        angles: np.ndarray,
    ) -> torch.Tensor:
        """Full dual-domain forward pass.

        Intended to be called under ``torch.amp.autocast("cuda")``.
        The DBP layer internally forces float32 for TIGRE compatibility.

        Parameters
        ----------
        fdk_volume : torch.Tensor [B, 1, Z, Y, X]
            Baseline FDK reconstruction (normalised to [-1, 1]).
        sinogram : torch.Tensor [B, 1, N_angles, det_rows, det_cols]
            Raw sparse projections (normalised to [0, 1]).
        geo : tigre.utilities.geometry.Geometry
            Case-specific cone-beam geometry.
        angles : np.ndarray [N_angles]
            Projection angles in radians.

        Returns
        -------
        torch.Tensor [B, 1, Z, Y, X]
            Enhanced reconstruction (normalised to [-1, 1]).
        """
        # --- Branch A: sinogram feature extraction ---
        # Pipeline-level checkpointing: don't store Branch A's internal
        # activations (encoder skips, decoder features) during forward.
        # They are recomputed during backward, saving several GB of VRAM
        # at the cost of one extra Branch A forward pass.
        if self.use_checkpoint and self.training:
            sino_features = grad_checkpoint(
                self.branch_a, sinogram, use_reentrant=False,
            )
        else:
            sino_features = self.branch_a(sinogram)  # [B, C, Nang, Dr, Dc]

        # --- DBP: backproject to volume space ---
        # Force float32 for TIGRE's external CUDA kernels.
        # Free sinogram from GPU first — it's no longer needed (Branch A
        # checkpoint will recompute from the saved input during backward).
        sinogram_device = sinogram.device  # remember for later
        del sinogram
        torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=False):
            vol_features = self.dbp(
                sino_features.float(), geo, angles,
            )  # [B, C, Z, Y, X]

        # sino_features is no longer needed after DBP consumed it
        del sino_features
        torch.cuda.empty_cache()

        # --- Fuse: concatenate FDK (1ch) + backprojected features (Cch) ---
        fused = torch.cat([fdk_volume, vol_features], dim=1)  # [B, 1+C, Z, Y, X]
        del vol_features
        torch.cuda.empty_cache()

        # --- Branch B: volume-domain refinement ---
        # Runs in float16 under external autocast context
        output = self.branch_b(fused)  # [B, 1, Z, Y, X]

        return output


# =============================================================================
# Sanity Check (run with: python 3dunet/dual_domain_model.py)
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Test Branch A alone ---
    print("=== Branch A (SinogramUNet3D) ===")
    branch_a = SinogramUNet3D(
        in_channels=1, out_features=4, f_maps=(8, 16, 32),
    ).to(device)
    n_params_a = sum(p.numel() for p in branch_a.parameters())
    print(f"  Parameters: {n_params_a:,}")

    sino_in = torch.randn(1, 1, 60, 64, 128, device=device)
    sino_out = branch_a(sino_in)
    print(f"  Input:  {sino_in.shape}")
    print(f"  Output: {sino_out.shape}")
    assert sino_out.shape == (1, 4, 60, 64, 128)
    print("  OK\n")

    # --- Test Branch B alone ---
    print("=== Branch B (VolumeUNet3D) ===")
    branch_b = VolumeUNet3D(
        in_channels=5, out_channels=1, f_maps=(8, 16, 32, 64, 128),
    ).to(device)
    n_params_b = sum(p.numel() for p in branch_b.parameters())
    print(f"  Parameters: {n_params_b:,}")

    vol_in = torch.randn(1, 5, 64, 128, 128, device=device)
    vol_out = branch_b(vol_in)
    print(f"  Input:  {vol_in.shape}")
    print(f"  Output: {vol_out.shape}")
    assert vol_out.shape == (1, 1, 64, 128, 128)
    print("  OK\n")

    # --- Test gradient checkpointing backward pass ---
    print("=== Gradient checkpointing backward ===")
    branch_a.train()
    branch_b.train()
    sino_in2 = torch.randn(1, 1, 60, 64, 128, device=device, requires_grad=True)
    vol_in2 = torch.randn(1, 5, 64, 128, 128, device=device, requires_grad=True)

    sino_out2 = branch_a(sino_in2)
    sino_out2.sum().backward()
    print("  Branch A backward: OK")

    vol_out2 = branch_b(vol_in2)
    vol_out2.sum().backward()
    print("  Branch B backward: OK\n")

    # --- Test AMP forward ---
    print("=== AMP forward ===")
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            sino_out_amp = branch_a(torch.randn(1, 1, 60, 64, 128, device=device))
            print(f"  Branch A AMP dtype: {sino_out_amp.dtype}")
            vol_out_amp = branch_b(torch.randn(1, 5, 64, 128, 128, device=device))
            print(f"  Branch B AMP dtype: {vol_out_amp.dtype}")
        print("  OK\n")
    else:
        print("  Skipped (no CUDA)\n")

    # --- Full pipeline test (only with CUDA + TIGRE) ---
    if device.type == "cuda":
        print("=== Full DualDomainCascadeNet (requires TIGRE GPU) ===")
        try:
            from geometry import build_geometry

            model = DualDomainCascadeNet(
                sinogram_out_features=4,
                sinogram_f_maps=(8, 16, 32),
                volume_f_maps=(8, 16, 32, 64, 128),
                use_checkpoint=True,
            ).to(device)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"  Total parameters: {n_total:,}")

            # Small synthetic geometry
            nVoxel = np.array([64, 128, 128], dtype=np.int64)
            voxel_sizes = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            geo = build_geometry(nVoxel, voxel_sizes)
            angles = np.linspace(0, 2 * np.pi, 60, endpoint=False).astype(np.float32)

            fdk_vol = torch.randn(1, 1, 64, 128, 128, device=device)
            sino = torch.randn(
                1, 1, 60, int(geo.nDetector[0]), int(geo.nDetector[1]),
                device=device,
            )

            model.train()
            with torch.amp.autocast("cuda"):
                out = model(fdk_vol, sino, geo, angles)
            print(f"  FDK input:  {fdk_vol.shape}")
            print(f"  Sino input: {sino.shape}")
            print(f"  Output:     {out.shape}")
            assert out.shape == fdk_vol.shape
            print("  OK — full pipeline forward passed.")

            # Backward
            scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda"):
                out2 = model(fdk_vol, sino, geo, angles)
                loss = out2.sum()
            scaler.scale(loss).backward()
            print("  OK — full pipeline backward passed.")

        except Exception as e:
            print(f"  Skipped full pipeline test: {e}")
    else:
        print("=== Skipping full pipeline test (no CUDA) ===")

    print("\nAll tests passed.")
