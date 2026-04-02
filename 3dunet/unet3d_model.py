"""
Custom 3D U-Net for Sparse-View CT Reconstruction
==================================================

Purpose
-------
This module defines a 3D U-Net that enhances coarse FDK (Feldkamp-Davis-Kress)
reconstructions from sparse-view cone-beam CT. Rather than directly predicting
the clean image, it uses **residual learning**: the network predicts the
*artifact component* (streaks caused by angular undersampling), and the final
output is computed as:

    Clean_Image = Input_FDK - Predicted_Artifacts

This residual formulation works better than direct mapping because:
  1. The identity mapping (input → output unchanged) is the trivial solution,
     so the network only needs to learn the *difference* — a much simpler
     function that is easier to optimise.
  2. Gradients flow more directly back to early layers (similar benefit to
     ResNets), which stabilises training for deep architectures.
  3. Sharp anatomical edges already present in the FDK input are preserved
     by default; the network only needs to remove the overlaid artifacts.

Architecture overview (default configuration)
----------------------------------------------
The network follows the classic U-Net encoder–bottleneck–decoder structure,
operating entirely in 3D (Conv3d, GroupNorm, MaxPool3d, ConvTranspose3d).

    Encoder: 4 stages at feature widths [16, 32, 64, 128]
    Bottleneck: 1 stage at feature width 256  (lowest resolution)
    Decoder: 4 stages at feature widths [128, 64, 32, 16]
    Final:   1×1×1 convolution mapping 16 features → 1 output channel

The bottleneck contains placeholder markers for a future Vision Graph Guiding
Transformer (VGGT) module that would inject global context at the coarsest
spatial resolution — where the receptive field is largest and long-range
dependencies (e.g., bilateral symmetry of the abdomen) can be captured most
efficiently.

Tensor conventions
------------------
All tensors throughout this network are 5-dimensional:

    [B, C, D, H, W]
     │  │  │  │  └─ Width   (X spatial axis — left/right)
     │  │  │  └──── Height  (Y spatial axis — anterior/posterior)
     │  │  └─────── Depth   (Z spatial axis — superior/inferior, i.e. slices)
     │  └────────── Channels (feature maps; 1 for input/output grayscale)
     └───────────── Batch    (number of patches processed in parallel)

The input patch shape used during training is [1, 1, 64, 128, 128] (set in
train_config.yaml), chosen so that the minimum Z dimension across the dataset
(~61 slices) can still produce at least one valid patch.

No sigmoid or softmax is applied — the output is a raw regression value in the
same physical units as the input (linear attenuation coefficients, μ, mm⁻¹).

Compatibility
-------------
This model matches the pytorch-3dunet library config:
    in_channels=1, out_channels=1, f_maps=[16, 32, 64, 128, 256]

It produces ~5.6M trainable parameters with the default feature map sizes.
"""

import torch
import torch.nn as nn


# =============================================================================
# Building Block: Double Convolution
# =============================================================================


class ConvBlock(nn.Module):
    """Two consecutive 3D convolution layers, each followed by group
    normalisation and ReLU activation.

    This is the fundamental feature-extraction unit used at every level of
    both the encoder and decoder. Two convolutions per level (rather than one)
    give each stage enough capacity to learn meaningful features before the
    spatial resolution changes.

    Layer sequence (repeated twice):
        Conv3d → GroupNorm → ReLU

    Why GroupNorm instead of BatchNorm?
    -----------------------------------
    BatchNorm3d computes mean and variance across the **batch dimension** for
    each channel. With very small batch sizes (1–2 patches, typical for 3D
    medical imaging due to GPU memory limits), these per-batch statistics are
    extremely noisy, leading to unstable training and poor generalisation.

    GroupNorm instead divides the channels into fixed groups and normalises
    within each group **per individual sample** — it never looks at other
    samples in the batch. This makes it completely independent of batch size,
    providing stable normalisation even with batch_size=1.

    With num_groups=8 (default) and feature maps [16, 32, 64, 128, 256]:
      - 16 channels  / 8 groups =  2 channels per group
      - 32 channels  / 8 groups =  4 channels per group
      - 64 channels  / 8 groups =  8 channels per group
      - 128 channels / 8 groups = 16 channels per group
      - 256 channels / 8 groups = 32 channels per group
    The requirement is that num_channels must be divisible by num_groups.

    This also aligns with the pytorch-3dunet library, which uses GroupNorm
    by default (layer_order: 'gcr' = GroupNorm → Conv → ReLU).

    Other design choices:
        - kernel_size=3, padding=1: preserves spatial dimensions (D, H, W
          remain unchanged), so we can concatenate encoder and decoder feature
          maps without cropping.
        - bias=False: GroupNorm already includes a learnable bias (the β
          affine parameter), so the Conv3d bias would be redundant and waste
          memory.
        - inplace=True on ReLU: overwrites the input tensor in-place to
          halve peak memory usage — important for 3D volumes which are
          memory-intensive.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps (channels).
    out_channels : int
        Number of output feature maps (channels).
    num_groups : int
        Number of groups for GroupNorm. Must evenly divide both in_channels
        (if in_channels == out_channels) and out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()

        self.block = nn.Sequential(
            # --- First convolution: change channel depth from in → out ---
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=3,  # 3×3×3 kernel — standard for volumetric data
                padding=1,     # "same" padding: output D,H,W == input D,H,W
                bias=False,    # redundant when followed by GroupNorm (has affine β)
            ),
            # GroupNorm: divide out_channels into num_groups groups, then
            # normalise (mean=0, var=1) within each group independently per
            # sample. Unlike BatchNorm, this does NOT depend on batch size.
            # The affine=True default adds learnable scale (γ) and shift (β)
            # parameters per channel, restoring representational capacity.
            nn.GroupNorm(num_groups, out_channels),
            # ReLU introduces non-linearity — without it, stacking linear
            # convolutions would collapse to a single linear operation.
            nn.ReLU(inplace=True),

            # --- Second convolution: refine features at the same depth ---
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, in_channels, D, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, out_channels, D, H, W].
            Spatial dimensions are preserved (same D, H, W as input).
        """
        return self.block(x)


# =============================================================================
# Encoder (Contracting Path)
# =============================================================================


class Encoder(nn.Module):
    """The contracting (downsampling) path of the U-Net.

    Each encoder stage:
      1. Applies a ConvBlock to extract features at the current resolution.
      2. Stores the output as a skip connection (for the decoder to use later).
      3. Applies 2×2×2 max-pooling to halve all spatial dimensions.

    With the default f_maps=[16, 32, 64, 128], the data flow through the
    encoder looks like this (for a [1, 1, 64, 128, 128] input patch):

        Input:  [B, 1,   64, 128, 128]
        Stage 0 ConvBlock → [B, 16,  64, 128, 128]  ← skip_0
        Pool    → [B, 16,  32,  64,  64]
        Stage 1 ConvBlock → [B, 32,  32,  64,  64]  ← skip_1
        Pool    → [B, 32,  16,  32,  32]
        Stage 2 ConvBlock → [B, 64,  16,  32,  32]  ← skip_2
        Pool    → [B, 64,   8,  16,  16]
        Stage 3 ConvBlock → [B, 128,  8,  16,  16]  ← skip_3
        Pool    → [B, 128,  4,   8,   8]   ← sent to bottleneck

    The skip connections carry high-resolution spatial detail that would
    otherwise be lost during downsampling. The decoder later concatenates
    these with its upsampled features to recover fine-grained structures
    (e.g., organ boundaries and small vessels).

    Parameters
    ----------
    in_channels : int
        Number of channels in the very first input (1 for grayscale CT).
    feature_maps : list[int]
        Number of feature maps at each encoder stage. Length determines
        the number of downsampling levels.
    num_groups : int
        Number of groups for GroupNorm in each ConvBlock.
    """

    def __init__(self, in_channels: int, feature_maps: list[int], num_groups: int = 8):
        super().__init__()

        # ModuleLists let PyTorch track these layers for .parameters(),
        # .to(device), .train()/.eval(), state_dict, etc.
        self.stages = nn.ModuleList()  # ConvBlock at each resolution level
        self.pools = nn.ModuleList()   # MaxPool3d after each ConvBlock

        # Build one ConvBlock + MaxPool pair per feature map entry.
        # The channel count threads through: in→f_maps[0]→f_maps[1]→...
        current_channels = in_channels
        for num_features in feature_maps:
            self.stages.append(ConvBlock(current_channels, num_features, num_groups))
            # 2×2×2 max-pooling halves each spatial dimension (D, H, W).
            # This progressively reduces resolution and increases the
            # effective receptive field of deeper layers.
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = num_features  # output channels become next input

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the input through all encoder stages.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, in_channels, D, H, W].

        Returns
        -------
        x : torch.Tensor
            Downsampled feature map after the final pool, ready for the
            bottleneck. Shape: [B, f_maps[-1], D/2^N, H/2^N, W/2^N].
        skip_connections : list[torch.Tensor]
            Feature maps from each stage *before* pooling, in order from
            highest resolution (stage 0) to lowest (stage N-1). The decoder
            will consume these in reverse order.
        """
        skip_connections = []

        for stage, pool in zip(self.stages, self.pools):
            # Extract features at this resolution level
            x = stage(x)
            # Save the full-resolution features for the decoder skip connection.
            # This is the defining characteristic of U-Net — without these,
            # the decoder would have to hallucinate all spatial detail.
            skip_connections.append(x)
            # Downsample: halve D, H, W — doubles the effective receptive field
            x = pool(x)

        return x, skip_connections


# =============================================================================
# Bottleneck (Bridge between Encoder and Decoder)
# =============================================================================


class Bottleneck(nn.Module):
    """The deepest layer of the U-Net, connecting encoder to decoder.

    At this point the feature maps have the *lowest spatial resolution* but
    the *highest channel count* (256 by default). This is where:
      - The receptive field is largest, so each spatial position "sees" the
        widest context of the original input volume.
      - Global patterns like bilateral symmetry, overall body contour, and
        large-scale streak artifact patterns are most naturally represented.

    VGGT integration point
    ----------------------
    The placeholder markers below indicate where a Vision Graph Guiding
    Transformer (VGGT) module would be inserted in the future. The bottleneck
    is the optimal location because:
      1. The small spatial dimensions (e.g. 4×8×8) make self-attention
         computationally feasible (cost is quadratic in token count).
      2. Global context is most valuable here — the encoder has already
         extracted local features, and the decoder will recover spatial detail
         from skip connections.
      3. The VGGT would model long-range spatial dependencies (e.g., artifact
         streaks spanning the full volume) that local convolutions miss.

    Parameters
    ----------
    in_channels : int
        Number of input channels (from the last encoder pool).
    out_channels : int
        Number of output channels (fed to the first decoder upconv).
    num_groups : int
        Number of groups for GroupNorm in the ConvBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()

        ### [INSERT VGGT MODULE HERE IN THE FUTURE] ###
        # Replace or wrap self.block with a Vision Graph Guiding Transformer
        # to inject global context at the lowest-resolution feature map.
        # The VGGT would sit between the encoder output and the ConvBlock,
        # or replace the ConvBlock entirely, depending on the design.

        self.block = ConvBlock(in_channels, out_channels, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the lowest-resolution feature map.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, in_channels, D_min, H_min, W_min] — the most
            downsampled representation (e.g. [B, 128, 4, 8, 8]).

        Returns
        -------
        torch.Tensor
            Shape [B, out_channels, D_min, H_min, W_min] — same spatial
            dimensions, but channel count changes (e.g. 128 → 256).
        """
        ### [INSERT VGGT MODULE HERE IN THE FUTURE] ###
        return self.block(x)


# =============================================================================
# Decoder (Expanding Path)
# =============================================================================


class Decoder(nn.Module):
    """The expanding (upsampling) path of the U-Net.

    Each decoder stage:
      1. Upsamples the feature map by 2× using transposed convolution,
         doubling all spatial dimensions (D, H, W).
      2. Concatenates the upsampled features with the corresponding encoder
         skip connection along the channel dimension.
      3. Applies a ConvBlock to fuse the combined information.

    The skip connections are consumed in *reverse* order — the deepest encoder
    skip (lowest resolution) is used first, and the shallowest (highest
    resolution) is used last. This mirrors the U-shape of the architecture.

    With the default f_maps=[16, 32, 64, 128] and bottleneck=256, the data
    flow through the decoder looks like this:

        From bottleneck:  [B, 256,  4,   8,   8]
        UpConv → [B, 128,  8,  16,  16]  + skip_3 [B, 128, 8, 16, 16]
        Cat    → [B, 256,  8,  16,  16]  → ConvBlock → [B, 128,  8,  16,  16]
        UpConv → [B,  64, 16,  32,  32]  + skip_2 [B, 64, 16, 32, 32]
        Cat    → [B, 128, 16,  32,  32]  → ConvBlock → [B,  64, 16,  32,  32]
        UpConv → [B,  32, 32,  64,  64]  + skip_1 [B, 32, 32, 64, 64]
        Cat    → [B,  64, 32,  64,  64]  → ConvBlock → [B,  32, 32,  64,  64]
        UpConv → [B,  16, 64, 128, 128]  + skip_0 [B, 16, 64, 128, 128]
        Cat    → [B,  32, 64, 128, 128]  → ConvBlock → [B,  16, 64, 128, 128]

    The output has the same spatial dimensions as the original input, with
    16 feature channels that the final 1×1×1 convolution maps to 1 channel.

    Parameters
    ----------
    feature_maps : list[int]
        Encoder feature map sizes in original order [16, 32, 64, 128].
        Will be reversed internally for the upward path.
    bottleneck_channels : int
        Number of channels coming out of the bottleneck (256 by default).
    num_groups : int
        Number of groups for GroupNorm in each ConvBlock.
    """

    def __init__(self, feature_maps: list[int], bottleneck_channels: int, num_groups: int = 8):
        super().__init__()

        self.upconvs = nn.ModuleList()  # Transposed convolutions for upsampling
        self.stages = nn.ModuleList()   # ConvBlocks for feature fusion

        # Reverse the encoder feature maps to build the decoder bottom-up:
        # [16, 32, 64, 128] → [128, 64, 32, 16]
        reversed_maps = list(reversed(feature_maps))

        current_channels = bottleneck_channels  # start from bottleneck output (256)
        for num_features in reversed_maps:
            # Transposed convolution: learnable upsampling that doubles spatial
            # dimensions (kernel=2, stride=2). Also reduces channel count from
            # current_channels → num_features (e.g. 256 → 128).
            # This is preferred over bilinear interpolation because the learned
            # kernel can adapt to the data distribution.
            self.upconvs.append(
                nn.ConvTranspose3d(
                    current_channels, num_features,
                    kernel_size=2,  # exactly 2× upsampling
                    stride=2,
                )
            )
            # After concatenation with the skip connection along dim=1:
            #   upsampled channels (num_features) + skip channels (num_features)
            #   = 2 × num_features
            # The ConvBlock fuses these back down to num_features.
            self.stages.append(ConvBlock(num_features * 2, num_features, num_groups))

            current_channels = num_features  # next stage takes this as input

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: list[torch.Tensor],
    ) -> torch.Tensor:
        """Upsample through all decoder stages, fusing with skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Bottleneck output, shape [B, bottleneck_channels, D_min, H_min, W_min].
        skip_connections : list[torch.Tensor]
            Encoder skip connections ordered from highest resolution (index 0)
            to lowest resolution (index -1). This method iterates them in
            reverse to match the bottom-up decoder path.

        Returns
        -------
        torch.Tensor
            Decoded feature map at the original input resolution,
            shape [B, f_maps[0], D, H, W] (e.g. [B, 16, 64, 128, 128]).
        """
        # reversed() yields skip connections from deepest to shallowest,
        # matching the decoder's bottom-up traversal order.
        for upconv, stage, skip in zip(
            self.upconvs, self.stages, reversed(skip_connections)
        ):
            # Step 1: Upsample — double spatial dimensions, reduce channels.
            # e.g. [B, 256, 4, 8, 8] → [B, 128, 8, 16, 16]
            x = upconv(x)

            # Step 2: Handle spatial dimension mismatches.
            # When any input dimension is odd (e.g. D=65), MaxPool3d floors
            # the result (65→32), but ConvTranspose3d doubles exactly (32→64),
            # leaving us one voxel short (64 vs 65). We pad the upsampled
            # tensor with zeros on the right/bottom/back to match the skip
            # connection's spatial size.
            if x.shape != skip.shape:
                x = nn.functional.pad(
                    x,
                    [
                        # F.pad takes pairs in reverse spatial order: W, H, D
                        # Each pair is (pad_left, pad_right) / (pad_top, pad_bottom) etc.
                        0, skip.shape[4] - x.shape[4],  # Width  padding (dim 4)
                        0, skip.shape[3] - x.shape[3],  # Height padding (dim 3)
                        0, skip.shape[2] - x.shape[2],  # Depth  padding (dim 2)
                    ],
                )

            # Step 3: Concatenate skip connection along the channel dimension.
            # The skip carries fine-grained spatial detail from the encoder
            # that was lost during downsampling. Concatenation (rather than
            # addition) lets the ConvBlock learn how much to weight each source.
            # Shape: [B, num_features, D, H, W] + [B, num_features, D, H, W]
            #      → [B, 2*num_features, D, H, W]
            x = torch.cat([skip, x], dim=1)

            # Step 4: Fuse the concatenated features with the ConvBlock.
            # Reduces 2*num_features back to num_features while integrating
            # the encoder's local detail with the decoder's global context.
            x = stage(x)

        return x


# =============================================================================
# Full Model: Residual 3D U-Net
# =============================================================================


class ResidualUNet3D(nn.Module):
    """3D U-Net with residual (artifact-predicting) learning for CT
    reconstruction enhancement.

    Instead of learning a direct mapping from noisy input to clean output,
    this network learns to predict the **artifact component**:

        Predicted_Artifacts = Network(Input_FDK)
        Clean_Image = Input_FDK - Predicted_Artifacts

    This residual formulation has three key advantages:
      1. **Easier optimisation**: the identity mapping is the default (zero
         artifact prediction = input passes through unchanged), so the
         network only needs to learn the residual difference.
      2. **Better gradient flow**: the skip connection from input to output
         provides a direct gradient path, reducing vanishing gradient issues.
      3. **Edge preservation**: anatomical structures already present in the
         FDK input are preserved by default; only the artifact overlay needs
         to be learned.

    The output layer has **no activation function** — this is a pure regression
    network outputting continuous values (linear attenuation coefficients, μ).
    Sigmoid/softmax would clamp the output to [0,1], which is wrong for
    physical μ values.

    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels (1 for single-energy CT).
    out_channels : int, default=1
        Number of output channels. Must equal in_channels for the residual
        subtraction to be dimensionally valid.
    f_maps : tuple[int, ...], default=(16, 32, 64, 128, 256)
        Feature map counts at each level. The first N-1 entries define the
        encoder stages; the last entry defines the bottleneck depth.
        Deeper/wider maps increase capacity but also memory and compute.
    num_groups : int, default=8
        Number of groups for GroupNorm in every ConvBlock. Must evenly
        divide every entry in f_maps. With the default f_maps, 8 divides
        all of [16, 32, 64, 128, 256] cleanly.

    Example
    -------
    >>> model = ResidualUNet3D()
    >>> # Input: one grayscale 3D patch [batch=1, channels=1, D=64, H=128, W=128]
    >>> fdk_input = torch.randn(1, 1, 64, 128, 128)
    >>> clean_output = model(fdk_input)
    >>> assert clean_output.shape == fdk_input.shape  # [1, 1, 64, 128, 128]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        f_maps: tuple[int, ...] = (16, 32, 64, 128, 256),
        num_groups: int = 8,
    ):
        super().__init__()

        # Residual subtraction requires matching channel counts: if the input
        # has C channels, the predicted artifact map must also have C channels
        # so element-wise subtraction (input - artifacts) is valid.
        assert in_channels == out_channels, (
            "Residual learning requires in_channels == out_channels "
            f"(got {in_channels} and {out_channels})"
        )

        # Validate that num_groups evenly divides every feature map size.
        # GroupNorm splits channels into groups — uneven division is an error.
        for f in f_maps:
            assert f % num_groups == 0, (
                f"num_groups={num_groups} does not evenly divide f_maps "
                f"entry {f}. Every feature map size must be divisible by "
                f"num_groups for GroupNorm to work."
            )

        # Split the feature map list into encoder levels and bottleneck.
        # Example with f_maps=[16, 32, 64, 128, 256]:
        #   encoder_maps   = [16, 32, 64, 128]   (4 encoder stages)
        #   bottleneck_in  = 128                  (last encoder stage output)
        #   bottleneck_out = 256                  (bottleneck feature depth)
        encoder_maps = list(f_maps[:-1])
        bottleneck_in = f_maps[-2]
        bottleneck_out = f_maps[-1]

        # --- Encoder ---
        # Progressively downsamples spatial dimensions while increasing
        # feature channels: [1]→[16]→[32]→[64]→[128]
        self.encoder = Encoder(in_channels, encoder_maps, num_groups)

        # --- Bottleneck ---
        # Processes the most compressed representation: [128]→[256]
        # Future VGGT module would be integrated here.
        self.bottleneck = Bottleneck(bottleneck_in, bottleneck_out, num_groups)

        # --- Decoder ---
        # Progressively upsamples back to original resolution while
        # decreasing feature channels: [256]→[128]→[64]→[32]→[16]
        self.decoder = Decoder(encoder_maps, bottleneck_out, num_groups)

        # --- Final 1×1×1 convolution ---
        # Maps the decoder's feature channels to the output channel count.
        # kernel_size=1 acts as a per-voxel linear combination of features.
        # NO activation function: the output is a raw regression prediction
        # of artifact intensity (can be positive or negative).
        self.final_conv = nn.Conv3d(
            encoder_maps[0],  # first encoder level = last decoder level (16)
            out_channels,     # 1 output channel
            kernel_size=1,    # pointwise convolution — no spatial mixing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict artifacts and subtract from input.

        The complete data flow for a [1, 1, 64, 128, 128] input patch:

            Input x ──────────────────────────────────┐ (residual shortcut)
              │                                        │
              ▼                                        │
            Encoder                                    │
              │  [1,  1, 64,128,128] → ... →           │
              │  [1,128,  4,  8,  8] (+ 4 skip conns)  │
              ▼                                        │
            Bottleneck                                 │
              │  [1,128,  4,  8,  8] → [1,256, 4,8,8]  │
              ▼                                        │
            Decoder (fuses skip connections)            │
              │  [1,256,  4,  8,  8] → ... →           │
              │  [1, 16, 64,128,128]                   │
              ▼                                        │
            Final Conv (1×1×1, no activation)          │
              │  [1, 16, 64,128,128] → [1,1,64,128,128]│
              ▼                                        │
            artifacts                                  │
              │                                        │
              ▼                                        ▼
            output = residual - artifacts
              │
              ▼
            [1, 1, 64, 128, 128]  (enhanced CT patch)

        Parameters
        ----------
        x : torch.Tensor
            Coarse FDK reconstruction patch.
            Shape: [B, in_channels, D, H, W].

        Returns
        -------
        torch.Tensor
            Enhanced (artifact-reduced) patch.
            Shape: [B, out_channels, D, H, W] (same as input).
        """
        # Save the original input for the residual subtraction at the end.
        # This is the "long skip connection" that defines residual learning.
        residual = x

        # --- Encoder: extract multi-scale features ---
        # x is progressively downsampled; skips hold full-resolution copies
        # at each level for the decoder to use later.
        x, skips = self.encoder(x)

        # --- Bottleneck: process the most compressed representation ---
        # This is where global context is captured. The spatial dimensions
        # are at their smallest (e.g. 4×8×8), so each "voxel" in the feature
        # map has an effective receptive field covering the entire input patch.
        x = self.bottleneck(x)

        # --- Decoder: progressively upsample and fuse with skip connections ---
        # Recovers spatial detail by combining the bottleneck's global context
        # with the encoder's preserved local features.
        x = self.decoder(x, skips)

        # --- Final convolution: map features to artifact prediction ---
        # No activation — the predicted artifact can be positive (bright
        # streaks) or negative (dark streaks), so we need the full real range.
        artifacts = self.final_conv(x)

        # --- Residual learning: subtract predicted artifacts ---
        # Clean_Image = Input_FDK - Predicted_Artifacts
        # If the network predicts zero everywhere, the input passes through
        # unchanged — this is the safe default the network starts near.
        return residual - artifacts


# =============================================================================
# Sanity Check (run with: python 3dunet/unet3d_model.py)
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, out_channels=1).to(device)

    # Count trainable parameters (~5.6M with default f_maps)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test with a patch matching train_config.yaml: [B=1, C=1, D=64, H=128, W=128]
    x = torch.randn(1, 1, 64, 128, 128, device=device)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"
    print("OK — input and output shapes match.")