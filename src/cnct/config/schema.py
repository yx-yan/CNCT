"""Typed configuration dataclasses for the ``cnct`` deep-learning pipeline.

All runtime parameters are expressed as frozen dataclasses. This guarantees:

    - Invalid values fail fast at construction time (via ``__post_init__``).
    - Configs are hashable and safe to pass across function boundaries.
    - ``dataclasses.asdict`` works for logging / serialisation round-trips.

The top-level :class:`TrainingCfg` and :class:`InferenceCfg` compose smaller
sub-configs (data, model, loss, optimiser, ...) so each section can be
specified in its own YAML block. Instances should be constructed via the
loaders in :mod:`cnct.config.loader`, not directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

_VALID_DEVICES = frozenset({"auto", "cpu", "cuda"})
_VALID_SPLITS = frozenset({"train", "val", "test"})
_VALID_SCHED_MODES = frozenset({"min", "max"})


@dataclass(frozen=True)
class GeometryCfg:
    """CT cone-beam geometry and unit-conversion parameters.

    This is an exact duplicate of
    :class:`cnct_dataprep.config.schema.GeometryCfg` — the two packages are
    fully decoupled and each carries its own geometry types.

    Attributes:
        dso_scale: Source-to-origin distance scale (must be > 1).
        dsd_scale: Source-to-detector distance scale (must be > 1).
        detector_col_margin: Detector column pitch multiplier (must be >= 1).
        mu_water: Linear attenuation of water at ~70 keV, in mm⁻¹.
        accuracy: ``tigre.Ax`` ray-integration step size in voxels.
    """

    dso_scale: float
    dsd_scale: float
    detector_col_margin: float
    mu_water: float
    accuracy: float

    def __post_init__(self) -> None:
        """Validate ranges of physical parameters.

        Raises:
            ValueError: If any parameter is outside a physically-sensible range.
        """
        if self.dso_scale <= 1.0:
            raise ValueError(
                f"dso_scale must be > 1.0 so the source is outside the volume; "
                f"got {self.dso_scale}"
            )
        if self.dsd_scale <= 1.0:
            raise ValueError(
                f"dsd_scale must be > 1.0; got {self.dsd_scale}"
            )
        if self.detector_col_margin < 1.0:
            raise ValueError(
                f"detector_col_margin must be >= 1.0; "
                f"got {self.detector_col_margin}"
            )
        if self.mu_water <= 0.0:
            raise ValueError(f"mu_water must be > 0; got {self.mu_water}")
        if self.accuracy <= 0.0:
            raise ValueError(f"accuracy must be > 0; got {self.accuracy}")


@dataclass(frozen=True)
class DataCfg:
    """Dataset layout and normalisation parameters.

    Attributes:
        data_dir: Directory containing ground-truth ``<CaseID>_0000.nii.gz``
            NIfTI files (in HU units).
        proj_dir: Directory containing per-case ``projections.npy`` sinograms
            (written by ``cnct_dataprep.projection``).
        fdk_dir: Directory containing per-case ``recon_fdk.npy`` FDK inputs
            (written by ``cnct_dataprep.reconstruction``).
        h5_root: Root of HDF5 split directories used for case discovery —
            only the filenames are read, not the HDF5 contents, so the same
            split that drives the single-domain pipeline is reused.
        spatial_scale: Downsample factor applied to the Y and X axes of the
            FDK and GT volumes. Z is unchanged. ``1.0`` disables downsampling.
        mu_min: Lower bound of the fixed mu normalisation range.
        mu_max: Upper bound of the fixed mu normalisation range.
    """

    data_dir: Path
    proj_dir: Path
    fdk_dir: Path
    h5_root: Path
    spatial_scale: float = 0.5
    mu_min: float = -0.02
    mu_max: float = 0.08

    def __post_init__(self) -> None:
        """Validate data layout and normalisation parameters.

        Raises:
            ValueError: If ``spatial_scale`` or the mu range is invalid.
        """
        if not 0.0 < self.spatial_scale <= 1.0:
            raise ValueError(
                f"spatial_scale must be in (0, 1]; got {self.spatial_scale}"
            )
        if self.mu_max <= self.mu_min:
            raise ValueError(
                f"mu_max ({self.mu_max}) must be > mu_min ({self.mu_min})"
            )


@dataclass(frozen=True)
class ModelCfg:
    """Dual-domain cascade model architecture.

    Attributes:
        sinogram_out_features: Number of feature channels produced by Branch A
            and backprojected by the DBP layer into volume space.
        sinogram_f_maps: Feature map sizes for each level of the Branch A
            3D U-Net, from the shallowest to the deepest level.
        volume_f_maps: Feature map sizes for each level of the Branch B
            3D U-Net.
        num_groups: ``num_groups`` passed to every :class:`GroupNorm` layer.
            Batch-size-independent normalisation; typical values are 4 or 8.
        use_checkpoint: If ``True``, enable gradient checkpointing in every
            ``ConvBlock`` in both branches (required for full-volume training
            on 48 GB GPUs).
    """

    sinogram_out_features: int = 4
    sinogram_f_maps: Tuple[int, ...] = (8, 16, 32, 64)
    volume_f_maps: Tuple[int, ...] = (8, 16, 32, 64, 128)
    num_groups: int = 8
    use_checkpoint: bool = True

    def __post_init__(self) -> None:
        """Validate architecture parameters.

        Raises:
            ValueError: If any parameter is non-positive or empty.
        """
        if self.sinogram_out_features <= 0:
            raise ValueError(
                f"sinogram_out_features must be > 0; "
                f"got {self.sinogram_out_features}"
            )
        if len(self.sinogram_f_maps) < 2:
            raise ValueError(
                f"sinogram_f_maps must have at least 2 levels; "
                f"got {self.sinogram_f_maps}"
            )
        if len(self.volume_f_maps) < 2:
            raise ValueError(
                f"volume_f_maps must have at least 2 levels; "
                f"got {self.volume_f_maps}"
            )
        if any(f <= 0 for f in self.sinogram_f_maps):
            raise ValueError(
                f"sinogram_f_maps entries must be > 0; "
                f"got {self.sinogram_f_maps}"
            )
        if any(f <= 0 for f in self.volume_f_maps):
            raise ValueError(
                f"volume_f_maps entries must be > 0; "
                f"got {self.volume_f_maps}"
            )
        if self.num_groups <= 0:
            raise ValueError(
                f"num_groups must be > 0; got {self.num_groups}"
            )


@dataclass(frozen=True)
class LossCfg:
    """Hybrid loss weights (L1 + 3D SSIM + 3D edge).

    The total loss is::

        L = alpha * L1 + beta * (1 - SSIM_3D) + gamma * L_edge_3D

    Attributes:
        alpha: L1 weight.
        beta: SSIM weight.
        gamma: Edge weight.
        ssim_window: Window side length for 3D SSIM (must be odd).
    """

    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 1.0
    ssim_window: int = 7

    def __post_init__(self) -> None:
        """Validate loss weights and SSIM window.

        Raises:
            ValueError: If a weight is negative or the window is invalid.
        """
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError(
                f"Loss weights must be non-negative; got "
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
            )
        if self.ssim_window < 3 or self.ssim_window % 2 == 0:
            raise ValueError(
                f"ssim_window must be an odd integer >= 3; "
                f"got {self.ssim_window}"
            )


@dataclass(frozen=True)
class OptimizerCfg:
    """AdamW optimiser hyperparameters.

    Attributes:
        lr: Initial learning rate.
        weight_decay: L2 regularisation factor.
    """

    lr: float = 2e-4
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        """Validate optimiser parameters.

        Raises:
            ValueError: If ``lr`` is non-positive or ``weight_decay`` is negative.
        """
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0; got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0; got {self.weight_decay}"
            )


@dataclass(frozen=True)
class SchedulerCfg:
    """``ReduceLROnPlateau`` scheduler configuration.

    Attributes:
        mode: ``"max"`` to monitor a metric that should increase (PSNR),
            ``"min"`` for a metric that should decrease.
        factor: Multiplicative LR reduction applied on plateau.
        patience: Number of epochs with no improvement before LR is reduced.
    """

    mode: str = "max"
    factor: float = 0.5
    patience: int = 10

    def __post_init__(self) -> None:
        """Validate scheduler parameters.

        Raises:
            ValueError: If any field is out of range.
        """
        if self.mode not in _VALID_SCHED_MODES:
            raise ValueError(
                f"scheduler mode must be one of {sorted(_VALID_SCHED_MODES)}; "
                f"got {self.mode!r}"
            )
        if not 0.0 < self.factor < 1.0:
            raise ValueError(
                f"scheduler factor must be in (0, 1); got {self.factor}"
            )
        if self.patience < 0:
            raise ValueError(
                f"scheduler patience must be >= 0; got {self.patience}"
            )


@dataclass(frozen=True, kw_only=True)
class TrainingCfg:
    """Top-level training configuration composing every sub-config.

    Attributes:
        geometry: Cone-beam geometry settings (required at DBP construction).
        data: Dataset layout and normalisation parameters.
        model: Dual-domain cascade architecture.
        loss: Hybrid loss weights.
        optimizer: AdamW hyperparameters.
        scheduler: ``ReduceLROnPlateau`` settings.
        epochs: Maximum number of epochs.
        seed: RNG seed for ``random``, ``numpy``, and ``torch``.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        checkpoint_dir: Directory for ``best_checkpoint.pytorch`` /
            ``last_checkpoint.pytorch``.
        resume: Path to a checkpoint to resume from; ``None`` trains from
            scratch.
        log_interval: Log a line every ``log_interval`` training steps.
        amp: If ``True``, enable torch AMP autocast + ``GradScaler``.
    """

    geometry: GeometryCfg
    data: DataCfg
    model: ModelCfg
    loss: LossCfg
    optimizer: OptimizerCfg
    scheduler: SchedulerCfg
    checkpoint_dir: Path
    epochs: int = 200
    seed: int = 42
    device: str = "auto"
    resume: Optional[Path] = None
    log_interval: int = 5
    amp: bool = True

    def __post_init__(self) -> None:
        """Validate training-loop parameters.

        Raises:
            ValueError: If any field is inconsistent.
        """
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0; got {self.epochs}")
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_VALID_DEVICES)}; "
                f"got {self.device!r}"
            )
        if self.log_interval <= 0:
            raise ValueError(
                f"log_interval must be > 0; got {self.log_interval}"
            )


@dataclass(frozen=True, kw_only=True)
class InferenceCfg:
    """Top-level inference configuration for the dual-domain cascade.

    Attributes:
        geometry: Cone-beam geometry settings (needed by the DBP layer).
        data: Dataset layout and normalisation parameters.
        model: Dual-domain cascade architecture (must match the checkpoint).
        checkpoint: Path to a trained ``best_checkpoint.pytorch``.
        output_dir: Directory for per-case ``<case_id>_recon.npy`` outputs.
        split: Which HDF5 split to run inference on (``"train"`` / ``"val"``
            / ``"test"``).
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        amp: If ``True``, wrap the forward pass in torch AMP autocast.
    """

    geometry: GeometryCfg
    data: DataCfg
    model: ModelCfg
    checkpoint: Path
    output_dir: Path
    split: str = "test"
    device: str = "auto"
    amp: bool = True

    def __post_init__(self) -> None:
        """Validate inference parameters.

        Raises:
            ValueError: If any field is inconsistent.
        """
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}; "
                f"got {self.split!r}"
            )
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_VALID_DEVICES)}; "
                f"got {self.device!r}"
            )
