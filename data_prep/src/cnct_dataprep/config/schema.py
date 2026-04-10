"""Typed configuration dataclasses for the cnct_dataprep pipeline.

All runtime parameters are expressed as frozen dataclasses. This guarantees:

    - Invalid values fail fast at construction time (via ``__post_init__``).
    - Configs are hashable and safe to pass across function boundaries.
    - ``dataclasses.asdict`` works for logging / serialisation round-trips.

Instances should be constructed via the loaders in
:mod:`cnct_dataprep.config.loader`, not directly — the loaders translate YAML
files to these dataclasses with proper path coercion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_VALID_FDK_FILTERS = frozenset(
    {"ram_lak", "shepp_logan", "cosine", "hamming", "hann"}
)


@dataclass(frozen=True)
class GeometryCfg:
    """CT cone-beam geometry and unit-conversion parameters.

    Attributes:
        dso_scale: Source-to-origin distance scale.
            ``DSO = max_radius * dso_scale`` where ``max_radius`` is the
            half-diagonal of the physical volume cross-section. Must be greater
            than 1 to ensure the source is outside every volume.
        dsd_scale: Source-to-detector distance scale.
            ``DSD = DSO * dsd_scale``. Controls magnification; must be greater
            than 1.
        detector_col_margin: Multiplier on the detector column pitch. Values
            greater than 1 pad the detector laterally to prevent projection
            truncation artifacts for off-centre voxels.
        n_angles: Number of projection angles distributed evenly over 360°.
            Shared across all stages; also used to resolve ``{n_angles}``
            placeholders in directory paths.
        mu_water: Linear attenuation of water at ~70 keV, in mm⁻¹. Used for the
            HU ↔ mu conversion formula ``mu = (HU + 1000) / 1000 * mu_water``.
        accuracy: ``tigre.Ax`` ray-integration step size in voxels. Lower values
            give more accurate forward projections at the cost of runtime.
    """

    dso_scale: float
    dsd_scale: float
    detector_col_margin: float
    n_angles: int
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
                f"detector_col_margin must be >= 1.0; got {self.detector_col_margin}"
            )
        if self.n_angles <= 0:
            raise ValueError(f"n_angles must be > 0; got {self.n_angles}")
        if self.mu_water <= 0.0:
            raise ValueError(f"mu_water must be > 0; got {self.mu_water}")
        if self.accuracy <= 0.0:
            raise ValueError(f"accuracy must be > 0; got {self.accuracy}")


@dataclass(frozen=True)
class ProjectionCfg:
    """Forward-projection pipeline configuration.

    Attributes:
        geometry: Cone-beam geometry settings (typically loaded from
            ``geometry.yaml`` and shared with :class:`FdkCfg`).
        data_dir: Directory containing ``<CaseID>_0000.nii.gz`` NIfTI files.
        proj_dir: Output directory for per-case ``projections.npy`` files.
        n_angles: Number of projection angles distributed evenly over 360°.
        case_start: Inclusive start index into the sorted case list.
        case_end: Exclusive end index into the sorted case list. ``None`` means
            "through the last case".
        max_cases: Hard cap on the number of cases after ``case_start`` /
            ``case_end`` slicing. ``None`` means no cap.
        save_png: If ``True``, save per-angle PNG visualisations alongside
            ``projections.npy``.
        proj_save_every: When ``save_png`` is ``True``, save every Nth projection
            angle as PNG.
        image_dpi: DPI for all saved PNGs.
    """

    geometry: GeometryCfg
    data_dir: Path
    proj_dir: Path
    n_angles: int
    case_start: int = 0
    case_end: Optional[int] = None
    max_cases: Optional[int] = None
    save_png: bool = True
    proj_save_every: int = 20
    image_dpi: int = 150

    def __post_init__(self) -> None:
        """Validate field values.

        Raises:
            ValueError: If any field is inconsistent.
        """
        if self.n_angles <= 0:
            raise ValueError(f"n_angles must be > 0; got {self.n_angles}")
        if self.case_start < 0:
            raise ValueError(
                f"case_start must be >= 0; got {self.case_start}"
            )
        if self.case_end is not None and self.case_end <= self.case_start:
            raise ValueError(
                f"case_end ({self.case_end}) must be > case_start "
                f"({self.case_start})"
            )
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError(
                f"max_cases must be > 0 or None; got {self.max_cases}"
            )
        if self.proj_save_every <= 0:
            raise ValueError(
                f"proj_save_every must be > 0; got {self.proj_save_every}"
            )
        if self.image_dpi <= 0:
            raise ValueError(f"image_dpi must be > 0; got {self.image_dpi}")


@dataclass(frozen=True)
class EvaluationCfg:
    """FDK evaluation pipeline configuration.

    The evaluation pipeline compares per-case FDK reconstructions against the
    original ground-truth NIfTI volumes and writes PSNR/SSIM metrics to a CSV.

    Attributes:
        geometry: Geometry configuration. Only ``mu_water`` is actually used
            by the evaluation pipeline (for HU → mu conversion of the ground
            truth), but we reuse the same dataclass for consistency with the
            projection and FDK configs.
        data_dir: Directory containing ground-truth ``<CaseID>_0000.nii.gz``
            NIfTI files (in HU units).
        fdk_dir: Input directory with per-case ``recon_fdk.npy`` files.
        eval_dir: Output directory for the CSV summary and comparison PNGs.
        case_start: Inclusive start index into the sorted case list.
        case_end: Exclusive end index; ``None`` = through the last case.
        max_cases: Hard cap on cases after slicing; ``None`` = no cap.
        save_png: If ``True``, save per-case side-by-side comparison PNGs
            under ``<eval_dir>/<case_id>/``.
        image_dpi: DPI for all saved PNGs.
    """

    geometry: GeometryCfg
    data_dir: Path
    fdk_dir: Path
    eval_dir: Path
    case_start: int = 0
    case_end: Optional[int] = None
    max_cases: Optional[int] = None
    save_png: bool = True
    image_dpi: int = 150

    def __post_init__(self) -> None:
        """Validate field values.

        Raises:
            ValueError: If any field is inconsistent.
        """
        if self.case_start < 0:
            raise ValueError(
                f"case_start must be >= 0; got {self.case_start}"
            )
        if self.case_end is not None and self.case_end <= self.case_start:
            raise ValueError(
                f"case_end ({self.case_end}) must be > case_start "
                f"({self.case_start})"
            )
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError(
                f"max_cases must be > 0 or None; got {self.max_cases}"
            )
        if self.image_dpi <= 0:
            raise ValueError(f"image_dpi must be > 0; got {self.image_dpi}")


@dataclass(frozen=True)
class FdkCfg:
    """FDK reconstruction pipeline configuration.

    Attributes:
        geometry: Cone-beam geometry settings. Must match the geometry used for
            forward projection or reconstructions will be mis-registered.
        data_dir: Directory containing ``<CaseID>_0000.nii.gz`` NIfTI files.
            Required so that geometry can be rebuilt per case from the header
            (``nVoxel`` / ``dVoxel`` vary across cases).
        proj_dir: Input directory with per-case ``projections.npy`` files.
        fdk_dir: Output directory for per-case ``recon_fdk.npy`` (and optional
            ``.nii.gz`` / PNGs).
        filter_name: FDK backprojection filter. One of ``"ram_lak"``,
            ``"shepp_logan"``, ``"cosine"``, ``"hamming"``, ``"hann"``.
        case_start: Inclusive start index into the sorted case list.
        case_end: Exclusive end index; ``None`` = through the last case.
        max_cases: Hard cap on cases after slicing; ``None`` = no cap.
        save_png: If ``True``, save reconstruction slice PNGs.
        save_nii: If ``True``, also save ``recon_fdk.nii.gz``.
        image_dpi: DPI for all saved PNGs.
    """

    geometry: GeometryCfg
    data_dir: Path
    proj_dir: Path
    fdk_dir: Path
    filter_name: str = "ram_lak"
    case_start: int = 0
    case_end: Optional[int] = None
    max_cases: Optional[int] = None
    save_png: bool = True
    save_nii: bool = False
    image_dpi: int = 150

    def __post_init__(self) -> None:
        """Validate field values.

        Raises:
            ValueError: If any field is inconsistent.
        """
        if self.filter_name not in _VALID_FDK_FILTERS:
            raise ValueError(
                f"filter_name must be one of {sorted(_VALID_FDK_FILTERS)}; "
                f"got {self.filter_name!r}"
            )
        if self.case_start < 0:
            raise ValueError(
                f"case_start must be >= 0; got {self.case_start}"
            )
        if self.case_end is not None and self.case_end <= self.case_start:
            raise ValueError(
                f"case_end ({self.case_end}) must be > case_start "
                f"({self.case_start})"
            )
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError(
                f"max_cases must be > 0 or None; got {self.max_cases}"
            )
        if self.image_dpi <= 0:
            raise ValueError(f"image_dpi must be > 0; got {self.image_dpi}")
