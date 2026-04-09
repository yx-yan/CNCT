"""FDK filtered back-projection reconstruction from cone-beam projections.

Pipeline stage 2/3. For each case with a ``projections.npy`` under
:attr:`FdkCfg.proj_dir`:

    1. Load projections and rebuild the identical cone-beam geometry from the
       NIfTI header (``nVoxel`` / ``dVoxel`` vary across cases).
    2. Run TIGRE's FDK filtered back-projection.
    3. Save ``recon_fdk.npy`` (always) and, optionally, mid-slice PNGs and a
       full ``recon_fdk.nii.gz``.

The number of projection angles is derived from ``projections.shape[0]`` so
reconstruction adapts automatically to whatever count was written at forward
projection time.

Output is consumed by :mod:`cnct_dataprep.evaluation.runner` (stage 3/3).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib
import numpy as np
import tigre
import tigre.algorithms
from tigre.utilities.geometry import Geometry

from ..config.schema import FdkCfg
from ..geometry.builder import build_geometry
from ..geometry.conversions import load_nifti_as_tigre, mu_to_hu
from ..utils.io import (
    ensure_dir,
    safe_load_nifti,
    safe_load_npy,
    safe_save_npy,
)
from ..utils.paths import case_nifti_path, case_output_dir, discover_cases

logger = logging.getLogger(__name__)


def reconstruct_case(case_id: str, cfg: FdkCfg) -> bool:
    """Reconstruct a single case from its projections using FDK.

    Args:
        case_id: Case identifier (matches ``<CaseID>_0000.nii.gz`` under
            ``cfg.data_dir`` and ``<case_id>/projections.npy`` under
            ``cfg.proj_dir``).
        cfg: FDK configuration.

    Returns:
        ``True`` on success; ``False`` if the case was skipped due to missing
        input or a recoverable I/O error.
    """
    nii_path = case_nifti_path(cfg.data_dir, case_id)
    proj_path = Path(cfg.proj_dir) / case_id / "projections.npy"

    if not proj_path.exists():
        logger.warning(
            "Skipping %s — projections.npy not found at %s", case_id, proj_path
        )
        return False

    logger.info("=== Reconstructing %s ===", case_id)

    # --- Load projections ------------------------------------------------
    try:
        projections = safe_load_npy(proj_path).astype(np.float32)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error(
            "Failed to load projections for %s (%s): %s", case_id, proj_path, exc
        )
        return False
    logger.info("  Projections shape: %s", tuple(projections.shape))

    # --- Rebuild geometry from NIfTI header (must match projection) -----
    try:
        nii_img = safe_load_nifti(nii_path)
    except (FileNotFoundError, OSError) as exc:
        logger.error(
            "Failed to load NIfTI for %s (%s): %s", case_id, nii_path, exc
        )
        return False

    nVoxel, voxel_sizes = load_nifti_as_tigre(nii_img)
    geo = build_geometry(nVoxel, voxel_sizes, cfg.geometry)

    # --- FDK reconstruction ---------------------------------------------
    n_angles = int(projections.shape[0])
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    recon = tigre.algorithms.fdk(
        projections, geo, angles, filter=cfg.filter_name
    )
    logger.info("  Reconstruction shape: %s", tuple(recon.shape))

    # --- Persist outputs -------------------------------------------------
    case_out = case_output_dir(cfg.fdk_dir, case_id)
    try:
        safe_save_npy(case_out / "recon_fdk.npy", recon)
    except OSError as exc:
        logger.error("Failed to save recon for %s: %s", case_id, exc)
        return False

    if cfg.save_png:
        try:
            _save_recon_slices(recon, geo, case_out, case_id, cfg.image_dpi)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to save PNG slices for %s (continuing): %s",
                case_id,
                exc,
            )

    if cfg.save_nii:
        try:
            _save_recon_nifti(recon, case_out, nii_img, cfg.geometry.mu_water)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to save NIfTI for %s (continuing): %s", case_id, exc
            )

    logger.info("  Saved to %s", case_out)
    return True


def _save_recon_slices(
    recon: np.ndarray,
    geo: Geometry,
    case_out: Path,
    case_id: str,
    image_dpi: int,
) -> None:
    """Save axial, coronal, and sagittal mid-slices of the reconstruction.

    Slice contrast uses the reconstruction's 1st/99th percentiles to avoid
    outlier-dominated contrast. Figure sizes reflect physical voxel
    dimensions in mm so aspect ratios are correct.

    Args:
        recon: Shape ``(Z, Y, X)``, mu units.
        geo: TIGRE geometry (for ``dVoxel``).
        case_out: Output directory.
        case_id: Case identifier for figure titles.
        image_dpi: DPI for saved PNGs.
    """
    vmin, vmax = np.percentile(recon, [1, 99])
    dz, dy, dx = (float(v) for v in geo.dVoxel)
    nz, ny, nx = recon.shape

    slices = {
        "axial": (recon[nz // 2, :, :], ny * dy, nx * dx),
        "coronal": (recon[:, ny // 2, :], nz * dz, nx * dx),
        "sagittal": (recon[:, :, nx // 2], nz * dz, ny * dy),
    }
    for name, (img, phys_h, phys_w) in slices.items():
        scale = 6.0 / max(phys_h, phys_w)
        fig, ax = plt.subplots(figsize=(phys_w * scale, phys_h * scale))
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{case_id} — {name}")
        ax.axis("off")
        fig.savefig(
            case_out / f"recon_{name}.png",
            bbox_inches="tight",
            dpi=image_dpi,
        )
        plt.close(fig)


def _save_recon_nifti(
    recon: np.ndarray,
    case_out: Path,
    nii_img: nib.Nifti1Image,
    mu_water: float,
) -> None:
    """Save the reconstruction as a NIfTI file in HU units.

    The output preserves the source NIfTI's affine and header so it aligns
    with the ground-truth volume in any viewer.

    Args:
        recon: Shape ``(Z, Y, X)``, mu units.
        case_out: Output directory.
        nii_img: Source NIfTI image (for affine + header).
        mu_water: Linear attenuation of water (mm⁻¹) for mu -> HU conversion.
    """
    recon_hu = mu_to_hu(recon, mu_water)
    recon_nii = nib.Nifti1Image(
        np.transpose(recon_hu, (2, 1, 0)),
        affine=nii_img.affine,
        header=nii_img.header,
    )
    nib.save(recon_nii, str(case_out / "recon_fdk.nii.gz"))


def run_fdk(cfg: FdkCfg) -> None:
    """Orchestrate FDK reconstruction across every configured case.

    Args:
        cfg: FDK configuration.
    """
    ensure_dir(cfg.fdk_dir)
    cases = discover_cases(
        cfg.data_dir,
        start=cfg.case_start,
        end=cfg.case_end,
        max_cases=cfg.max_cases,
    )
    if not cases:
        logger.warning(
            "No cases found under %s; nothing to do", cfg.data_dir
        )
        return

    logger.info(
        "FDK-reconstructing %d case(s) with filter=%s",
        len(cases),
        cfg.filter_name,
    )

    n_success = 0
    n_skipped = 0
    for case_id in cases:
        if reconstruct_case(case_id, cfg):
            n_success += 1
        else:
            n_skipped += 1

    logger.info(
        "FDK reconstruction complete: %d succeeded, %d skipped/failed",
        n_success,
        n_skipped,
    )
