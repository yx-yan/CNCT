"""Forward projection of CT volumes into simulated cone-beam X-ray projections.

Pipeline stage 1/3. For each NIfTI volume under :attr:`ProjectionCfg.data_dir`:

    1. Load and transpose to TIGRE's (Z, Y, X) axis order.
    2. Convert HU to linear attenuation (mu).
    3. Build per-case cone-beam geometry from the NIfTI header.
    4. Run TIGRE's GPU-accelerated Siddon ray-tracing (``tigre.Ax``).
    5. Save ``projections.npy`` (and optional PNG previews) to
       ``cfg.proj_dir/<case>/``.

Output is consumed by :mod:`cnct_dataprep.reconstruction.fdk` (stage 2/3).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np
import tigre
from tigre.utilities.geometry import Geometry

from ..config.schema import ProjectionCfg
from ..geometry.builder import build_geometry
from ..geometry.conversions import hu_to_mu
from ..utils.io import ensure_dir, safe_load_nifti, safe_save_npy
from ..utils.paths import case_nifti_path, case_output_dir, discover_cases

logger = logging.getLogger(__name__)


def make_angles(n_angles: int) -> np.ndarray:
    """Generate uniformly-spaced projection angles over 360 degrees.

    Args:
        n_angles: Number of projection angles.

    Returns:
        A float64 ``(n_angles,)`` array of angles in radians, spanning
        ``[0, 2π)`` with ``endpoint=False``.

    Raises:
        ValueError: If ``n_angles`` is not strictly positive.
    """
    if n_angles <= 0:
        raise ValueError(f"n_angles must be > 0; got {n_angles}")
    return np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)


def project_case(
    case_id: str,
    cfg: ProjectionCfg,
    angles: np.ndarray,
) -> bool:
    """Run forward projection for a single case.

    Args:
        case_id: Case identifier (matches ``<CaseID>_0000.nii.gz`` under
            ``cfg.data_dir``).
        cfg: Projection configuration.
        angles: Projection angles in radians, shape ``(n_angles,)``.

    Returns:
        ``True`` if the case was processed and persisted successfully;
        ``False`` if it was skipped due to a recoverable I/O error (input
        missing, output write failed). Non-recoverable errors (TIGRE kernel
        crashes, shape mismatches) propagate.
    """
    nii_path = case_nifti_path(cfg.data_dir, case_id)
    logger.info("=== Processing %s ===", case_id)

    # --- Load and preprocess volume --------------------------------------
    try:
        nii_img = safe_load_nifti(nii_path)
        volume = nii_img.get_fdata().astype(np.float32)
    except (FileNotFoundError, OSError) as exc:
        logger.error("Failed to load NIfTI for %s (%s): %s", case_id, nii_path, exc)
        return False

    voxel_sizes = np.array(
        nii_img.header.get_zooms()[:3], dtype=np.float32
    )
    logger.info(
        "  Volume shape (X,Y,Z)=%s, voxel sizes (mm)=%s",
        tuple(volume.shape),
        voxel_sizes.tolist(),
    )

    # Transpose NIfTI (X, Y, Z) -> TIGRE (Z, Y, X); copy for contiguity.
    volume = np.transpose(volume, (2, 1, 0)).copy()
    nVoxel = np.array(volume.shape, dtype=np.int64)

    # HU -> linear attenuation (mm^-1), contiguous for GPU.
    volume = hu_to_mu(volume, cfg.geometry.mu_water)
    volume = np.ascontiguousarray(volume)

    # --- Build geometry and run TIGRE forward projection ----------------
    geo = build_geometry(nVoxel, voxel_sizes, cfg.geometry)
    projections = tigre.Ax(volume, geo, angles)
    logger.info("  Projections shape: %s", tuple(projections.shape))

    # --- Persist outputs -------------------------------------------------
    case_out = case_output_dir(cfg.proj_dir, case_id)
    try:
        safe_save_npy(case_out / "projections.npy", projections)
    except OSError as exc:
        logger.error("Failed to save projections for %s: %s", case_id, exc)
        return False

    if cfg.save_png:
        try:
            _save_projection_pngs(
                projections, volume, geo, angles, case_out, case_id, cfg
            )
        except Exception as exc:  # noqa: BLE001 — matplotlib / I/O failures
            logger.warning(
                "Failed to save PNG previews for %s (continuing): %s",
                case_id,
                exc,
            )

    logger.info("  Saved to %s", case_out)
    return True


def _save_projection_pngs(
    projections: np.ndarray,
    volume: np.ndarray,
    geo: Geometry,
    angles: np.ndarray,
    case_out: Path,
    case_id: str,
    cfg: ProjectionCfg,
) -> None:
    """Save per-angle projection PNGs plus a mid-axial volume slice.

    Contrast is global across all projection angles (so every preview uses
    the same ``vmin``/``vmax``). Figure sizes reflect physical detector /
    voxel dimensions in mm so aspect ratios are correct.

    Args:
        projections: Shape ``(n_angles, det_rows, det_cols)``, mu units.
        volume: Shape ``(Z, Y, X)``, mu units.
        geo: TIGRE geometry object used for physical-size calculations.
        angles: Shape ``(n_angles,)``, radians.
        case_out: Per-case output directory.
        case_id: Case identifier for figure titles.
        cfg: Projection configuration (``proj_save_every`` and ``image_dpi``).
    """
    vmin_proj = float(projections.min())
    vmax_proj = float(projections.max())

    phys_det_h = float(geo.nDetector[0] * geo.dDetector[0])
    phys_det_w = float(geo.nDetector[1] * geo.dDetector[1])
    det_scale = 6.0 / max(phys_det_h, phys_det_w)

    n_angles = projections.shape[0]
    for i in range(0, n_angles, cfg.proj_save_every):
        fig, ax = plt.subplots(
            figsize=(phys_det_w * det_scale, phys_det_h * det_scale)
        )
        ax.imshow(
            projections[i],
            cmap="gray",
            aspect="auto",
            vmin=vmin_proj,
            vmax=vmax_proj,
        )
        ax.set_title(f"{case_id} — {np.degrees(angles[i]):.1f}°")
        ax.axis("off")
        fig.savefig(
            case_out / f"proj_{i:03d}.png",
            bbox_inches="tight",
            dpi=cfg.image_dpi,
        )
        plt.close(fig)

    # Mid axial slice of the input volume.
    phys_vol_h = float(geo.nVoxel[1] * geo.dVoxel[1])
    phys_vol_w = float(geo.nVoxel[2] * geo.dVoxel[2])
    vol_scale = 6.0 / max(phys_vol_h, phys_vol_w)
    mid_z = volume.shape[0] // 2

    fig, ax = plt.subplots(
        figsize=(phys_vol_w * vol_scale, phys_vol_h * vol_scale)
    )
    ax.imshow(volume[mid_z], cmap="gray", aspect="auto")
    ax.set_title(f"{case_id} — axial slice z={mid_z}")
    ax.axis("off")
    fig.savefig(
        case_out / "volume_axial_mid.png",
        bbox_inches="tight",
        dpi=cfg.image_dpi,
    )
    plt.close(fig)


def run_projection(cfg: ProjectionCfg) -> None:
    """Orchestrate forward projection across every configured case.

    Cases are discovered from ``cfg.data_dir`` and filtered by
    ``case_start``/``case_end``/``max_cases``. Per-case failures are logged
    and skipped so one bad case does not kill the whole batch.

    Args:
        cfg: Projection configuration.
    """
    ensure_dir(cfg.proj_dir)
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
        "Forward-projecting %d case(s) with %d angles over 360 degrees",
        len(cases),
        cfg.n_angles,
    )
    angles = make_angles(cfg.n_angles)

    n_success = 0
    n_failed = 0
    for case_id in cases:
        if project_case(case_id, cfg, angles):
            n_success += 1
        else:
            n_failed += 1

    logger.info(
        "Forward projection complete: %d succeeded, %d failed",
        n_success,
        n_failed,
    )
