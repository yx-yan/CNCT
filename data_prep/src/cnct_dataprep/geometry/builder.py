"""TIGRE cone-beam geometry construction.

This is the single source of truth for how the pipeline builds a
:class:`tigre.utilities.geometry.Geometry` from per-case volume metadata.
Forward projection and FDK reconstruction MUST call :func:`build_geometry` with
the same inputs to guarantee identical geometry — otherwise reconstructions
will be mis-registered.
"""
from __future__ import annotations

import logging

import numpy as np
import tigre
from tigre.utilities.geometry import Geometry

from ..config.schema import GeometryCfg

logger = logging.getLogger(__name__)


def build_geometry(
    nVoxel: np.ndarray,
    voxel_sizes: np.ndarray,
    cfg: GeometryCfg,
) -> Geometry:
    """Build a TIGRE cone-beam geometry from per-case volume metadata.

    Geometry is derived entirely from the NIfTI header so it adapts to each
    case's volume size and voxel spacing. The key invariants are:

        * ``DSO = max_radius * cfg.dso_scale``, where ``max_radius`` is the
          XY half-diagonal of the physical volume. This guarantees the source
          is outside every volume and removes the circular FOV artifact.
        * ``DSD = DSO * cfg.dsd_scale`` (controls magnification).
        * Detector has ``nVoxel[0]`` rows and ``max(nVoxel[1], nVoxel[2])``
          columns. The column pitch is multiplied by ``cfg.detector_col_margin``
          to prevent projection truncation for off-centre voxels.

    Conventions:
        * ``nVoxel`` is in (Z, Y, X) order (TIGRE convention).
        * ``voxel_sizes`` is in (X, Y, Z) order (NIfTI convention); reordered
          internally to (Z, Y, X) for TIGRE's ``dVoxel``.

    Args:
        nVoxel: Volume dimensions in (Z, Y, X) order, shape ``(3,)``,
            dtype ``int64``.
        voxel_sizes: Voxel spacing in (X, Y, Z) order, shape ``(3,)``,
            dtype ``float32``.
        cfg: Cone-beam geometry configuration.

    Returns:
        A fully configured :class:`tigre.utilities.geometry.Geometry` ready for
        :func:`tigre.Ax`, :func:`tigre.Atb`, or :func:`tigre.algorithms.fdk`.

    Raises:
        ValueError: If ``nVoxel`` or ``voxel_sizes`` does not have shape
            ``(3,)``.
    """
    if nVoxel.shape != (3,):
        raise ValueError(
            f"nVoxel must have shape (3,); got {nVoxel.shape}"
        )
    if voxel_sizes.shape != (3,):
        raise ValueError(
            f"voxel_sizes must have shape (3,); got {voxel_sizes.shape}"
        )

    geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = nVoxel
    geo.dVoxel = np.array(
        [voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]]
    )
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # Source-to-origin distance scaled so the source is always outside the volume.
    max_radius = np.sqrt(
        (geo.sVoxel[1] / 2) ** 2 + (geo.sVoxel[2] / 2) ** 2
    )
    geo.DSO = max_radius * cfg.dso_scale
    geo.DSD = geo.DSO * cfg.dsd_scale

    # Detector sized to cover the full volume cross-section with margin.
    magnification = geo.DSD / geo.DSO
    geo.nDetector = np.array([nVoxel[0], max(nVoxel[1], nVoxel[2])])
    geo.dDetector = np.array(
        [
            geo.dVoxel[0] * magnification,
            geo.dVoxel[2] * magnification * cfg.detector_col_margin,
        ]
    )
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])
    geo.accuracy = cfg.accuracy

    logger.debug(
        "Built geometry: nVoxel=%s dVoxel=%s DSO=%.2f DSD=%.2f "
        "nDetector=%s dDetector=%s",
        geo.nVoxel.tolist(),
        geo.dVoxel.tolist(),
        float(geo.DSO),
        float(geo.DSD),
        geo.nDetector.tolist(),
        geo.dDetector.tolist(),
    )

    return geo
