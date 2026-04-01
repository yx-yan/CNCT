"""Shared CBCT geometry and unit-conversion utilities.

This module centralises the TIGRE geometry construction and HU/mu conversion
logic used by both the forward-projection and reconstruction pipelines.
Keeping a single source of truth prevents geometry divergence between
projection.py and fdk.py, which would cause mis-registered reconstructions.
"""

import numpy as np
import tigre

from config import DSO_SCALE, DSD_SCALE, DETECTOR_COL_MARGIN, ACCURACY, MU_WATER


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def hu_to_mu(volume_hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    """Convert Hounsfield units to linear attenuation coefficients (mu, mm-1).

    The Hounsfield scale is defined as:
        HU = (mu - mu_water) / mu_water * 1000

    Rearranging:
        mu = (HU + 1000) / 1000 * mu_water

    Negative mu values are clipped to 0 (air/void has no physical attenuation).

    Parameters
    ----------
    volume_hu : ndarray
        Volume in Hounsfield units.
    mu_water : float
        Linear attenuation of water at ~70 keV (default from config.py).

    Returns
    -------
    ndarray (float32)
        Volume in linear attenuation units (mm-1).
    """
    return np.clip(
        (volume_hu.astype(np.float32) + 1000.0) / 1000.0 * mu_water, 0.0, None
    )


def mu_to_hu(volume_mu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    """Convert linear attenuation coefficients back to Hounsfield units.

    Inverse of :func:`hu_to_mu`.

    Parameters
    ----------
    volume_mu : ndarray
        Volume in linear attenuation units (mm-1).
    mu_water : float
        Linear attenuation of water at ~70 keV.

    Returns
    -------
    ndarray (float32)
        Volume in Hounsfield units.
    """
    return (volume_mu / mu_water * 1000.0 - 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------

def build_geometry(
    nVoxel: np.ndarray, voxel_sizes: np.ndarray
) -> tigre.utilities.geometry.Geometry:
    """Build a TIGRE cone-beam geometry from voxel grid dimensions.

    The geometry is derived entirely from the NIfTI header so it adapts to
    each case's volume size and voxel spacing.  Both projection.py and fdk.py
    must call this function to guarantee identical geometry.

    Parameters
    ----------
    nVoxel : ndarray, shape (3,), dtype int64
        Number of voxels in (Z, Y, X) order (TIGRE convention).
    voxel_sizes : ndarray, shape (3,), dtype float32
        Voxel spacing in (X, Y, Z) order (NIfTI convention).
        Reordered internally to (Z, Y, X) for TIGRE.

    Returns
    -------
    tigre.utilities.geometry.Geometry
        Fully configured cone-beam geometry object.
    """
    geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = nVoxel
    geo.dVoxel = np.array([voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]])
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # Source-to-origin distance scaled so the source is always outside the object
    max_radius = np.sqrt((geo.sVoxel[1] / 2) ** 2 + (geo.sVoxel[2] / 2) ** 2)
    geo.DSO = max_radius * DSO_SCALE
    geo.DSD = geo.DSO * DSD_SCALE

    # Detector sized to cover the full volume cross-section with margin
    magnification = geo.DSD / geo.DSO
    geo.nDetector = np.array([nVoxel[0], max(nVoxel[1], nVoxel[2])])
    geo.dDetector = np.array([
        geo.dVoxel[0] * magnification,
        geo.dVoxel[2] * magnification * DETECTOR_COL_MARGIN,
    ])
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])
    geo.accuracy = ACCURACY

    return geo


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def load_nifti_as_tigre(nii_img) -> tuple:
    """Extract volume shape and voxel sizes from a loaded NIfTI image.

    Returns values in the formats expected by :func:`build_geometry`:
    nVoxel in (Z, Y, X) order and voxel_sizes in (X, Y, Z) order.

    Parameters
    ----------
    nii_img : nibabel.nifti1.Nifti1Image
        A loaded NIfTI image (from ``nib.load(...)``).

    Returns
    -------
    nVoxel : ndarray, shape (3,), dtype int64
        Volume dimensions in (Z, Y, X) order.
    voxel_sizes : ndarray, shape (3,), dtype float32
        Voxel spacing in (X, Y, Z) order (NIfTI convention).
    """
    shape = nii_img.header.get_data_shape()  # (X, Y, Z)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    nVoxel = np.array([shape[2], shape[1], shape[0]], dtype=np.int64)
    return nVoxel, voxel_sizes
