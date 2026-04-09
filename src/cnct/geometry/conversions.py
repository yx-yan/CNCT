"""HU ↔ mu conversion and NIfTI-to-TIGRE metadata extraction.

This module is duplicated from :mod:`cnct_dataprep.geometry.conversions` so
that the deep-learning package (``cnct``) has no runtime dependency on the
data-preparation package (``cnct_dataprep``). Both implementations must stay
numerically identical — the DBP layer in :mod:`cnct.models` depends on the
same geometry the FDK forward projection was built with.
"""
from __future__ import annotations

from typing import Tuple

import nibabel as nib
import numpy as np


def hu_to_mu(volume_hu: np.ndarray, mu_water: float) -> np.ndarray:
    """Convert Hounsfield units to linear attenuation coefficients (mm⁻¹).

    The Hounsfield scale is defined as::

        HU = (mu - mu_water) / mu_water * 1000

    Rearranging::

        mu = (HU + 1000) / 1000 * mu_water

    Negative mu values are clipped to ``0`` because air/void has no physical
    attenuation.

    Args:
        volume_hu: Volume in Hounsfield units. Any dtype; cast to ``float32``
            before conversion.
        mu_water: Linear attenuation of water at the target energy (mm⁻¹).

    Returns:
        Volume in linear attenuation units (mm⁻¹), dtype ``float32``, clipped
        to ``[0, ∞)``.
    """
    return np.clip(
        (volume_hu.astype(np.float32) + 1000.0) / 1000.0 * mu_water,
        0.0,
        None,
    )


def mu_to_hu(volume_mu: np.ndarray, mu_water: float) -> np.ndarray:
    """Convert linear attenuation coefficients back to Hounsfield units.

    Inverse of :func:`hu_to_mu`. Note that values originally clipped to ``0`` by
    ``hu_to_mu`` cannot be recovered.

    Args:
        volume_mu: Volume in linear attenuation units (mm⁻¹).
        mu_water: Linear attenuation of water at the target energy (mm⁻¹).

    Returns:
        Volume in Hounsfield units, dtype ``float32``.
    """
    return (volume_mu / mu_water * 1000.0 - 1000.0).astype(np.float32)


def load_nifti_as_tigre(
    nii_img: nib.Nifti1Image,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract volume dimensions and voxel spacing from a NIfTI header.

    The returned arrays are in the formats expected by
    :func:`cnct.geometry.builder.build_geometry`:

        - ``nVoxel`` is in (Z, Y, X) order (TIGRE convention).
        - ``voxel_sizes`` is in (X, Y, Z) order (NIfTI convention); the builder
          reorders it to (Z, Y, X) internally.

    Args:
        nii_img: A loaded :class:`nibabel.Nifti1Image`.

    Returns:
        A tuple ``(nVoxel, voxel_sizes)`` where:

            * ``nVoxel`` is a shape-``(3,)`` ``int64`` array in (Z, Y, X) order.
            * ``voxel_sizes`` is a shape-``(3,)`` ``float32`` array in (X, Y, Z)
              order.
    """
    shape = nii_img.header.get_data_shape()  # (X, Y, Z)
    voxel_sizes = np.array(nii_img.header.get_zooms()[:3], dtype=np.float32)
    nVoxel = np.array([shape[2], shape[1], shape[0]], dtype=np.int64)
    return nVoxel, voxel_sizes
