"""Safe filesystem-I/O helpers for the ``cnct_dataprep`` package."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def safe_load_nifti(path: PathLike) -> nib.Nifti1Image:
    """Load a NIfTI file, raising a descriptive error if it is missing.

    Args:
        path: Filesystem path to a ``.nii`` or ``.nii.gz`` file.

    Returns:
        The loaded :class:`nibabel.Nifti1Image`. Note that the volume data is
        lazily loaded — call ``.get_fdata()`` on the result to materialize it.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected NIfTI file at {p}, but file does not exist"
        )
    return nib.load(str(p))


def safe_load_npy(path: PathLike) -> np.ndarray:
    """Load a ``.npy`` file, raising a descriptive error if it is missing.

    Args:
        path: Filesystem path to a NumPy ``.npy`` file.

    Returns:
        The loaded :class:`numpy.ndarray`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected NumPy array at {p}, but file does not exist"
        )
    return np.load(p)


def safe_save_npy(path: PathLike, array: np.ndarray) -> None:
    """Save a NumPy array to ``.npy``, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, array)
    logger.debug(
        "Saved array shape=%s dtype=%s → %s", array.shape, array.dtype, p
    )


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and any missing parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
