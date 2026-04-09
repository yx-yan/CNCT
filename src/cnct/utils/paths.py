"""Case discovery and per-case path helpers for the ``cnct`` DL package.

Case discovery for the DL pipeline is driven by the HDF5 split directories
written by the data-preparation stage (``h5_3dunet/{train,val,test}/``) —
filenames alone determine the split, not the HDF5 contents — so the DL
pipeline reuses whatever split the FDK pipeline already committed to.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_CASE_SUFFIX = "_0000.nii.gz"


def discover_split_cases(h5_root: PathLike, split: str) -> List[str]:
    """Return a sorted list of case IDs for a given HDF5 split.

    Args:
        h5_root: Root directory containing per-split subdirectories
            (``train/``, ``val/``, ``test/``).
        split: Split name (``"train"``, ``"val"``, or ``"test"``).

    Returns:
        Sorted list of case IDs; each is the basename (stem) of an ``.h5``
        file under ``<h5_root>/<split>/``.

    Raises:
        FileNotFoundError: If ``<h5_root>/<split>/`` does not exist.
    """
    split_dir = Path(h5_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"HDF5 split directory not found: {split_dir}"
        )
    case_ids = sorted(p.stem for p in split_dir.glob("*.h5"))
    logger.info(
        "Discovered %d case(s) in split '%s' under %s",
        len(case_ids),
        split,
        split_dir,
    )
    return case_ids


def case_nifti_path(data_dir: PathLike, case_id: str) -> Path:
    """Return the full NIfTI path for ``case_id`` under ``data_dir``.

    Args:
        data_dir: Directory containing ``<CaseID>_0000.nii.gz`` files.
        case_id: Case identifier (without the ``_0000.nii.gz`` suffix).

    Returns:
        ``<data_dir>/<case_id>_0000.nii.gz`` — not checked for existence.
    """
    return Path(data_dir) / f"{case_id}{_CASE_SUFFIX}"


def case_sinogram_path(proj_dir: PathLike, case_id: str) -> Path:
    """Return the ``projections.npy`` path for ``case_id``."""
    return Path(proj_dir) / case_id / "projections.npy"


def case_fdk_path(fdk_dir: PathLike, case_id: str) -> Path:
    """Return the ``recon_fdk.npy`` path for ``case_id``."""
    return Path(fdk_dir) / case_id / "recon_fdk.npy"
