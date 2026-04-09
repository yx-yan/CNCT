"""Case discovery and per-case output directory helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_CASE_SUFFIX = "_0000.nii.gz"


def discover_cases(
    data_dir: PathLike,
    start: int = 0,
    end: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> List[str]:
    """Return a sorted list of case IDs under ``data_dir``.

    A "case" is any file matching ``<CaseID>_0000.nii.gz`` in ``data_dir``; the
    returned IDs strip the ``_0000.nii.gz`` suffix.

    Args:
        data_dir: Directory containing ``<CaseID>_0000.nii.gz`` files.
        start: Inclusive index into the sorted case list (slicing).
        end: Exclusive index into the sorted case list. ``None`` means "to end".
        max_cases: Hard cap on the number of cases returned after ``start``/
            ``end`` slicing. ``None`` means no cap.

    Returns:
        List of case-ID strings.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    case_ids: List[str] = []
    for entry in sorted(root.glob(f"*{_CASE_SUFFIX}")):
        case_id = entry.name.removesuffix(_CASE_SUFFIX)
        case_ids.append(case_id)

    sliced = case_ids[start:end]
    if max_cases is not None:
        sliced = sliced[:max_cases]

    logger.info(
        "Discovered %d case(s) under %s (using %d after slicing)",
        len(case_ids), root, len(sliced),
    )
    return sliced


def case_nifti_path(data_dir: PathLike, case_id: str) -> Path:
    """Return the full NIfTI path for ``case_id`` under ``data_dir``."""
    return Path(data_dir) / f"{case_id}{_CASE_SUFFIX}"


def case_output_dir(root: PathLike, case_id: str) -> Path:
    """Return ``<root>/<case_id>``, creating it if it does not exist."""
    p = Path(root) / case_id
    p.mkdir(parents=True, exist_ok=True)
    return p
