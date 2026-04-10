"""YAML loaders that translate config files into typed dataclasses.

The loaders are the only place where path strings are coerced to
:class:`pathlib.Path`. All code downstream of ``load_*_cfg`` should receive
fully-typed, pre-validated config objects.

Composition:
    Stage configs (``projection.yaml``, ``fdk.yaml``) reference a geometry
    config via a ``geometry_config`` field containing a path relative to the
    stage YAML's directory. The loaders automatically resolve and load it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .schema import EvaluationCfg, FdkCfg, GeometryCfg, ProjectionCfg

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict.

    Args:
        path: Filesystem path to a YAML file.

    Returns:
        Parsed YAML top-level mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML top level is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level YAML in {path} must be a mapping, "
            f"got {type(data).__name__}"
        )
    return data


def _resolve_geometry_ref(
    stage_data: Dict[str, Any], stage_path: Path
) -> GeometryCfg:
    """Pop the ``geometry_config`` field from a stage YAML and load it.

    The referenced path is resolved relative to the stage YAML's directory.

    Args:
        stage_data: Parsed stage YAML as a mutable dict. The ``geometry_config``
            key is removed in-place so the remainder can be splatted into the
            stage dataclass constructor.
        stage_path: Filesystem path of the stage YAML file. Used as the base
            for resolving relative geometry references.

    Returns:
        The loaded :class:`GeometryCfg`.

    Raises:
        ValueError: If ``geometry_config`` is missing from ``stage_data``.
    """
    geo_ref = stage_data.pop("geometry_config", None)
    if geo_ref is None:
        raise ValueError(
            f"Stage config {stage_path} is missing required "
            f"'geometry_config' field"
        )
    geo_path = (stage_path.parent / Path(geo_ref)).resolve()
    return load_geometry_cfg(geo_path)


def load_geometry_cfg(path: PathLike) -> GeometryCfg:
    """Load a :class:`GeometryCfg` from a YAML file.

    Args:
        path: Path to a geometry YAML file.

    Returns:
        A validated :class:`GeometryCfg`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    yaml_path = Path(path).resolve()
    data = _load_yaml(yaml_path)
    logger.debug("Loading GeometryCfg from %s", yaml_path)
    return GeometryCfg(**data)


def load_projection_cfg(path: PathLike) -> ProjectionCfg:
    """Load a :class:`ProjectionCfg` from a YAML file.

    The YAML must contain a ``geometry_config`` field pointing to a geometry
    YAML file (relative to the stage YAML's directory).

    Args:
        path: Path to a projection YAML file.

    Returns:
        A validated :class:`ProjectionCfg` with ``geometry`` populated from
        the referenced geometry YAML.

    Raises:
        FileNotFoundError: If ``path`` or the referenced geometry file does not
            exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    stage_path = Path(path).resolve()
    data = _load_yaml(stage_path)
    geometry = _resolve_geometry_ref(data, stage_path)

    # Resolve {base_dir} and {n_angles} placeholders in path fields.
    base_dir = data.pop("base_dir", "")
    data.setdefault("n_angles", geometry.n_angles)
    n_angles = data["n_angles"]
    for key in ("data_dir", "proj_dir"):
        if key in data:
            data[key] = str(data[key]).format(base_dir=base_dir, n_angles=n_angles)

    for key in ("data_dir", "proj_dir"):
        if key in data:
            data[key] = Path(data[key])

    logger.debug("Loading ProjectionCfg from %s", stage_path)
    return ProjectionCfg(geometry=geometry, **data)


def load_fdk_cfg(path: PathLike) -> FdkCfg:
    """Load an :class:`FdkCfg` from a YAML file.

    The YAML must contain a ``geometry_config`` field pointing to a geometry
    YAML file (relative to the stage YAML's directory).

    Args:
        path: Path to an FDK YAML file.

    Returns:
        A validated :class:`FdkCfg` with ``geometry`` populated from the
        referenced geometry YAML.

    Raises:
        FileNotFoundError: If ``path`` or the referenced geometry file does not
            exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    stage_path = Path(path).resolve()
    data = _load_yaml(stage_path)
    geometry = _resolve_geometry_ref(data, stage_path)

    # Resolve {base_dir} and {n_angles} placeholders from geometry.
    base_dir = data.pop("base_dir", "")
    n_angles = geometry.n_angles
    for key in ("data_dir", "proj_dir", "fdk_dir"):
        if key in data:
            data[key] = str(data[key]).format(base_dir=base_dir, n_angles=n_angles)

    for key in ("data_dir", "proj_dir", "fdk_dir"):
        if key in data:
            data[key] = Path(data[key])

    logger.debug("Loading FdkCfg from %s", stage_path)
    return FdkCfg(geometry=geometry, **data)


def load_evaluation_cfg(path: PathLike) -> EvaluationCfg:
    """Load an :class:`EvaluationCfg` from a YAML file.

    The YAML must contain a ``geometry_config`` field pointing to a geometry
    YAML file (relative to the stage YAML's directory).

    Args:
        path: Path to an evaluation YAML file.

    Returns:
        A validated :class:`EvaluationCfg` with ``geometry`` populated from
        the referenced geometry YAML.

    Raises:
        FileNotFoundError: If ``path`` or the referenced geometry file does not
            exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    stage_path = Path(path).resolve()
    data = _load_yaml(stage_path)
    geometry = _resolve_geometry_ref(data, stage_path)

    # Resolve {base_dir} and {n_angles} placeholders from geometry.
    base_dir = data.pop("base_dir", "")
    n_angles = geometry.n_angles
    for key in ("data_dir", "fdk_dir", "eval_dir"):
        if key in data:
            data[key] = str(data[key]).format(base_dir=base_dir, n_angles=n_angles)

    for key in ("data_dir", "fdk_dir", "eval_dir"):
        if key in data:
            data[key] = Path(data[key])

    logger.debug("Loading EvaluationCfg from %s", stage_path)
    return EvaluationCfg(geometry=geometry, **data)
