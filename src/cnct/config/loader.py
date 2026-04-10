"""YAML loaders that translate config files into typed dataclasses.

Top-level training/inference YAML files have the shape::

    geometry_config: geometry.yaml   # relative to this file's directory
    data:
      data_dir: ...
      proj_dir: ...
      ...
    model:
      sinogram_f_maps: [8, 16, 32, 64]
      ...
    loss: {...}
    optimizer: {...}
    scheduler: {...}
    epochs: 200
    ...

``geometry_config`` is resolved to an absolute path relative to the stage
YAML's directory, matching the pattern used by :mod:`cnct_dataprep`. All
path strings are coerced to :class:`pathlib.Path` at load time so downstream
code receives fully-typed, pre-validated config objects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .schema import (
    DataCfg,
    GeometryCfg,
    InferenceCfg,
    LossCfg,
    ModelCfg,
    OptimizerCfg,
    SchedulerCfg,
    TrainingCfg,
)

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
    """Pop the ``geometry_config`` field and load the referenced geometry file.

    Args:
        stage_data: Parsed stage YAML as a mutable dict. The ``geometry_config``
            key is removed in-place.
        stage_path: Filesystem path of the stage YAML file.

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


def _build_data_cfg(section: Dict[str, Any], base_dir: str = "") -> DataCfg:
    """Construct a :class:`DataCfg` from a YAML sub-section (path coercion)."""
    coerced = dict(section)
    for key in ("data_dir", "proj_dir", "fdk_dir", "h5_root"):
        if key in coerced:
            coerced[key] = Path(str(coerced[key]).format(base_dir=base_dir))
    return DataCfg(**coerced)


def _build_model_cfg(section: Dict[str, Any]) -> ModelCfg:
    """Construct a :class:`ModelCfg` from a YAML sub-section (list→tuple)."""
    coerced = dict(section)
    for key in ("sinogram_f_maps", "volume_f_maps"):
        if key in coerced and not isinstance(coerced[key], tuple):
            coerced[key] = tuple(coerced[key])
    return ModelCfg(**coerced)


def load_training_cfg(path: PathLike) -> TrainingCfg:
    """Load a :class:`TrainingCfg` from a YAML file.

    Args:
        path: Path to a training YAML config.

    Returns:
        A validated :class:`TrainingCfg` with every nested sub-config
        constructed.

    Raises:
        FileNotFoundError: If ``path`` or the referenced geometry file does not
            exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    stage_path = Path(path).resolve()
    data = _load_yaml(stage_path)

    geometry = _resolve_geometry_ref(data, stage_path)

    base_dir = data.pop("base_dir", "")

    if "data" not in data:
        raise ValueError(
            f"Training config {stage_path} is missing required 'data' section"
        )
    if "model" not in data:
        raise ValueError(
            f"Training config {stage_path} is missing required 'model' section"
        )

    data_cfg = _build_data_cfg(data.pop("data"), base_dir=base_dir)
    model_cfg = _build_model_cfg(data.pop("model"))
    loss_cfg = LossCfg(**data.pop("loss", {}))
    optimizer_cfg = OptimizerCfg(**data.pop("optimizer", {}))
    scheduler_cfg = SchedulerCfg(**data.pop("scheduler", {}))

    for key in ("checkpoint_dir", "resume"):
        if key in data and data[key] is not None:
            data[key] = Path(str(data[key]).format(base_dir=base_dir))

    logger.debug("Loading TrainingCfg from %s", stage_path)
    return TrainingCfg(
        geometry=geometry,
        data=data_cfg,
        model=model_cfg,
        loss=loss_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        **data,
    )


def load_inference_cfg(path: PathLike) -> InferenceCfg:
    """Load an :class:`InferenceCfg` from a YAML file.

    Args:
        path: Path to an inference YAML config.

    Returns:
        A validated :class:`InferenceCfg` with every nested sub-config
        constructed.

    Raises:
        FileNotFoundError: If ``path`` or the referenced geometry file does not
            exist.
        ValueError: If the YAML is malformed or contains invalid values.
        TypeError: If unknown keys are present.
    """
    stage_path = Path(path).resolve()
    data = _load_yaml(stage_path)

    geometry = _resolve_geometry_ref(data, stage_path)
    base_dir = data.pop("base_dir", "")

    if "data" not in data:
        raise ValueError(
            f"Inference config {stage_path} is missing required 'data' section"
        )
    if "model" not in data:
        raise ValueError(
            f"Inference config {stage_path} is missing required 'model' section"
        )

    data_cfg = _build_data_cfg(data.pop("data"), base_dir=base_dir)
    model_cfg = _build_model_cfg(data.pop("model"))

    for key in ("checkpoint", "output_dir"):
        if key in data:
            data[key] = Path(str(data[key]).format(base_dir=base_dir))

    logger.debug("Loading InferenceCfg from %s", stage_path)
    return InferenceCfg(
        geometry=geometry,
        data=data_cfg,
        model=model_cfg,
        **data,
    )
