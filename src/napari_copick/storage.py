"""OME-Zarr-aware read helpers for copick volume entities."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import zarr
from copick.util.ome import get_level_path, get_multiscales


@dataclass(frozen=True)
class MultiscaleLevel:
    """A metadata-selected array and the information used to select it."""

    array: zarr.Array
    level: int
    path: str
    multiscale: Mapping[str, Any]
    dataset: Mapping[str, Any]


def _entity_name(entity: Any) -> str:
    """Return a useful identifier for errors without depending on one entity type."""
    for attribute in ("tomo_type", "name"):
        value = getattr(entity, attribute, None)
        if value:
            return f"{type(entity).__name__} {value!r}"
    return type(entity).__name__


def open_multiscale_level(entity: Any, requested_level: int) -> MultiscaleLevel:
    """Open an entity read-only and select a pyramid level through OME metadata.

    Requests beyond the end of the pyramid are clamped to its final level to
    preserve the resolution selector's existing behavior.
    """
    entity_name = _entity_name(entity)
    if isinstance(requested_level, bool) or not isinstance(requested_level, int):
        raise TypeError(f"{entity_name}: resolution level must be an integer")
    if requested_level < 0:
        raise ValueError(f"{entity_name}: resolution level cannot be negative (got {requested_level})")

    store = entity.zarr()
    if store is None:
        raise ValueError(f"{entity_name}: no Zarr store is available")

    group = zarr.open_group(store=store, mode="r")
    try:
        multiscales = get_multiscales(group)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{entity_name}: OME-Zarr multiscales metadata is missing or invalid") from error

    if not multiscales:
        raise ValueError(f"{entity_name}: OME-Zarr multiscales metadata contains no entries")

    multiscale = multiscales[0]
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise ValueError(f"{entity_name}: OME-Zarr multiscale contains no datasets")

    effective_level = min(requested_level, len(datasets) - 1)
    try:
        path = get_level_path(group, effective_level)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(
            f"{entity_name}: could not resolve requested level {requested_level} "
            f"(effective level {effective_level}) from OME-Zarr metadata",
        ) from error

    try:
        array = group[path]
    except KeyError as error:
        raise ValueError(
            f"{entity_name}: requested level {requested_level} resolved to effective level "
            f"{effective_level}, but declared dataset path {path!r} does not exist",
        ) from error

    return MultiscaleLevel(
        array=array,
        level=effective_level,
        path=path,
        multiscale=multiscale,
        dataset=datasets[effective_level],
    )


def _scale_values(dataset: Mapping[str, Any], axes_count: int, *, description: str) -> tuple[float, ...]:
    transformations = dataset.get("coordinateTransformations")
    if not isinstance(transformations, Sequence) or isinstance(transformations, (str, bytes)):
        raise ValueError(f"{description} has no coordinate transformations")

    scale = next(
        (
            transform.get("scale")
            for transform in transformations
            if isinstance(transform, Mapping) and transform.get("type") == "scale"
        ),
        None,
    )
    if not isinstance(scale, Sequence) or isinstance(scale, (str, bytes)):
        raise ValueError(f"{description} has no scale coordinate transformation")
    if len(scale) != axes_count:
        raise ValueError(f"{description} scale has {len(scale)} values for {axes_count} axes")

    try:
        return tuple(float(value) for value in scale)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} scale contains a non-numeric value") from error


def spatial_scale(level: MultiscaleLevel, base_voxel_size: float) -> tuple[float, float, float]:
    """Return the selected level's ``(z, y, x)`` napari scale in Angstrom."""
    axes = level.multiscale.get("axes")
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise ValueError("OME-Zarr multiscale axes are missing or invalid")

    axis_names = []
    for axis in axes:
        name = axis.get("name") if isinstance(axis, Mapping) else axis
        if not isinstance(name, str):
            raise ValueError("OME-Zarr multiscale contains an axis without a valid name")
        axis_names.append(name)

    spatial_indices = []
    for name in ("z", "y", "x"):
        matches = [index for index, axis_name in enumerate(axis_names) if axis_name == name]
        if len(matches) != 1:
            raise ValueError(f"OME-Zarr multiscale must contain exactly one {name!r} axis")
        spatial_indices.append(matches[0])

    datasets = level.multiscale.get("datasets")
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise ValueError("OME-Zarr multiscale contains no datasets")

    first_scale = _scale_values(datasets[0], len(axis_names), description="OME-Zarr level 0")
    selected_scale = _scale_values(level.dataset, len(axis_names), description=f"OME-Zarr level {level.level}")

    result = []
    for index in spatial_indices:
        if first_scale[index] == 0:
            raise ValueError(f"OME-Zarr level 0 scale for axis {axis_names[index]!r} cannot be zero")
        result.append(float(base_voxel_size) * selected_scale[index] / first_scale[index])

    return tuple(result)
