"""Instance-aware napari layers: colormaps for instance and panoptic labels, and the adapters that connect the shared
instance browser to points, filament and labels layers.

Colours follow copick-shared-ui (``instance_rgb``), so an ID has the same colour in picks, filaments and instance or
panoptic segmentations, in napari, ChimeraX and copick-web.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
from copick_shared_ui.util.filaments import point_at_arc_length
from copick_shared_ui.util.instances import (
    InstanceRow,
    bbox_center,
    instance_color_dict,
    instance_colors,
    rows_from_counts,
    rows_from_filaments,
    rows_from_points,
)
from napari.utils import DirectLabelColormap

logger = logging.getLogger(__name__)

SEGMENTATION_TYPE_KEY = "copick_segmentation_type"
COUNTS_KEY = "copick_instance_counts"
HIDDEN_KEY = "copick_hidden_ids"
PANOPTIC_ROWS_KEY = "copick_panoptic_rows"
BBOXES_KEY = "copick_instance_bboxes"
COLOR_MODE_KEY = "copick_color_mode"


def instance_colormap(ids: Iterable[int], hidden: Iterable[int] = (), opacity: float = 1.0) -> DirectLabelColormap:
    """A direct colormap for instance IDs: 0 and hidden IDs transparent, unknown IDs grey."""
    color_dict: Dict[Optional[int], Any] = {
        k: np.asarray(v) for k, v in instance_color_dict(ids, opacity, hidden).items()
    }
    color_dict[None] = np.array([0.6, 0.6, 0.6, opacity])
    return DirectLabelColormap(color_dict=color_dict)


def segment_colormap(rows: List[InstanceRow], hidden: Iterable[int] = ()) -> DirectLabelColormap:
    """A direct colormap for a panoptic segment-index volume (row keys are segment indices)."""
    hidden = set(hidden)
    color_dict: Dict[Optional[int], Any] = {0: np.zeros(4)}
    for r in rows:
        color_dict[r.row_key] = np.zeros(4) if r.row_key in hidden else np.asarray(r.color, dtype=float)
    color_dict[None] = np.array([0.6, 0.6, 0.6, 1.0])
    return DirectLabelColormap(color_dict=color_dict)


def focus_label(viewer: Any, layer: Any, key: int) -> None:
    """Focus the centre of a label's bounding box (cached in the layer metadata by the loader, else computed)."""
    box = (layer.metadata.get(BBOXES_KEY) or {}).get(int(key))
    if box is not None:
        centre = bbox_center(box)
    else:
        idx = np.argwhere(np.asarray(layer.data) == key)
        if not len(idx):
            return
        centre = (idx.min(axis=0) + idx.max(axis=0)) / 2.0
    focus_viewer(viewer, centre * np.asarray(layer.scale)[-3:])


def focus_viewer(viewer: Any, zyx: Iterable[float]) -> None:
    """Centre the camera on a world point and move the slice there."""
    zyx = [float(v) for v in zyx]
    try:
        n = viewer.dims.ndim
        if n >= 3:
            viewer.dims.set_point(n - 3, zyx[0])
            viewer.dims.set_point(n - 2, zyx[1])
            viewer.dims.set_point(n - 1, zyx[2])
        viewer.camera.center = tuple(zyx)
    except Exception as e:  # pragma: no cover - viewer state edge cases
        logger.warning(f"Could not focus viewer: {e}")


class InstanceAdapter:
    """Connects the instance browser to one layer. Subclasses implement rows, focus and visibility."""

    kind = ""
    title = "Instances"
    capabilities: Dict[str, bool] = {}

    def __init__(self, layer: Any, viewer: Any):
        self.layer = layer
        self.viewer = viewer

    def rows(self) -> List[InstanceRow]:
        return []

    def focus(self, key: int) -> None:
        pass

    def set_visible(self, keys: Set[int]) -> None:
        pass


class PointsAdapter(InstanceAdapter):
    """Picks (points with an ``instance_id`` feature)."""

    kind = "picks"
    title = "Pick instances"
    capabilities = {"color_toggle": True}

    def _ids(self) -> np.ndarray:
        f = self.layer.features
        if "instance_id" not in f.columns or len(f) != len(self.layer.data):
            return np.zeros(len(self.layer.data), dtype=np.int64)
        return np.nan_to_num(f["instance_id"].to_numpy(dtype=float)).astype(np.int64)

    def _base(self) -> np.ndarray:
        return np.asarray(self.layer.metadata.get("copick_base_color", (1.0, 1.0, 1.0, 1.0)), dtype=float)

    def rows(self) -> List[InstanceRow]:
        f = self.layer.features
        scores = f["score"].to_numpy(dtype=float) if "score" in f.columns and len(f) == len(self.layer.data) else None
        return rows_from_points(self._ids(), scores, tuple(self._base()))

    def focus(self, key: int) -> None:
        pts = np.asarray(self.layer.data)[self._ids() == key]
        if len(pts):
            focus_viewer(self.viewer, pts.mean(axis=0) * np.asarray(self.layer.scale)[-3:])

    def set_visible(self, keys: Set[int]) -> None:
        self.layer.shown = np.isin(self._ids(), list(keys))

    def set_color_by_instance(self, on: bool) -> None:
        ids = self._ids()
        base = self._base()
        self.layer.face_color = instance_colors(ids, base) if on else np.tile(base, (len(ids), 1))
        self.layer.metadata[COLOR_MODE_KEY] = "instance" if on else "object"


class FilamentsAdapter(InstanceAdapter):
    """A filament centreline layer (its session holds the filaments)."""

    kind = "filaments"
    title = "Filaments"
    capabilities = {
        "reverse": True,
        "delete": True,
        "new": True,
        "merge": True,
        "merge_tip": "Join the selected filaments end to end (into the current one)",
    }

    def __init__(self, layer: Any, viewer: Any):
        super().__init__(layer, viewer)
        from napari_copick.filament_layers import SESSION_KEY

        self.session = layer.metadata[SESSION_KEY]

    def rows(self) -> List[InstanceRow]:
        rows = rows_from_filaments(self.session.to_list())
        known = {r.instance_id for r in rows}
        for i, cps in self.session.pending.items():  # filaments still being started (< 2 control points)
            if i not in known:
                rows.append(InstanceRow(instance_id=i, count=len(cps), label="pending"))
        return sorted(rows, key=lambda r: r.instance_id)

    def focus(self, key: int) -> None:
        if key in self.session.filaments:
            xyz = point_at_arc_length(np.asarray(self.session.filaments[key].points), 0.5)
        else:
            cps = self.session.controls(key)
            if not len(cps):
                return
            xyz = cps.mean(axis=0)
        focus_viewer(self.viewer, xyz[::-1])

    def set_visible(self, keys: Set[int]) -> None:
        f = self.layer.features
        if "instance_id" in f.columns and len(f) == len(self.layer.data):
            self.layer.shown = np.isin(f["instance_id"].to_numpy(dtype=np.int64), list(keys))


class LabelsAdapter(InstanceAdapter):
    """An instance segmentation (labels layer of instance IDs)."""

    kind = "instance"
    title = "Instances"
    capabilities = {"new": True, "delete": True, "merge": True}

    def counts(self) -> Dict[int, int]:
        return self.layer.metadata.get(COUNTS_KEY, {})

    def rows(self) -> List[InstanceRow]:
        return rows_from_counts(self.counts(), label=self.layer.metadata.get("copick_source_object_name", ""))

    def focus(self, key: int) -> None:
        focus_label(self.viewer, self.layer, key)

    def set_visible(self, keys: Set[int]) -> None:
        ids = set(self.counts())
        hidden = ids - set(keys)
        self.layer.metadata[HIDDEN_KEY] = hidden
        self.layer.colormap = instance_colormap(ids, hidden)


class PanopticAdapter(InstanceAdapter):
    """The segment-index layer of a panoptic segmentation (read-only)."""

    kind = "panoptic"
    title = "Panoptic segments"

    def rows(self) -> List[InstanceRow]:
        return list(self.layer.metadata.get(PANOPTIC_ROWS_KEY, []))

    def focus(self, key: int) -> None:
        focus_label(self.viewer, self.layer, key)

    def set_visible(self, keys: Set[int]) -> None:
        rows = self.rows()
        hidden = {r.row_key for r in rows} - set(keys)
        self.layer.colormap = segment_colormap(rows, hidden)


def adapter_for(layer: Any, viewer: Any) -> Optional[InstanceAdapter]:
    """The browser adapter for a layer, or None if it has no instances."""
    if layer is None:
        return None
    meta = getattr(layer, "metadata", {}) or {}
    kind = meta.get("copick_kind")
    if kind == "filaments":
        return FilamentsAdapter(layer, viewer)
    if kind == "panoptic_segments":
        return PanopticAdapter(layer, viewer)
    if meta.get(SEGMENTATION_TYPE_KEY) == "instance":
        return LabelsAdapter(layer, viewer)
    if "copick_picks" in meta or kind == "picks":
        return PointsAdapter(layer, viewer)
    return None
