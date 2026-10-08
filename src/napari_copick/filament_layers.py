"""napari layers for traced filaments.

A filament set is shown as a points layer of its regenerated centreline samples, drawn where the filament crosses
the current slice (see ``slicing``), which render as tubes in 3D. Editing adds a second points
layer holding the control points, with features ``instance_id`` and ``order``. Both layers share one
``FilamentEditSession`` (copick-shared-ui), which owns the curves; every edit of the control layer is translated into
session edits and both layers are redrawn from the session. A vectors layer of arrows along each filament shows its
direction (its point order, what "reverse" flips); it follows the centreline layer (refresh, visibility, removal).

Coordinates are Angstrom in z, y, x order, matching tomograms loaded with ``scale = voxel size``.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from copick_shared_ui.util.filament_session import FilamentEditSession
from copick_shared_ui.util.filaments import direction_markers
from copick_shared_ui.util.instances import instance_colors

from napari_copick.slicing import points_slicing, show_points_within, vectors_near_slice

logger = logging.getLogger(__name__)

SESSION_KEY = "copick_filament_session"
KIND_KEY = "copick_kind"
CENTRELINE_KIND = "filaments"
CONTROLS_KIND = "filament_controls"
DIRECTION_KIND = "filament_directions"
DIRECTION_LAYER_KEY = "copick_direction_layer"
DEFAULT_RADIUS = 50.0


def _base_rgba(session: FilamentEditSession) -> np.ndarray:
    obj = session.object
    color = getattr(obj, "color", None) or (255, 255, 255, 255)
    return np.asarray(color, dtype=float)[:4] / 255.0


def _radius(session: FilamentEditSession) -> float:
    return float(session.radius or DEFAULT_RADIUS)


def layer_name(session: FilamentEditSession, controls: bool = False) -> str:
    prefix = "Filament controls" if controls else "Filaments"
    return f"{prefix}: {session.object_name} ({session.user_id} | {session.session_id})"


def centreline_layer_data(session: FilamentEditSession) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """(N, 3) zyx samples of all centrelines, features (``instance_id``, ``order``) and (N, 4) colours."""
    lines = session.polylines()
    if not lines:
        empty = pd.DataFrame({"instance_id": np.zeros(0, dtype=np.int64), "order": np.zeros(0, dtype=np.int64)})
        return np.zeros((0, 3)), empty, np.zeros((0, 4))
    xyz = np.concatenate(list(lines.values()))
    ids = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in lines.items()])
    order = np.concatenate([np.arange(len(p), dtype=np.int64) for p in lines.values()])
    features = pd.DataFrame({"instance_id": ids, "order": order})
    return xyz[:, ::-1].copy(), features, instance_colors(ids, _base_rgba(session))


def controls_layer_data(session: FilamentEditSession) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """(N, 3) zyx control points of all filaments, features (``instance_id``, ``order``) and (N, 4) colours."""
    controls = {i: c for i, c in session.all_controls().items() if len(c)}
    if not controls:
        empty = pd.DataFrame({"instance_id": np.zeros(0, dtype=np.int64), "order": np.zeros(0, dtype=np.int64)})
        return np.zeros((0, 3)), empty, np.zeros((0, 4))
    xyz = np.concatenate(list(controls.values()))
    ids = np.concatenate([np.full(len(c), i, dtype=np.int64) for i, c in controls.items()])
    order = np.concatenate([np.arange(len(c), dtype=np.int64) for c in controls.values()])
    features = pd.DataFrame({"instance_id": ids, "order": order})
    return xyz[:, ::-1].copy(), features, instance_colors(ids, _base_rgba(session))


def add_centreline_layer(viewer: Any, session: FilamentEditSession, run: Any = None) -> Any:
    """Add the display layer of a filament set."""
    data, features, colors = centreline_layer_data(session)
    # Metadata goes in at construction: napari makes a new layer active before add_points returns.
    metadata = {
        KIND_KEY: CENTRELINE_KIND,
        SESSION_KEY: session,
        "copick_run": run or session.run,
        "copick_source_object_name": session.object_name,
        "copick_user_id": session.user_id,
        "copick_session_id": session.session_id,
    }
    layer = viewer.add_points(
        data,
        name=layer_name(session),
        size=_radius(session),
        face_color=colors if len(colors) else "white",
        border_width=0,
        features=features,
        opacity=0.8,
        metadata=metadata,
        **points_slicing(near=True),
    )
    show_points_within(viewer, _radius(session))
    layer.mode = "pan_zoom"
    _add_direction_layer(viewer, layer, session)
    return layer


def refresh_centreline_layer(layer: Any, session: FilamentEditSession) -> None:
    data, features, colors = centreline_layer_data(session)
    layer.data = data
    layer.features = features
    if len(colors):
        layer.face_color = colors
    layer.name = layer_name(session)
    directions = getattr(layer, "metadata", {}).get(DIRECTION_LAYER_KEY)
    if directions is not None:
        vectors, vcolors = direction_layer_data(session)
        directions.data = vectors
        if len(vcolors):
            directions.edge_color = vcolors


def direction_layer_data(session: FilamentEditSession) -> Tuple[np.ndarray, np.ndarray]:
    """(K, 2, 3) zyx arrows (position, vector) along every filament, and their (K, 4) instance colours."""
    radius = _radius(session)
    spacing = max(16.0 * radius, 40.0 * session.step)
    length = 4.0 * radius
    vectors, ids = [], []
    for instance_id, line in session.polylines().items():
        pos, d = direction_markers(line, spacing)
        if len(pos):
            vectors.append(np.stack([pos[:, ::-1], d[:, ::-1] * length], axis=1))
            ids.append(np.full(len(pos), instance_id, dtype=np.int64))
    if not vectors:
        return np.zeros((0, 2, 3)), np.zeros((0, 4))
    ids = np.concatenate(ids)
    return np.concatenate(vectors), instance_colors(ids, _base_rgba(session))


def _add_direction_layer(viewer: Any, centreline: Any, session: FilamentEditSession) -> Any:
    """Arrows showing the filaments' direction, kept in step with the centreline layer."""
    vectors, colors = direction_layer_data(session)
    directions = viewer.add_vectors(
        vectors,
        name=f"Directions: {session.object_name} ({session.user_id} | {session.session_id})",
        edge_color=colors if len(colors) else "white",
        edge_width=1.2 * _radius(session),
        length=1.0,
        vector_style="arrow",
        opacity=1.0,
        blending="translucent_no_depth",  # drawn over the centreline samples, which would hide them in 3D
        metadata={KIND_KEY: DIRECTION_KIND, SESSION_KEY: session},
    )
    vectors_near_slice(directions)  # arrows show in the slices they pass through, like the centreline samples
    centreline.metadata[DIRECTION_LAYER_KEY] = directions
    viewer.layers.selection.active = centreline  # the centreline layer stays the one the Annotate tab follows

    def on_visible(event=None):
        directions.visible = centreline.visible

    def on_removed(event):
        if event.value is centreline and directions in viewer.layers:
            viewer.layers.remove(directions)
            viewer.layers.events.removed.disconnect(on_removed)

    centreline.events.visible.connect(on_visible)
    viewer.layers.events.removed.connect(on_removed)
    return directions


def add_controls_layer(viewer: Any, session: FilamentEditSession) -> Any:
    """Add the editable control-point layer of a filament set."""
    data, features, colors = controls_layer_data(session)
    layer = viewer.add_points(
        data,
        name=layer_name(session, controls=True),
        size=max(_radius(session) * 0.6, 2 * session.step),
        face_color=colors if len(colors) else "white",
        border_color="white",
        border_width=0.15,
        features=features,
        feature_defaults={"instance_id": session.active_id, "order": -1},
        metadata={KIND_KEY: CONTROLS_KIND, SESSION_KEY: session, "copick_run": session.run},
        **points_slicing(near=False),  # control points only in the slice they were placed in
    )
    return layer


def refresh_controls_layer(layer: Any, session: FilamentEditSession) -> None:
    data, features, colors = controls_layer_data(session)
    layer.data = data
    layer.features = features
    if len(colors):
        layer.face_color = colors
    layer.feature_defaults = {"instance_id": session.active_id, "order": -1}
    layer.selected_data = set()


def controls_from_layer(layer: Any) -> Dict[int, np.ndarray]:
    """Control points per filament from a control layer: ``{id: (n, 3) xyz}`` ordered by the ``order`` feature."""
    data = np.asarray(layer.data, dtype=float).reshape(-1, 3)
    if len(data) == 0:
        return {}
    features = layer.features
    ids = pd.to_numeric(features["instance_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    order = pd.to_numeric(features["order"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    out = {}
    for i in np.unique(ids):
        rows = np.nonzero(ids == i)[0]
        rows = rows[np.argsort(order[rows], kind="stable")]
        out[int(i)] = data[rows][:, ::-1]
    return out


class ControlLayerSync:
    """Keeps a control layer, its session and the centreline layer in step.

    Connect ``on_data_event`` to ``controls_layer.events.data``. ``on_change(message)`` is called after every applied
    edit (and with an error message when an edit was refused).
    """

    def __init__(
        self,
        controls_layer: Any,
        centreline_layer: Any,
        session: FilamentEditSession,
        on_change: Optional[Callable[[str], None]] = None,
    ):
        self.controls_layer = controls_layer
        self.centreline_layer = centreline_layer
        self.session = session
        self.on_change = on_change or (lambda msg: None)
        self._busy = False

    def redraw(self) -> None:
        self._busy = True
        try:
            refresh_controls_layer(self.controls_layer, self.session)
            if self.centreline_layer is not None:
                refresh_centreline_layer(self.centreline_layer, self.session)
        finally:
            self._busy = False

    def on_data_event(self, event: Any) -> None:
        if self._busy:
            return
        action = getattr(event, "action", None)
        action = getattr(action, "value", action)
        if action not in ("added", "changed", "removed"):
            return  # ignore "adding" / "changing" (drags) / "removing"
        try:
            if action == "added":
                self._apply_added(getattr(event, "data_indices", ()))
            else:
                self._apply_layer_state()
            message = ""
        except ValueError as e:
            message = str(e)
            logger.warning(message)
        self.redraw()
        self.on_change(message)

    def _apply_added(self, indices) -> None:
        data = np.asarray(self.controls_layer.data, dtype=float).reshape(-1, 3)
        n = len(data)
        rows = sorted({int(i) % n for i in indices if n and -n <= int(i) < n}) or ([n - 1] if n else [])
        for r in rows:
            self.session.insert_point(data[r][::-1])

    def _apply_layer_state(self) -> None:
        current = controls_from_layer(self.controls_layer)
        for instance_id in set(current) | set(self.session.ids()):
            new = current.get(instance_id, np.zeros((0, 3)))
            old = self.session.controls(instance_id)
            if len(new) == len(old) and np.allclose(new, old):
                continue
            if len(new) != len(old) and not self.session.can_add_remove(instance_id):
                raise ValueError(
                    f"Filament {instance_id} is a B-spline fit: its control points can be moved but not added or "
                    "removed. Use 'Convert to Catmull-Rom' first.",
                )
            self.session.set_controls(instance_id, new)
