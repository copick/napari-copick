"""Conversion between copick picks and napari points layers.

copick's particle centre is ``location + t``, with ``t`` the translation of the point's transform (tomogram frame,
Angstrom). A points layer has no separate shift, so it shows the centre, and keeps each loaded point's transform,
instance id and score so that saving gives them back.
"""

import colorsys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from copick.models import CopickLocation, CopickPoint

# Features of a point placed in napari: unassigned, full score, and no copick point it came from.
FEATURE_DEFAULTS = {"instance_id": 0, "score": 1.0, "copick_index": -1}

# Layer metadata key of the loaded points' (N, 4, 4) transforms; ``copick_index`` indexes into it.
TRANSFORMS_KEY = "copick_transforms"


def picks_to_layer_data(points: Optional[Sequence[CopickPoint]]) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Points layer data for copick points.

    Returns:
        (N, 3) particle centres ``location + t`` in z, y, x order (Angstrom), the features (``instance_id``,
        ``score``, ``copick_index``) and the (N, 4, 4) transforms.
    """
    points = list(points or [])
    transforms = np.array([np.asarray(p.transformation, dtype=float) for p in points]).reshape(-1, 4, 4)
    locations = np.array([[p.location.x, p.location.y, p.location.z] for p in points], dtype=float).reshape(-1, 3)
    centres = locations + transforms[:, :3, 3]
    features = pd.DataFrame(
        {
            "instance_id": np.array([_instance_id(p.instance_id) for p in points], dtype=np.int64),
            "score": np.array([_score(p.score) for p in points], dtype=float),
            "copick_index": np.arange(len(points), dtype=np.int64),
        },
    )
    return centres[:, ::-1].copy(), features, transforms


def layer_to_points(
    data: np.ndarray,
    scale: Sequence[float],
    features: Optional[pd.DataFrame],
    transforms: Optional[np.ndarray],
) -> List[CopickPoint]:
    """copick points for a points layer's data (z, y, x, in the layer's scale units).

    A point loaded from copick (``copick_index`` >= 0) keeps its transform, shift included, and its location is the
    displayed centre minus that shift: an unmoved point is unchanged, a moved one keeps its shift relative to where it
    now is. A point placed in napari gets the identity transform, so its location is the displayed position.
    """
    data = np.asarray(data, dtype=float).reshape(-1, 3)
    xyz = data[:, ::-1] * np.asarray(scale, dtype=float)[-3:][::-1]
    transforms = np.zeros((0, 4, 4)) if transforms is None else np.asarray(transforms, dtype=float).reshape(-1, 4, 4)
    n = len(xyz)
    instance_ids = _column(features, "instance_id", n)
    scores = _column(features, "score", n)
    indices = _column(features, "copick_index", n)

    points = []
    for i in range(n):
        index = _instance_id(indices[i]) if np.isfinite(indices[i]) else -1
        transform = transforms[index].copy() if 0 <= index < len(transforms) else np.eye(4)
        location = xyz[i] - transform[:3, 3]
        points.append(
            CopickPoint(
                location=CopickLocation(x=float(location[0]), y=float(location[1]), z=float(location[2])),
                transformation_=transform.tolist(),
                instance_id=_instance_id(instance_ids[i]) if np.isfinite(instance_ids[i]) else 0,
                score=_score(scores[i]),
            ),
        )
    return points


def reset_added_features(layer: Any, event: Any) -> None:
    """Give points added to ``layer`` the :data:`FEATURE_DEFAULTS`.

    napari copies the selected points' features into the defaults for the next point, so a point placed after
    selecting a loaded one would inherit its instance id, score and transform. Connect to ``layer.events.data``.
    """
    action = getattr(event, "action", None)
    if getattr(action, "value", action) != "added":
        return
    features = layer.features
    n = len(features)
    rows = sorted({int(i) % n for i in getattr(event, "data_indices", ()) if n and -n <= int(i) < n})
    if not rows:
        return
    for key, value in FEATURE_DEFAULTS.items():
        if key in features.columns:
            features.iloc[rows, features.columns.get_loc(key)] = value


def is_filament(obj: Any) -> bool:
    """Whether a pickable object is declared a filament.

    copick >= 1.28 has ``is_filament``; older copick carries the spec in ``metadata["copick"]["filament"]``.
    """
    flag = getattr(obj, "is_filament", None)
    if isinstance(flag, bool):
        return flag
    metadata = getattr(obj, "metadata", None)
    namespace = metadata.get("copick") if isinstance(metadata, dict) else None
    return isinstance(namespace, dict) and namespace.get("filament") is not None


def instance_colors(instance_ids: Sequence[int], base_rgba: Sequence[float]) -> np.ndarray:
    """(N, 4) float RGBA: each instance its own hue (golden-ratio steps), unassigned (0) points the object's colour."""
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    colors = np.tile(np.asarray(base_rgba, dtype=float)[:4], (len(ids), 1))
    for value in np.unique(ids[ids > 0]):
        colors[ids == value, :3] = colorsys.hsv_to_rgb((value * 0.618033988749895) % 1.0, 0.65, 0.95)
    return colors


def _column(features: Optional[pd.DataFrame], name: str, n: int) -> np.ndarray:
    if features is None or name not in features.columns or len(features) != n:
        return np.full(n, np.nan)
    return pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)


def _instance_id(value: Any) -> int:
    return 0 if value is None else int(value)


def _score(value: Any) -> float:
    return 1.0 if value is None or not np.isfinite(float(value)) else float(value)
