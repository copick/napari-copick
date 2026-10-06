"""Picks keep their shift, instance id and score through a points layer."""

from types import SimpleNamespace

import numpy as np
import pytest
from copick.models import CopickLocation, CopickPoint, PickableObject
from copick.ops.open import new_config

from napari_copick.pick_layers import (
    FEATURE_DEFAULTS,
    TRANSFORMS_KEY,
    is_filament,
    layer_to_points,
    picks_to_layer_data,
    reset_added_features,
)
from napari_copick.save_utils import save_picks_to_copick


def _rotation_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _point(location, shift=(0.0, 0.0, 0.0), deg=0.0, instance_id=0, score=1.0):
    transform = np.eye(4)
    transform[:3, :3] = _rotation_z(deg)
    transform[:3, 3] = shift
    return CopickPoint(
        location=CopickLocation(x=location[0], y=location[1], z=location[2]),
        transformation_=transform.tolist(),
        instance_id=instance_id,
        score=score,
    )


@pytest.fixture
def run(tmp_path):
    root = new_config(
        str(tmp_path / "config.json"),
        str(tmp_path / "overlay"),
        pickable_objects=[
            PickableObject(name="ribosome", is_particle=True, label=1, color=[255, 0, 0, 255], radius=100),
            PickableObject(
                name="microtubule",
                is_particle=True,
                label=2,
                color=[0, 255, 0, 255],
                metadata={"copick": {"filament": {"polar": True}}},
            ),
        ],
    )
    return root.new_run("TS-1")


POINTS = [
    _point((100.0, 200.0, 300.0), shift=(4.0, -6.0, 2.5), deg=30, instance_id=3, score=0.5),
    _point((10.0, 20.0, 30.0), deg=90, instance_id=4, score=0.75),
]


def test_layer_shows_particle_centres():
    data, features, transforms = picks_to_layer_data(POINTS)
    assert np.allclose(data[0], [302.5, 194.0, 104.0])  # z, y, x of location + t
    assert np.allclose(data[1], [30.0, 20.0, 10.0])
    assert features["instance_id"].tolist() == [3, 4]
    assert features["score"].tolist() == [0.5, 0.75]
    assert features["copick_index"].tolist() == [0, 1]
    assert np.allclose(transforms[0], POINTS[0].transformation)


def test_save_keeps_transforms_ids_and_scores(run):
    data, features, transforms = picks_to_layer_data(POINTS)
    data = np.vstack([data, [[5.0, 6.0, 7.0]]])  # a point placed in napari
    features.loc[2] = FEATURE_DEFAULTS
    data[1] += [0.0, 0.0, 10.0]  # the second point moved 10 A along x
    layer = SimpleNamespace(data=data, scale=(1.0, 1.0, 1.0), features=features, metadata={TRANSFORMS_KEY: transforms})

    assert save_picks_to_copick(
        {"layer": layer, "run": run, "object_name": "ribosome", "session_id": "1", "user_id": "napari"},
    )
    saved = run.get_picks(object_name="ribosome", user_id="napari", session_id="1")[0].points
    assert saved[0].location == POINTS[0].location
    assert np.allclose(saved[0].transformation, POINTS[0].transformation), "t and rotation survive"
    assert (saved[0].instance_id, saved[0].score) == (3, 0.5)
    assert (saved[1].location.x, saved[1].location.y) == (20.0, 20.0)
    assert np.allclose(saved[1].transformation, POINTS[1].transformation)
    assert (saved[2].location.x, saved[2].location.y, saved[2].location.z) == (7.0, 6.0, 5.0)
    assert np.allclose(saved[2].transformation, np.eye(4))
    assert (saved[2].instance_id, saved[2].score) == (0, 1.0)


def test_a_layer_without_copick_features_saves_new_picks():
    points = layer_to_points(np.array([[1.0, 2.0, 3.0]]), (10.0, 10.0, 10.0), None, None)
    assert (points[0].location.x, points[0].location.y, points[0].location.z) == (30.0, 20.0, 10.0)
    assert np.allclose(points[0].transformation, np.eye(4))
    assert (points[0].instance_id, points[0].score) == (0, 1.0)


def test_added_points_do_not_inherit_the_selection():
    from napari.layers import Points

    data, features, _ = picks_to_layer_data(POINTS)
    layer = Points(data, features=features, feature_defaults=FEATURE_DEFAULTS)
    layer.events.data.connect(lambda event: reset_added_features(layer, event))
    layer.selected_data = {0}  # napari copies point 0's features into the defaults
    layer.add([[1.0, 1.0, 1.0]])
    assert layer.features.iloc[2].to_dict() == FEATURE_DEFAULTS
    assert layer.features["copick_index"].tolist() == [0, 1, -1]


def test_load_draws_centres_and_saves_back_unchanged(make_napari_viewer, run):
    from napari_copick.data_loader import DataLoader

    picks = run.new_picks(object_name="microtubule", user_id="tracer", session_id="1")
    picks.points = POINTS
    picks.store()
    viewer = make_napari_viewer()
    parent = SimpleNamespace(viewer=viewer, root=run.root, info_label=SimpleNamespace(setText=lambda _text: None))
    DataLoader(parent).load_picks(picks, run)

    layer = viewer.layers[-1]
    assert np.allclose(layer.data[0], [302.5, 194.0, 104.0])
    assert layer.features["instance_id"].tolist() == [3, 4]
    assert not np.allclose(layer.face_color[0], layer.face_color[1]), "filaments are coloured by instance"

    assert save_picks_to_copick(
        {"layer": layer, "run": run, "object_name": "microtubule", "session_id": "2", "user_id": "tracer"},
    )
    saved = run.get_picks(object_name="microtubule", user_id="tracer", session_id="2")[0].points
    for got, expected in zip(saved, POINTS):
        assert got.location == expected.location
        assert np.allclose(got.transformation, expected.transformation)
        assert (got.instance_id, got.score) == (expected.instance_id, expected.score)


def test_is_filament_without_the_copick_property():
    assert is_filament(SimpleNamespace(metadata={"copick": {"filament": {}}}))
    assert not is_filament(SimpleNamespace(metadata={"copick": {"filament": None}}))
    assert not is_filament(SimpleNamespace(metadata={}))
