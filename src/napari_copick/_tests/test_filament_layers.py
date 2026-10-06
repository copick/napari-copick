"""Filament layers: the centreline and control layers follow a FilamentEditSession, and control-layer edits are
regenerated through copick core."""

from types import SimpleNamespace

import numpy as np
import pytest
from copick.ops.open import new_config
from copick_shared_ui.util.filament_session import FilamentEditSession

from napari_copick.filament_layers import (
    ControlLayerSync,
    centreline_layer_data,
    controls_from_layer,
    controls_layer_data,
)


@pytest.fixture
def session(tmp_path):
    root = new_config(str(tmp_path / "config.json"), str(tmp_path / "overlay"))
    root.new_object(name="microtubule", is_particle=True, radius=120, filament={"polar": True})
    run = root.new_run("r1")
    run.new_voxel_spacing(10.0)
    s = FilamentEditSession(run, "microtubule", "u", "1", step=10.0)
    for p in ([0, 0, 0], [100, 0, 0], [200, 50, 0]):
        s.insert_point(p)
    s.new_filament()
    for p in ([0, 300, 10], [0, 300, 200]):
        s.insert_point(p)
    return s


class FakePointsLayer:
    """Just enough of a napari points layer for ControlLayerSync."""

    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.face_color = None
        self.feature_defaults = {}
        self.selected_data = set()
        self.name = ""


def _event(action, indices=()):
    return SimpleNamespace(action=SimpleNamespace(value=action), data_indices=indices)


def test_layer_data_shapes_and_order(session):
    data, features, colors = centreline_layer_data(session)
    assert data.shape[1] == 3 and len(data) == len(features) == len(colors)
    assert set(features["instance_id"]) == {1, 2}
    # zyx order: filament 2 runs along z
    first2 = data[features["instance_id"].to_numpy() == 2]
    assert first2[0, 0] == pytest.approx(10.0) and first2[-1, 0] == pytest.approx(200.0)
    cdata, cfeat, _ = controls_layer_data(session)
    assert len(cdata) == 5 and list(cfeat["order"]) == [0, 1, 2, 0, 1]


def test_controls_from_layer_round_trip(session):
    cdata, cfeat, _ = controls_layer_data(session)
    layer = FakePointsLayer(cdata, cfeat)
    got = controls_from_layer(layer)
    assert np.allclose(got[1], session.controls(1)) and np.allclose(got[2], session.controls(2))


def test_sync_add_move_remove(session):
    cdata, cfeat, _ = controls_layer_data(session)
    controls = FakePointsLayer(cdata, cfeat)
    centre_data, centre_feat, _ = centreline_layer_data(session)
    centre = FakePointsLayer(centre_data, centre_feat)
    messages = []
    sync = ControlLayerSync(controls, centre, session, on_change=messages.append)

    # add a point (as napari does: appended row, defaults copied) -> goes to the active filament (2)
    session.active_id = 2
    controls.data = np.vstack([controls.data, [400, 300, 0]])  # zyx
    import pandas as pd

    controls.features = pd.concat(
        [controls.features, pd.DataFrame({"instance_id": [1], "order": [0]})],
        ignore_index=True,
    )
    sync.on_data_event(_event("added", (len(controls.data) - 1,)))
    assert len(session.controls(2)) == 3 and np.allclose(session.controls(2)[-1], [0, 300, 400])
    assert len(controls.data) == 6  # redrawn from the session

    # move filament 1's middle point
    rows = np.nonzero(controls.features["instance_id"].to_numpy() == 1)[0]
    moved = controls.data.copy()
    moved[rows[1]] = [0, 40, 100]
    controls.data = moved
    sync.on_data_event(_event("changed", (rows[1],)))
    assert np.allclose(session.controls(1)[1], [100, 40, 0])

    # remove filament 1's first point
    keep = np.ones(len(controls.data), dtype=bool)
    keep[rows[0]] = False
    controls.data = controls.data[keep]
    controls.features = controls.features[keep].reset_index(drop=True)
    sync.on_data_event(_event("removed", (rows[0],)))
    assert len(session.controls(1)) == 2
    assert messages[-1] == ""
    # "changing" (drag in progress) is ignored
    before = session.controls(1).copy()
    sync.on_data_event(_event("changing"))
    assert np.allclose(session.controls(1), before)
