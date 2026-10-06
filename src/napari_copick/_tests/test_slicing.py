"""Filaments and picks stay visible near the current slice on napari >= 0.9 (thick slices replace out_of_slice_display)."""

import numpy as np
import pytest
from copick.ops.open import new_config
from copick_shared_ui.util.filament_session import FilamentEditSession
from napari.components import ViewerModel

from napari_copick.filament_layers import add_centreline_layer, add_controls_layer
from napari_copick.slicing import THICK_SLICES, show_points_within, single_slice


@pytest.fixture
def session(tmp_path):
    root = new_config(str(tmp_path / "config.json"), str(tmp_path / "overlay"))
    root.new_object(name="microtubule", is_particle=True, radius=120, filament={"polar": True})
    run = root.new_run("r1")
    run.new_voxel_spacing(10.0)
    s = FilamentEditSession(run, "microtubule", "u", "1", step=10.0)
    for p in ([0, 300, 10], [0, 300, 400]):  # runs along z
        s.insert_point(p)
    return s


@pytest.fixture
def viewer():
    v = ViewerModel()
    v.add_image(np.zeros((50, 50, 50), dtype=np.float32), scale=(10.0, 10.0, 10.0))
    return v


def test_filament_crossing_the_slice_is_drawn(viewer, session):
    layer = add_centreline_layer(viewer, session)
    viewer.dims.set_point(0, 200.0)
    # the samples within the filament radius of the slice are drawn, not just one that happens to lie in it
    assert len(layer._view_data) > 3
    if THICK_SLICES:
        assert viewer.dims.thickness[0] >= 120.0
        assert layer.projection_mode == "rescale_linear"


def test_control_points_only_in_their_slice(viewer, session):
    layer = add_controls_layer(viewer, session)
    if THICK_SLICES:
        assert layer.projection_mode == "none"
    viewer.dims.set_point(0, 10.0)
    assert len(layer._view_data) == 1


@pytest.mark.skipif(not THICK_SLICES, reason="thick slices replace out_of_slice_display from napari 0.9")
def test_thickness_only_grows_and_tomograms_keep_one_slice(viewer):
    viewer.dims.thickness = (300.0, 0.0, 0.0)
    show_points_within(viewer, 120.0)
    assert viewer.dims.thickness[0] == 300.0
    image = viewer.layers[0]
    single_slice(image)
    assert image.projection_mode == "none"
