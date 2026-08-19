from types import SimpleNamespace

import numpy as np
import pytest

from napari_copick.async_loaders import save_segmentation_worker
from napari_copick.save_utils import get_tomogram_shape_at_level_0, save_segmentation_to_copick


class RecordingSegmentation:
    def __init__(self):
        self.calls = []

    def from_numpy(self, data, *, levels, dtype):
        self.calls.append((np.array(data), levels, dtype))


class RecordingRun:
    def __init__(self):
        self.name = "run-1"
        self.created = []

    def new_segmentation(self, **kwargs):
        segmentation = RecordingSegmentation()
        self.created.append((kwargs, segmentation))
        return segmentation


def exhaust(generator):
    messages = []
    while True:
        try:
            messages.append(next(generator))
        except StopIteration as stop:
            return messages, stop.value


def test_level_zero_shape_follows_metadata_path(make_ome_store, fake_entity):
    store, arrays = make_ome_store(paths=("full-resolution", "preview"))
    tomogram = fake_entity(store)
    voxel_spacing = SimpleNamespace(tomograms=[tomogram])

    assert get_tomogram_shape_at_level_0(SimpleNamespace(), voxel_spacing) == arrays[0].shape


def test_synchronous_save_scales_and_delegates(make_ome_store, fake_entity):
    store, arrays = make_ome_store(paths=("full-resolution",), scales=((1.0, 1.0, 1.0),))
    run = RecordingRun()
    voxel_spacing = SimpleNamespace(voxel_size=4.0, tomograms=[fake_entity(store)])
    layer = SimpleNamespace(data=np.array([[[0, 1], [2, 3]]], dtype=np.int16))
    messages = []

    success = save_segmentation_to_copick(
        {
            "layer": layer,
            "run": run,
            "voxel_spacing": voxel_spacing,
            "session_id": "session",
            "user_id": "user",
            "segmentation_name": "objects",
            "is_multilabel": True,
        },
        messages.append,
    )

    assert success is True
    assert messages == ["Saved segmentation 'objects' to run 'run-1'"]
    kwargs, segmentation = run.created[0]
    assert kwargs == {
        "voxel_size": 4.0,
        "name": "objects",
        "session_id": "session",
        "user_id": "user",
        "is_multilabel": True,
    }
    data, levels, dtype = segmentation.calls[0]
    assert data.shape == arrays[0].shape
    assert data.dtype == np.uint8
    assert set(np.unique(data)) == {0, 1, 2, 3}
    assert levels == 1
    assert dtype is np.uint8


@pytest.mark.parametrize(
    ("mode", "expected_values", "expected_session_ids", "expected_multilabel"),
    [
        ({}, {0, 2, 5}, ["session"], False),
        ({"is_multilabel": True}, {0, 2, 5}, ["session"], True),
        ({"convert_to_binary": True}, {0, 1}, ["session"], False),
        ({"split_instances": True}, {0, 1}, ["session-0", "session-1"], False),
    ],
)
def test_async_save_modes_delegate_expected_data(
    make_ome_store,
    fake_entity,
    mode,
    expected_values,
    expected_session_ids,
    expected_multilabel,
):
    store, _ = make_ome_store(paths=("full",), scales=((1.0, 1.0, 1.0),))
    run = RecordingRun()
    voxel_spacing = SimpleNamespace(voxel_size=4.0, tomograms=[fake_entity(store)])
    layer = SimpleNamespace(data=np.zeros((4, 5, 6), dtype=np.uint16))
    layer.data[0, 0, 0] = 2
    layer.data[1, 1, 1] = 5
    params = {
        "layer": layer,
        "run": run,
        "voxel_spacing": voxel_spacing,
        "session_id": "session",
        "user_id": "user",
        "segmentation_name": "objects",
        "exist_ok": True,
        **mode,
    }

    _, result = exhaust(save_segmentation_worker.__wrapped__(params))

    assert result["success"] is True
    assert [kwargs for kwargs, _ in run.created] == [
        {
            "voxel_size": 4.0,
            "name": "objects",
            "session_id": session_id,
            "user_id": "user",
            "is_multilabel": expected_multilabel,
            "exist_ok": True,
        }
        for session_id in expected_session_ids
    ]
    for _, segmentation in run.created:
        data, levels, dtype = segmentation.calls[0]
        assert set(np.unique(data)) <= expected_values
        assert data.dtype == np.uint8
        assert levels == 1
        assert dtype is np.uint8


def test_synchronous_save_reports_writer_failure(make_ome_store, fake_entity):
    store, _ = make_ome_store(paths=("full",), scales=((1.0, 1.0, 1.0),))
    voxel_spacing = SimpleNamespace(voxel_size=4.0, tomograms=[fake_entity(store)])
    messages = []

    class FailingRun(RecordingRun):
        def new_segmentation(self, **kwargs):
            raise RuntimeError("writer unavailable")

    success = save_segmentation_to_copick(
        {
            "layer": SimpleNamespace(data=np.zeros((4, 5, 6))),
            "run": FailingRun(),
            "voxel_spacing": voxel_spacing,
            "session_id": "session",
            "user_id": "user",
            "segmentation_name": "objects",
        },
        messages.append,
    )

    assert success is False
    assert messages == ["Error saving segmentation: writer unavailable"]
