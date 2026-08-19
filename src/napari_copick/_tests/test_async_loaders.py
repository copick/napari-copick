import numpy as np
import pytest

from napari_copick.async_loaders import load_segmentation_worker, load_tomogram_worker


def exhaust(generator):
    messages = []
    while True:
        try:
            messages.append(next(generator))
        except StopIteration as stop:
            return messages, stop.value


def test_tomogram_worker_loads_selected_data_and_metadata(make_ome_store, fake_entity):
    store, arrays = make_ome_store(paths=("full", "overview"))
    tomogram = fake_entity(store)

    messages, result = exhaust(load_tomogram_worker.__wrapped__(tomogram, 1))

    assert messages == [
        "Opening zarr group for wbp...",
        "Loading resolution level 1...",
        "Loading image data into memory...",
    ]
    np.testing.assert_array_equal(result["data"], arrays[1])
    assert isinstance(result["data"], np.ndarray)
    assert result["voxel_size"] == (10.0, 12.0, 16.0)
    assert result["resolution_level"] == 1
    assert result["name"] == "[run-1] Tomogram: wbp (Level 1)"
    assert result["tomogram"] is tomogram


def test_segmentation_worker_clamps_and_loads_read_only(make_ome_store, fake_entity):
    store, arrays = make_ome_store(paths=("labels",), scales=((3.0, 3.0, 3.0),))
    segmentation = fake_entity(store, kind="segmentation", voxel_size=6.0)

    messages, result = exhaust(load_segmentation_worker.__wrapped__(segmentation, 2))

    assert messages == [
        "Opening zarr group for ribosome...",
        "Loading segmentation data from level 0...",
        "Loading segmentation data into memory...",
    ]
    np.testing.assert_array_equal(result["data"], arrays[0])
    assert result["voxel_size"] == (6.0, 6.0, 6.0)
    assert result["resolution_level"] == 0
    assert result["segmentation"] is segmentation


def test_segmentation_worker_preserves_helpful_storage_error(fake_entity):
    segmentation = fake_entity(None, kind="segmentation")

    with pytest.raises(ValueError, match="Error loading segmentation: .*no Zarr store"):
        exhaust(load_segmentation_worker.__wrapped__(segmentation, 0))
