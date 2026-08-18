import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from napari_copick.storage import open_multiscale_level, spatial_scale


@pytest.mark.parametrize(
    ("zarr_format", "nested", "paths"),
    [
        (2, False, ("0", "1")),
        (2, False, ("fine", "coarse")),
        (3, True, ("0", "1")),
        (3, True, ("s0", "preview")),
    ],
)
def test_selects_metadata_declared_level(make_ome_store, fake_entity, zarr_format, nested, paths):
    store, arrays = make_ome_store(zarr_format=zarr_format, nested=nested, paths=paths)

    level = open_multiscale_level(fake_entity(store), 1)

    assert level.level == 1
    assert level.path == paths[1]
    np.testing.assert_array_equal(level.array[:], arrays[1])
    assert spatial_scale(level, 4.0) == (10.0, 12.0, 16.0)


def test_clamps_requested_level_to_last_dataset(make_ome_store, fake_entity):
    store, arrays = make_ome_store(paths=("only",), scales=((3.0, 3.0, 3.0),))

    level = open_multiscale_level(fake_entity(store), 2)

    assert level.level == 0
    assert level.path == "only"
    np.testing.assert_array_equal(level.array[:], arrays[0])
    assert spatial_scale(level, 7.5) == (7.5, 7.5, 7.5)


def test_opens_store_read_only(make_ome_store, fake_entity, monkeypatch):
    store, _ = make_ome_store()
    modes = []
    real_open_group = zarr.open_group

    def recording_open_group(*args, **kwargs):
        modes.append(kwargs.get("mode"))
        return real_open_group(*args, **kwargs)

    monkeypatch.setattr("napari_copick.storage.zarr.open_group", recording_open_group)

    open_multiscale_level(fake_entity(store, kind="segmentation"), 0)

    assert modes == ["r"]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "metadata is missing or invalid"),
        ({"ome": {"multiscales": []}}, "contains no entries"),
        ({"ome": {"multiscales": [{"axes": [], "datasets": []}]}}, "contains no datasets"),
    ],
)
def test_rejects_missing_or_empty_metadata(fake_entity, metadata, message):
    store = MemoryStore()
    group = zarr.group(store=store, zarr_format=3)
    group.attrs.update(metadata)

    with pytest.raises(ValueError, match=message):
        open_multiscale_level(fake_entity(store), 0)


def test_rejects_negative_level(make_ome_store, fake_entity):
    store, _ = make_ome_store()

    with pytest.raises(ValueError, match="cannot be negative"):
        open_multiscale_level(fake_entity(store), -1)


def test_reports_missing_declared_path(make_ome_store, fake_entity):
    store, _ = make_ome_store(paths=("fine",), scales=((1.0, 1.0, 1.0),))
    group = zarr.open_group(store=store, mode="r+")
    ome = dict(group.attrs["ome"])
    multiscales = list(ome["multiscales"])
    multiscales[0] = dict(multiscales[0])
    multiscales[0]["datasets"] = [
        {"path": "missing", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]},
    ]
    ome["multiscales"] = multiscales
    group.attrs["ome"] = ome

    with pytest.raises(ValueError, match="declared dataset path 'missing' does not exist"):
        open_multiscale_level(fake_entity(store), 2)


@pytest.mark.parametrize(
    ("axes", "first_scale", "selected_scale", "message"),
    [
        ([{"name": "z"}, {"name": "y"}], [1, 1], [2, 2], "exactly one 'x' axis"),
        ([{"name": "z"}, {"name": "z"}, {"name": "x"}], [1, 1, 1], [2, 2, 2], "exactly one 'z' axis"),
        ([{"name": "z"}, {"name": "y"}, {"name": "x"}], None, [2, 2, 2], "has no scale"),
        ([{"name": "z"}, {"name": "y"}, {"name": "x"}], [1, 1], [2, 2, 2], "2 values for 3 axes"),
        ([{"name": "z"}, {"name": "y"}, {"name": "x"}], [0, 1, 1], [2, 2, 2], "cannot be zero"),
    ],
)
def test_rejects_invalid_spatial_transform(
    make_ome_store,
    fake_entity,
    axes,
    first_scale,
    selected_scale,
    message,
):
    store, _ = make_ome_store()
    group = zarr.open_group(store=store, mode="r+")
    ome = dict(group.attrs["ome"])
    multiscale = dict(ome["multiscales"][0])
    multiscale["axes"] = axes
    datasets = [dict(dataset) for dataset in multiscale["datasets"]]
    datasets[0]["coordinateTransformations"] = [] if first_scale is None else [{"type": "scale", "scale": first_scale}]
    datasets[1]["coordinateTransformations"] = [{"type": "scale", "scale": selected_scale}]
    multiscale["datasets"] = datasets
    ome["multiscales"] = [multiscale]
    group.attrs["ome"] = ome
    level = open_multiscale_level(fake_entity(store), 1)

    with pytest.raises(ValueError, match=message):
        spatial_scale(level, 4.0)
