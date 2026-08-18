from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore


@pytest.fixture
def make_ome_store():
    def factory(
        *,
        zarr_format=3,
        paths=("fine", "preview"),
        scales=((2.0, 4.0, 8.0), (5.0, 12.0, 32.0)),
        nested=True,
        include_distractor=True,
    ):
        store = MemoryStore()
        group = zarr.group(store=store, zarr_format=zarr_format)
        arrays = []
        datasets = []
        for index, (path, scale) in enumerate(zip(paths, scales, strict=True)):
            data = np.full((4 - index, 5 - index, 6 - index), index + 1, dtype=np.float32)
            group.create_array(path, data=data, chunks=data.shape)
            arrays.append(data)
            datasets.append(
                {
                    "path": path,
                    "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
                },
            )

        if include_distractor:
            group.create_array("distractor", data=np.zeros((1, 1, 1), dtype=np.float32))

        multiscales = [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "angstrom"},
                    {"name": "y", "type": "space", "unit": "angstrom"},
                    {"name": "x", "type": "space", "unit": "angstrom"},
                ],
                "datasets": datasets,
            },
        ]
        if nested:
            group.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}
        else:
            group.attrs["multiscales"] = multiscales
        return store, arrays

    return factory


class FakeEntity:
    def __init__(self, store, *, kind="tomogram", voxel_size=4.0):
        self._store = store
        self.run = SimpleNamespace(name="run-1")
        if kind == "tomogram":
            self.tomo_type = "wbp"
            self.meta = SimpleNamespace(tomo_type="wbp")
            self.voxel_spacing = SimpleNamespace(
                meta=SimpleNamespace(voxel_size=voxel_size),
                run=self.run,
            )
        else:
            self.name = "ribosome"
            self.user_id = "user"
            self.session_id = "session"
            self.is_multilabel = False
            self.voxel_size = voxel_size
            self.meta = SimpleNamespace(name=self.name, voxel_size=voxel_size)

    def zarr(self):
        return self._store


@pytest.fixture
def fake_entity():
    return FakeEntity
