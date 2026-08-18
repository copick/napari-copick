from types import SimpleNamespace

import numpy as np

from napari_copick.data_loader import DataLoader


class FakeLabel:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


class FakeLayer:
    def __init__(self):
        self.metadata = {}
        self.contrast_was_reset = False

    def reset_contrast_limits(self):
        self.contrast_was_reset = True


class FakeViewer:
    def __init__(self):
        self.image_call = None
        self.labels_call = None
        self.image_layer = None
        self.labels_layer = None

    def add_image(self, data, **kwargs):
        self.image_call = (data, kwargs)
        self.image_layer = FakeLayer()
        return self.image_layer

    def add_labels(self, data, **kwargs):
        self.labels_call = (data, kwargs)
        self.labels_layer = FakeLayer()
        return self.labels_layer


class FakeTree:
    def __init__(self):
        self.removed = []

    def remove_loading_indicator(self, item):
        self.removed.append(item)


def make_parent(entity):
    viewer = FakeViewer()
    return SimpleNamespace(
        viewer=viewer,
        info_label=FakeLabel(),
        tree_view=FakeTree(),
        loading_items={entity: "tree-item"},
        loading_workers={},
        root=SimpleNamespace(
            pickable_objects=[SimpleNamespace(name="ribosome", label=7, color=(20, 40, 60, 255))],
        ),
        class_labels_mapping={},
        _remove_operation=lambda operation_id: None,
        get_copick_colormap=lambda: {},
    )


def test_tomogram_result_adds_scaled_image_with_metadata(fake_entity):
    tomogram = fake_entity(None)
    parent = make_parent(tomogram)
    loader = DataLoader(parent)
    data = np.ones((2, 3, 4), dtype=np.float32)

    loader._on_tomogram_loaded(
        {
            "tomogram": tomogram,
            "data": data,
            "voxel_size": (10.0, 12.0, 16.0),
            "name": "tomogram layer",
            "resolution_level": 1,
        },
    )

    loaded_data, kwargs = parent.viewer.image_call
    assert loaded_data is data
    assert kwargs == {"scale": (10.0, 12.0, 16.0), "name": "tomogram layer"}
    callback_layer = parent.viewer.image_layer
    assert callback_layer.contrast_was_reset is True
    assert callback_layer.metadata["copick_tomogram"] is tomogram
    assert callback_layer.metadata["copick_resolution_level"] == 1
    assert parent.info_label.text == "Loaded Tomogram: wbp (Resolution Level 1)"


def test_segmentation_result_adds_scaled_labels_and_colormap(fake_entity):
    segmentation = fake_entity(None, kind="segmentation")
    parent = make_parent(segmentation)
    loader = DataLoader(parent)
    data = np.ones((2, 3, 4), dtype=np.uint8)
    loader._on_segmentation_loaded(
        {
            "segmentation": segmentation,
            "data": data,
            "voxel_size": (8.0, 9.0, 10.0),
            "name": "segmentation layer",
            "resolution_level": 1,
        },
    )

    loaded_data, kwargs = parent.viewer.labels_call
    callback_layer = parent.viewer.labels_layer
    assert loaded_data is data
    assert kwargs == {"name": "segmentation layer", "scale": (8.0, 9.0, 10.0)}
    assert callback_layer.painting_labels == [1]
    assert callback_layer.metadata["copick_segmentation"] is segmentation
    assert callback_layer.metadata["copick_resolution_level"] == 1
    assert parent.class_labels_mapping == {1: "ribosome"}
    assert parent.info_label.text == "Loaded Segmentation: ribosome (Resolution Level 1)"
