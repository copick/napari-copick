"""GUI smoke: open, paint, save, train against a service, inspect and accept, undo, in a real napari viewer."""

import numpy as np

from napari_copick.painting import Painting
from napari_copick.painting_widget import PaintingWidget


def test_paint_train_accept_undo(make_napari_viewer, qtbot, run, fake_cellcanvas):
    viewer = make_napari_viewer()
    tomogram = run.get_voxel_spacing(10.0).get_tomogram("denoised")
    image = viewer.add_image(np.zeros((8, 12, 16), np.float32), scale=(10.0,) * 3)
    image.metadata["copick_tomogram"] = tomogram
    widget = PaintingWidget(viewer)
    viewer.window.add_dock_widget(widget)

    widget.open_painting()
    layer = widget.painting_layer
    assert layer.data.shape == (8, 12, 16) and not layer.data.any()
    layer.data[0, 0, 0], layer.data[1, 1, 1] = 1, 2  # as napari's brush would
    widget.service_url.setText(fake_cellcanvas.url)
    widget.dataset.setText("demo")
    widget.proposal_name.setText("cellcanvas-prediction")
    widget.train_and_predict()
    qtbot.waitUntil(lambda: widget.proposal is not None and widget._worker is None, timeout=10000)
    assert fake_cellcanvas.uploads[-1]["values"] == [1, 2]
    assert widget.proposal_layer.data.max() == 2

    # Someone else saves the painting meanwhile (e.g. copick-web): accepting from the stale view is refused.
    other = Painting.open(run, 10.0, "denoised", "cellcanvas-painting", "napari", "1")
    labels = other.labels()
    labels[7, 11, 15] = 1
    other.save(labels, other.revision, source={"kind": "browser"})
    widget.proposal_label.setValue(2)
    widget.accept_label.setValue(2)
    assert widget.accept() is None and "changed" in widget.status.text()

    widget._reload()
    result = widget.accept()
    assert result["kept_human_voxels"] == 2  # the painted 2 and the browser-painted 1 are kept
    assert (layer.data == 2).sum() == 8 * 12 * 16 - 2 and layer.data[0, 0, 0] == 1 and layer.data[7, 11, 15] == 1
    widget.undo()
    assert (layer.data == 2).sum() == 1
    widget.undo()  # the browser edit is undone from napari too
    assert layer.data[7, 11, 15] == 0
