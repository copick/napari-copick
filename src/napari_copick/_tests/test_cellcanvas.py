"""The CellCanvas client uploads a painting on its grid and publishes only digest-checked predictions."""

import numpy as np
import pytest

from napari_copick.cellcanvas import CellCanvasError, CellCanvasService
from napari_copick.painting import Painting


def painted(run):
    painting = Painting.open(run, 10.0, "denoised", "painting", "kyle", "1")
    labels = painting.labels()
    labels[0, 0, 0], labels[1, 1, 1] = 1, 2
    painting.save(labels, None)
    return painting


def test_train_predict_publish_accept(run, fake_cellcanvas):
    painting = painted(run)
    service = CellCanvasService(fake_cellcanvas.url, "demo")
    trained = service.train(painting)
    upload = fake_cellcanvas.uploads[-1]
    assert upload["revision"] == painting.revision == trained["painting_revision"]
    assert upload["values"] == [1, 2] and upload["translation"] == [0.0, 0.0, 0.0]
    prediction = service.predict()
    assert prediction.painting_revision == painting.revision
    proposal = service.publish(prediction, run, 10.0, "prediction", "cellcanvas", "1")
    record = proposal.record
    assert record["producer"] == "cellcanvas" and record["painting_revision"] == painting.revision
    result = painting.accept(painting.revision, proposal, 2, 2, record["digest"])
    assert result["kept_human_voxels"] == 1  # the human 2 stays; the human 1 is not in the class-2 mask
    assert (painting.labels() == 1).sum() == 1


def test_unpainted_and_mismatched_grids_are_refused(run, fake_cellcanvas):
    service = CellCanvasService(fake_cellcanvas.url, "demo")
    with pytest.raises(CellCanvasError) as error:
        service.train(Painting.open(run, 10.0, "denoised", "painting", "kyle", "1"))
    assert error.value.code == "no_painting"
    fake_cellcanvas.scale = [20.0, 20.0, 20.0]
    with pytest.raises(CellCanvasError) as error:
        service.train(painted(run))
    assert error.value.code == "geometry_mismatch"


def test_prediction_not_matching_its_record_is_not_published(run, fake_cellcanvas):
    service = CellCanvasService(fake_cellcanvas.url, "demo")
    service.train(painted(run))
    fake_cellcanvas.corrupt = True
    with pytest.raises(CellCanvasError) as error:
        service.predict()
    assert error.value.code == "not_complete"
    assert not run.get_segmentations(name="prediction")


def test_unreachable_service_is_reported(run):
    with pytest.raises(CellCanvasError) as error:
        CellCanvasService("http://127.0.0.1:9", "demo", timeout=2).describe()
    assert error.value.code == "unreachable"
