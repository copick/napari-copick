"""Paintings and proposals keep the storage rules copick-web relies on: revisions, shared undo, complete-only
proposals and acceptance into unknown voxels only."""

import numpy as np
import pytest
import zarr

from napari_copick._tests.conftest import SHAPE
from napari_copick.painting import Painting, PaintingError, Proposal, label_digest


def open_painting(run, create=True):
    return Painting.open(run, 10.0, "denoised", "painting", "kyle", "1", create=create)


def test_new_painting_is_unknown_on_the_reconstruction_grid(run):
    painting = open_painting(run)
    assert painting.revision is None
    assert painting.labels().shape == SHAPE and not painting.labels().any()
    assert painting.grid == {"shape": list(SHAPE), "scale_angstrom": [10.0] * 3, "translation_angstrom": [0.0] * 3}


def test_save_is_revision_checked_and_undoable(run):
    painting = open_painting(run)
    labels = painting.labels()
    labels[2, 3:5, 4:6] = 2
    first = painting.save(labels, None)
    assert first["changed_voxels"] == 4 and first["revision"]
    with pytest.raises(PaintingError) as stale:
        painting.save(labels, None)
    assert stale.value.code == "revision_conflict"
    labels[1, 0, 0] = 1
    second = painting.save(labels, first["revision"])
    undone = painting.undo(second["revision"])
    assert undone["changed_voxels"] == 1 and undone["undone"] == {"kind": "napari"}
    assert (painting.labels() == 2).sum() == 4 and not (painting.labels() == 1).any()


def test_outside_write_blocks_undo_instead_of_rolling_it_back(run):
    painting = open_painting(run)
    labels = painting.labels()
    labels[0, 0, 0] = 2
    revision = painting.save(labels, None)["revision"]
    zarr.open_group(painting.segmentation.zarr(), mode="r+")["0"][0, 0, 1] = 1
    with pytest.raises(PaintingError, match="outside"):
        painting.undo(revision)


def test_wrong_shape_is_refused(run):
    painting = open_painting(run)
    with pytest.raises(PaintingError) as error:
        painting.save(np.zeros((4, 4, 4), np.uint8), None)
    assert error.value.code == "geometry_mismatch"


def test_accept_fills_unknown_voxels_only_and_is_undoable(run):
    painting = open_painting(run)
    labels = painting.labels()
    labels[0, 0, :4] = 1  # human background inside the proposed region
    revision = painting.save(labels, None)["revision"]
    predicted = np.zeros(SHAPE, np.uint8)
    predicted[0, 0, :] = 2
    proposal = Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", predicted, "cellcanvas")
    result = painting.accept(revision, proposal, 2, 2, proposal.record["digest"])
    assert result["changed_voxels"] == SHAPE[2] - 4 and result["kept_human_voxels"] == 4
    stored = painting.labels()
    assert (stored[0, 0, :4] == 1).all() and (stored[0, 0, 4:] == 2).all()
    assert result["source"]["producer"] == "cellcanvas"
    painting.undo(result["revision"])
    assert (painting.labels() == 2).sum() == 0


def test_accept_box_limits_the_region(run):
    painting = open_painting(run)
    predicted = np.full(SHAPE, 2, np.uint8)
    proposal = Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", predicted, "cellcanvas")
    result = painting.accept(None, proposal, 2, 2, proposal.record["digest"], box=[[3, 4], [0, 99], [0, 99]])
    assert result["changed_voxels"] == SHAPE[1] * SHAPE[2]


def test_incomplete_or_changed_proposals_are_refused(run):
    painting = open_painting(run)
    predicted = np.full(SHAPE, 2, np.uint8)
    proposal = Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", predicted, "cellcanvas")
    inspected = proposal.record["digest"]
    group = zarr.open_group(proposal.segmentation.zarr(), mode="r+")
    group["0"][0, 0, 0] = 1  # a later, unfinished write
    with pytest.raises(PaintingError) as error:
        painting.accept(None, proposal, 2, 2, inspected)
    assert error.value.code == "not_complete"
    Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", predicted * 0 + 1, "cellcanvas")
    with pytest.raises(PaintingError) as error:
        painting.accept(None, proposal, 1, 1, inspected)
    assert error.value.code == "revision_conflict"
    del group.attrs["proposal"]
    with pytest.raises(PaintingError) as error:
        painting.accept(None, proposal, 1, 1, inspected)
    assert error.value.code == "not_complete"


def test_paintings_and_proposals_stay_separate(run):
    painting = open_painting(run)
    with pytest.raises(PaintingError):
        Proposal.publish(run, 10.0, "painting", "kyle", "1", np.ones(SHAPE, np.uint8), "cellcanvas")
    Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", np.ones(SHAPE, np.uint8), "cellcanvas")
    with pytest.raises(PaintingError):
        Painting.open(run, 10.0, "denoised", "prediction", "cellcanvas", "1")
    assert painting.labels().sum() == 0


def test_digest_matches_stored_bytes(run):
    labels = np.arange(np.prod(SHAPE), dtype=np.uint8).reshape(SHAPE)
    proposal = Proposal.publish(run, 10.0, "prediction", "cellcanvas", "1", labels, "cellcanvas")
    assert proposal.record["digest"] == label_digest(labels)


def test_output_written_by_another_tool_is_accepted_once_marked_complete(run):
    painting = open_painting(run)
    # e.g. an nnInteractive object published into copick by its server
    segmentation = run.new_segmentation(
        voxel_size=10.0,
        name="nninteractive",
        session_id="1",
        is_multilabel=True,
        user_id="nninteractive",
    )
    labels = np.zeros(SHAPE, np.uint8)
    labels[2:4, 2:4, 2:4] = 2
    segmentation.from_numpy(labels, levels=1, dtype=np.uint8)
    proposal = Proposal(segmentation)
    with pytest.raises(PaintingError) as error:
        painting.accept(None, proposal, 2, 2, "unknown")
    assert error.value.code == "not_complete"
    proposal = Proposal.mark_complete(segmentation, "nninteractive", revision="sha256:reported")
    assert proposal.record["revision"] == "sha256:reported"
    result = painting.accept(None, proposal, 2, 2, proposal.record["digest"])
    assert result["changed_voxels"] == 8 and result["source"]["producer"] == "nninteractive"
    with pytest.raises(PaintingError):
        Proposal.mark_complete(painting.segmentation, "nninteractive")
