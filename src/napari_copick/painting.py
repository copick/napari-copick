"""Paintings and model proposals stored in copick, in the same form copick-web reads and writes.

A *painting* is a ``uint8`` multilabel copick segmentation on a reconstruction's full-resolution grid. ``0`` is
unknown (never annotated); positive values are classes, so background is only ever an explicit class. Every saved
edit gets a new ``attrs["painting"]["revision"]`` and keeps the voxels it replaced under ``_painting_undo``, so an
edit made in napari can be undone in the browser and vice versa, and a save based on a stale revision is refused.

A *proposal* is model output in its own segmentation. It is marked complete by ``attrs["proposal"]`` (producer and
digest of the stored labels), written only after the labels. It can be accepted into a painting only while its
labels still match that digest, and accepting fills only unknown voxels: human labels are never overwritten.

Example::

    painting = Painting.open(run, 10.0, "denoised", "painting", "kyle", "1")
    labels = painting.labels()
    labels[20, 30:40, 30:40] = 2
    painting.save(labels, painting.revision)

    proposal = Proposal.open(run, 10.0, "cellcanvas-prediction", "cellcanvas", "1")
    painting.accept(painting.revision, proposal, proposal_label=2, label=2, digest=proposal.record["digest"])
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional

import numpy as np
import zarr

PAINTING = "painting"
PROPOSAL = "proposal"
UNDO_GROUP = "_painting_undo"
MAX_UNDO_GESTURES = 20
UNIT_ANGSTROM = {"angstrom": 1.0, "ångström": 1.0, "nanometer": 10.0, "nm": 10.0}


class PaintingError(RuntimeError):
    """A refused painting or proposal operation; ``code`` says why (e.g. ``revision_conflict``)."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def label_digest(labels: np.ndarray) -> str:
    """Digest of stored label bytes, as recorded in ``attrs["proposal"]`` and in undo entries."""
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


def reconstruction_grid(tomogram) -> dict:
    """Shape, voxel size and origin (z, y, x; angstrom) of a copick tomogram's full-resolution level."""
    group = zarr.open_group(tomogram.zarr(), mode="r")
    multiscales = (group.attrs.get("multiscales") or [{}])[0]
    units = {a.get("unit", "angstrom") for a in multiscales.get("axes", []) if a.get("type") == "space"}
    units = units or {"angstrom"}
    if len(units) != 1 or next(iter(units)) not in UNIT_ANGSTROM:
        raise PaintingError(f"Unsupported tomogram units {sorted(units)}", "geometry_mismatch")
    per_unit = UNIT_ANGSTROM[next(iter(units))]
    scale, translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    for transform in (multiscales.get("datasets") or [{}])[0].get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = [float(v) for v in transform["scale"][-3:]]
        elif transform.get("type") == "translation":
            translation = [float(v) for v in transform["translation"][-3:]]
    return {
        "shape": list(group["0"].shape),
        "scale_angstrom": [v * per_unit for v in scale],
        "translation_angstrom": [v * per_unit for v in translation],
    }


def _segmentation(run, voxel_size: float, name: str, user_id: str, session_id: str):
    for refresh in (False, True):
        if refresh:
            run.refresh_segmentations()
        found = run.get_segmentations(
            name=name,
            user_id=user_id,
            session_id=session_id,
            voxel_size=voxel_size,
            is_multilabel=True,
        )
        if found:
            return found[0]
    return None


class Proposal:
    """Completed model output stored as a copick multilabel segmentation."""

    def __init__(self, segmentation):
        self.segmentation = segmentation

    @classmethod
    def open(cls, run, voxel_size: float, name: str, user_id: str, session_id: str) -> "Proposal":
        segmentation = _segmentation(run, voxel_size, name, user_id, session_id)
        if segmentation is None:
            raise PaintingError(f"Proposal '{name}' ({user_id}/{session_id}) not found.", "not_found")
        return cls(segmentation)

    @classmethod
    def publish(
        cls,
        run,
        voxel_size: float,
        name: str,
        user_id: str,
        session_id: str,
        labels: np.ndarray,
        producer: str,
        **details,
    ) -> "Proposal":
        """Write ``labels`` and only then mark them a completed proposal of ``producer``.

        Any previous completion record is removed before writing, so a reader never takes a partial write as
        complete. The record is not crash-atomic with the labels: an interrupted write leaves an incomplete
        proposal that acceptance refuses.
        """
        labels = np.asarray(labels)
        if labels.ndim != 3 or (labels.size and (labels.min() < 0 or labels.max() > 255)):
            raise PaintingError("Proposal labels must be a 3D array of classes 0-255.", "invalid_request")
        segmentation = _segmentation(run, voxel_size, name, user_id, session_id)
        if segmentation is None:
            segmentation = run.new_segmentation(
                voxel_size=voxel_size,
                name=name,
                session_id=session_id,
                is_multilabel=True,
                user_id=user_id,
            )
        else:
            group = zarr.open_group(segmentation.zarr(), mode="r+")
            if PAINTING in group.attrs:
                raise PaintingError("Refusing to overwrite a painting with model output.", "invalid_request")
            group.attrs.pop(PROPOSAL, None)
        segmentation.from_numpy(labels.astype(np.uint8), levels=1, dtype=np.uint8)
        return cls.mark_complete(segmentation, producer, **details)

    @classmethod
    def mark_complete(cls, segmentation, producer: str, **details) -> "Proposal":
        """Record the stored labels of ``segmentation`` as a completed proposal of ``producer``.

        For output another tool wrote into copick (for example an nnInteractive object published through
        ``RemoteInferenceClient.publish_to_copick``): call only after that tool reported its write finished, and
        pass what it reported (such as its read-back revision) as ``details``.
        """
        group = zarr.open_group(segmentation.zarr(), mode="r+")
        if PAINTING in group.attrs:
            raise PaintingError("A painting is human annotation, not a proposal.", "invalid_request")
        group.attrs[PROPOSAL] = {
            "complete": True,
            "producer": producer,
            "digest": label_digest(np.asarray(group["0"][:])),
            **details,
        }
        return cls(segmentation)

    def _group(self):
        return zarr.open_group(self.segmentation.zarr(), mode="r")

    @property
    def record(self) -> dict:
        """The completion record; refused if there is none (the proposal may still be written)."""
        record = dict(self._group().attrs.get(PROPOSAL) or {})
        if not record.get("complete"):
            raise PaintingError("This segmentation has no completed-proposal record.", "not_complete")
        return record

    def labels(self) -> np.ndarray:
        """Stored labels, refused unless they still match the completion record."""
        record = self.record
        labels = np.asarray(self._group()["0"][:])
        if label_digest(labels) != record.get("digest"):
            raise PaintingError(
                "Proposal labels differ from its completion record; it may be mid-write.",
                "not_complete",
            )
        return labels


class Painting:
    """A human painting of one reconstruction, shared with copick-web through copick storage."""

    def __init__(self, segmentation, tomogram):
        self.segmentation = segmentation
        self.tomogram = tomogram

    @classmethod
    def open(
        cls,
        run,
        voxel_size: float,
        tomo_type: str,
        name: str,
        user_id: str,
        session_id: str,
        create: bool = True,
    ) -> "Painting":
        """Open (or create, all unknown) the painting of ``run``'s ``tomo_type`` reconstruction."""
        spacing = run.get_voxel_spacing(voxel_size)
        tomogram = spacing.get_tomogram(tomo_type) if spacing is not None else None
        if tomogram is None:
            raise PaintingError("Reconstruction not found in the copick project.", "not_found")
        segmentation = _segmentation(run, voxel_size, name, user_id, session_id)
        if segmentation is None:
            if not create:
                raise PaintingError(f"Painting '{name}' ({user_id}/{session_id}) not found.", "not_found")
            shape = reconstruction_grid(tomogram)["shape"]
            segmentation = run.new_segmentation(
                voxel_size=voxel_size,
                name=name,
                session_id=session_id,
                is_multilabel=True,
                user_id=user_id,
                exist_ok=True,
            )
            segmentation.from_numpy(np.zeros(shape, dtype=np.uint8), levels=1, dtype=np.uint8)
            zarr.open_group(segmentation.zarr(), mode="r+").attrs[PAINTING] = {"revision": None, "undo": []}
        group = zarr.open_group(segmentation.zarr(), mode="r")
        if PROPOSAL in group.attrs:
            raise PaintingError(
                "This segmentation is model output; paint into a separate segmentation.",
                "invalid_request",
            )
        return cls(segmentation, tomogram)

    # state ----------------------------------------------------------------------------------------------------

    def _group(self, mode: str = "r"):
        return zarr.open_group(self.segmentation.zarr(), mode=mode)

    @staticmethod
    def _meta(group) -> dict:
        if PAINTING in group.attrs:
            return dict(group.attrs[PAINTING])
        legacy = group.attrs.get("cellcanvas") or {}  # paintings saved before the shared format
        return {"revision": legacy.get("revision"), "undo": []}

    @property
    def grid(self) -> dict:
        return reconstruction_grid(self.tomogram)

    @property
    def revision(self) -> Optional[str]:
        """Changes on every saved edit, from any tool; ``None`` until the first edit."""
        return self._meta(self._group()).get("revision")

    @property
    def can_undo(self) -> bool:
        return bool(self._meta(self._group()).get("undo"))

    def labels(self) -> np.ndarray:
        return np.asarray(self._group()["0"][:])

    # edits ----------------------------------------------------------------------------------------------------

    def save(self, labels: np.ndarray, base_revision: Optional[str], source: Optional[dict] = None) -> dict:
        """Store ``labels`` as one undoable edit, refused if the painting changed since ``base_revision``."""
        labels = np.asarray(labels)
        if labels.size and (labels.min() < 0 or labels.max() > 255):
            raise PaintingError("Painting classes must be 0 (unknown) to 255.", "invalid_request")
        return self._edit(base_revision, lambda _: labels.astype(np.uint8), source or {"kind": "napari"})

    def _edit(self, base_revision: Optional[str], change, source: Optional[dict]) -> dict:
        group = self._group("r+")
        array = group["0"]
        shape = self.grid["shape"]
        if list(array.shape) != shape:
            raise PaintingError("Painting shape differs from the reconstruction.", "geometry_mismatch")
        meta = self._meta(group)
        if base_revision != meta.get("revision"):
            raise PaintingError("Painting changed since it was loaded; reload it.", "revision_conflict")
        before = np.asarray(array[:])
        after = np.asarray(change(before.copy()))
        if list(after.shape) != shape:
            raise PaintingError(
                f"Labels {list(after.shape)} do not match the reconstruction {shape}.",
                "geometry_mismatch",
            )
        indices = np.flatnonzero(before.ravel() != after.ravel())
        if not len(indices):
            return {"revision": base_revision, "changed_voxels": 0, "can_undo": bool(meta.get("undo"))}
        history = list(meta.get("undo", []))
        store = group.require_group(UNDO_GROUP)
        # An out-of-band writer invalidates history rather than being silently rolled back.
        if history and store[history[-1]].attrs["digest"] != label_digest(before):
            history = []
        revision = uuid.uuid4().hex
        entry = store.create_group(revision)
        entry.create_dataset("indices", data=indices)
        entry.create_dataset("before", data=before.ravel()[indices])
        entry.attrs.update(digest=label_digest(after), gesture=revision, source=source)
        history.append(revision)
        while len(history) > MAX_UNDO_GESTURES:
            history.pop(0)
        array[:] = after
        meta.update(revision=revision, undo=history)
        group.attrs[PAINTING] = meta
        for key in list(store.group_keys()):
            if key not in history:
                del store[key]
        return {"revision": revision, "changed_voxels": int(len(indices)), "can_undo": True}

    def undo(self, base_revision: Optional[str]) -> dict:
        """Restore the last edit (from any tool), refused if the stored labels are not what it left behind."""
        group = self._group("r+")
        meta = self._meta(group)
        if base_revision != meta.get("revision"):
            raise PaintingError("Painting changed since it was loaded; reload it.", "revision_conflict")
        history = list(meta.get("undo", []))
        if not history:
            raise PaintingError("Nothing to undo.", "nothing_to_undo")
        store = group[UNDO_GROUP]
        array = group["0"]
        before = np.asarray(array[:])
        if store[history[-1]].attrs["digest"] != label_digest(before):
            raise PaintingError("Painting was edited outside the shared history; undo refused.", "revision_conflict")
        after = before.copy()
        gesture = store[history[-1]].attrs["gesture"]
        source = store[history[-1]].attrs.get("source")
        removed = []
        while history and store[history[-1]].attrs["gesture"] == gesture:
            key = history.pop()
            after.ravel()[store[key]["indices"][:]] = store[key]["before"][:]
            removed.append(key)
        revision = uuid.uuid4().hex
        array[:] = after
        meta.update(revision=revision, undo=history)
        group.attrs[PAINTING] = meta
        for key in removed:
            del store[key]
        return {
            "revision": revision,
            "changed_voxels": int((before != after).sum()),
            "can_undo": bool(history),
            "undone": source,
        }

    def accept(
        self,
        base_revision: Optional[str],
        proposal: Proposal,
        proposal_label: int,
        label: int,
        digest: str,
        box: Optional[list] = None,
    ) -> dict:
        """Copy one label of a completed proposal into unknown voxels as ``label``; human labels are kept.

        ``digest`` is the proposal digest that was inspected: a proposal rewritten since then is refused.
        ``box`` limits acceptance to ``[[z0, z1], [y0, y1], [x0, x1]]`` (half-open voxel indices).
        """
        if not 0 < int(label) < 256:
            raise PaintingError("Accepted label must be a class 1-255.", "invalid_request")
        if proposal.segmentation.zarr() == self.segmentation.zarr():
            raise PaintingError("A painting cannot accept itself.", "invalid_request")
        record = proposal.record
        if digest != record.get("digest"):
            raise PaintingError("Proposal changed since it was inspected; inspect it again.", "revision_conflict")
        values = proposal.labels()
        if list(values.shape) != self.grid["shape"]:
            raise PaintingError("Proposal shape differs from the reconstruction.", "geometry_mismatch")
        mask = values == int(proposal_label)
        if box is not None:
            window = np.zeros_like(mask)
            window[tuple(slice(int(lo), int(hi)) for lo, hi in box)] = True
            mask &= window
        if not mask.any():
            raise PaintingError(f"Proposal has no voxels of label {proposal_label} here.", "invalid_request")
        source = {
            "kind": "accepted_proposal",
            "proposal": {
                "name": proposal.segmentation.name,
                "user_id": proposal.segmentation.user_id,
                "session_id": proposal.segmentation.session_id,
            },
            "proposal_label": int(proposal_label),
            "proposal_digest": record["digest"],
            "producer": record.get("producer"),
        }
        kept = {}

        def change(labels):
            kept["human"] = int((mask & (labels != 0)).sum())
            labels[mask & (labels == 0)] = int(label)
            return labels

        result = self._edit(base_revision, change, source)
        return {**result, "kept_human_voxels": kept.get("human", 0), "source": source}
