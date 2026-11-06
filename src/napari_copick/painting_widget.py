"""Paint a copick reconstruction in napari, train CellCanvas on it and accept completed proposals.

Open a tomogram with the copick widget, select its image layer, then use this dock widget:

1. *Open painting* adds the shared painting (created all-unknown if missing) as a Labels layer;
2. paint with napari's label tools and *Save* (refused if the painting was saved elsewhere since it was opened);
3. *Train and predict* uploads the saved painting to a CellCanvas service and shows the completed prediction as a
   proposal layer;
4. *Accept* copies one proposal class into unknown painting voxels; *Undo* reverts the last save or acceptance,
   including ones made in copick-web.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from napari.qt.threading import create_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_copick.cellcanvas import CellCanvasService
from napari_copick.painting import Painting, PaintingError, Proposal

PAINTING_KEY = "copick_painting"
PROPOSAL_KEY = "copick_proposal"


class PaintingWidget(QWidget):
    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.painting: Optional[Painting] = None
        self.revision: Optional[str] = None
        self.painting_layer = None
        self.proposal: Optional[Proposal] = None
        self.proposal_digest: Optional[str] = None
        self.proposal_layer = None
        self._worker = None

        layout = QVBoxLayout(self)
        target = QGroupBox("Painting")
        form = QFormLayout(target)
        self.painting_name = QLineEdit("cellcanvas-painting")
        self.user_id = QLineEdit("napari")
        self.session_id = QLineEdit("1")
        form.addRow("Name", self.painting_name)
        form.addRow("User", self.user_id)
        form.addRow("Session", self.session_id)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open painting")
        self.save_button = QPushButton("Save")
        self.undo_button = QPushButton("Undo")
        for button in (self.open_button, self.save_button, self.undo_button):
            buttons.addWidget(button)
        form.addRow(buttons)
        layout.addWidget(target)

        backend = QGroupBox("CellCanvas")
        form = QFormLayout(backend)
        self.service_url = QLineEdit("http://127.0.0.1:8001")
        self.dataset = QLineEdit()
        self.proposal_name = QLineEdit("cellcanvas-prediction")
        form.addRow("Service", self.service_url)
        form.addRow("Dataset", self.dataset)
        form.addRow("Proposal", self.proposal_name)
        self.train_button = QPushButton("Train and predict")
        form.addRow(self.train_button)
        layout.addWidget(backend)

        accept = QGroupBox("Accept proposal")
        form = QFormLayout(accept)
        self.proposal_label = QSpinBox()
        self.proposal_label.setRange(1, 255)
        self.accept_label = QSpinBox()
        self.accept_label.setRange(1, 255)
        self.current_slice = QCheckBox("Only the current z slice")
        form.addRow("Proposal class", self.proposal_label)
        form.addRow("As painting class", self.accept_label)
        form.addRow(self.current_slice)
        self.load_proposal_button = QPushButton("Show proposal")
        self.accept_button = QPushButton("Accept into unknown voxels")
        row = QHBoxLayout()
        row.addWidget(self.load_proposal_button)
        row.addWidget(self.accept_button)
        form.addRow(row)
        layout.addWidget(accept)

        self.status = QLabel("Select a copick tomogram layer, then open its painting.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

        self.open_button.clicked.connect(self.open_painting)
        self.save_button.clicked.connect(self.save)
        self.undo_button.clicked.connect(self.undo)
        self.train_button.clicked.connect(self.train_and_predict)
        self.load_proposal_button.clicked.connect(self.show_proposal)
        self.accept_button.clicked.connect(self.accept)
        self._update_buttons()

    # helpers --------------------------------------------------------------------------------------------------

    def _report(self, text: str) -> None:
        self.status.setText(text)

    def _update_buttons(self) -> None:
        opened = self.painting is not None
        idle = self._worker is None
        for button in (self.save_button, self.undo_button, self.train_button, self.load_proposal_button):
            button.setEnabled(opened and idle)
        self.accept_button.setEnabled(opened and idle and self.proposal is not None)
        self.open_button.setEnabled(idle)

    def _tomogram_layer(self):
        layers = [self.viewer.layers.selection.active, *self.viewer.layers]
        for layer in layers:
            if layer is not None and "copick_tomogram" in getattr(layer, "metadata", {}):
                return layer
        return None

    @property
    def _run(self):
        return self.painting.tomogram.voxel_spacing.run

    @property
    def _voxel_size(self) -> float:
        return self.painting.tomogram.voxel_spacing.voxel_size

    def _show(self, attribute: str, data: np.ndarray, name: str, key: str, value, **kwargs):
        layer = getattr(self, attribute)
        if layer is not None and layer in self.viewer.layers:
            layer.data = data
        else:
            # Same world coordinates as napari-copick's tomogram layers: copick voxel size, no origin offset.
            layer = self.viewer.add_labels(data, name=name, scale=(self._voxel_size,) * 3, **kwargs)
            setattr(self, attribute, layer)
        layer.metadata[key] = value
        return layer

    # painting -------------------------------------------------------------------------------------------------

    def open_painting(self, tomogram=None) -> None:
        if tomogram is None:
            layer = self._tomogram_layer()
            if layer is None:
                self._report("Open a copick tomogram first (no layer carries a copick tomogram).")
                return
            tomogram = layer.metadata["copick_tomogram"]
        try:
            self.painting = Painting.open(
                tomogram.voxel_spacing.run,
                tomogram.voxel_spacing.voxel_size,
                tomogram.tomo_type,
                self.painting_name.text(),
                self.user_id.text(),
                self.session_id.text(),
            )
        except PaintingError as error:
            self._report(f"Cannot open painting: {error}")
            return
        self._reload()
        self._report(f"Painting '{self.painting_name.text()}' opened on {tomogram.tomo_type}.")
        self._update_buttons()

    def _reload(self) -> None:
        self.revision = self.painting.revision
        layer = self._show("painting_layer", self.painting.labels(), "painting", PAINTING_KEY, self.painting)
        layer.metadata["painting_revision"] = self.revision

    def save(self) -> Optional[dict]:
        try:
            result = self.painting.save(np.asarray(self.painting_layer.data), self.revision)
        except PaintingError as error:
            self._report(f"Not saved: {error}")
            return None
        self.revision = result["revision"]
        self.painting_layer.metadata["painting_revision"] = self.revision
        self._report(f"Saved {result['changed_voxels']} changed voxels.")
        return result

    def undo(self) -> Optional[dict]:
        try:
            result = self.painting.undo(self.revision)
        except PaintingError as error:
            self._report(f"Not undone: {error}")
            return None
        self._reload()
        self._report(f"Undid the last edit ({result['changed_voxels']} voxels restored).")
        return result

    # CellCanvas -----------------------------------------------------------------------------------------------

    def train_and_predict(self) -> None:
        """Save unsaved paint, then train, predict and publish the proposal off the GUI thread."""
        if self.save() is None:
            return
        service = CellCanvasService(self.service_url.text(), self.dataset.text())
        run, voxel_size, name = self._run, self._voxel_size, self.proposal_name.text()
        user, session = self.user_id.text(), self.session_id.text()
        painting = self.painting

        def work():
            trained = service.train(painting)
            prediction = service.predict()
            if prediction.painting_revision != trained["painting_revision"]:
                raise RuntimeError("The prediction was made from another painting revision.")
            return service.publish(prediction, run, voxel_size, name, user, session)

        self._worker = create_worker(work)
        self._worker.returned.connect(self._published)
        self._worker.errored.connect(lambda error: self._report(f"CellCanvas: {error}"))
        self._worker.finished.connect(self._finished)
        self._report("Training and predicting...")
        self._update_buttons()
        self._worker.start()

    def _finished(self) -> None:
        self._worker = None
        self._update_buttons()

    def _published(self, proposal: Proposal) -> None:
        self._load_proposal(proposal)
        record = proposal.record
        self._report(
            f"Prediction {record.get('prediction_version')} from painting revision "
            f"{record.get('painting_revision')} published as a proposal.",
        )

    # proposals ------------------------------------------------------------------------------------------------

    def show_proposal(self, proposal: Optional[Proposal] = None) -> None:
        try:
            proposal = proposal or Proposal.open(
                self._run,
                self._voxel_size,
                self.proposal_name.text(),
                self.user_id.text(),
                self.session_id.text(),
            )
            self._load_proposal(proposal)
        except PaintingError as error:
            self._report(f"No proposal to show: {error}")
            return
        self._report(f"Proposal from {self.proposal.record.get('producer')} shown.")

    def _load_proposal(self, proposal: Proposal) -> None:
        labels = proposal.labels()
        self.proposal = proposal
        self.proposal_digest = proposal.record["digest"]
        layer = self._show("proposal_layer", labels, "proposal", PROPOSAL_KEY, proposal, opacity=0.4)
        layer.metadata["proposal_digest"] = self.proposal_digest
        self._update_buttons()

    def accept(self) -> Optional[dict]:
        box = None
        if self.current_slice.isChecked():
            z = int(round(self.viewer.dims.point[0] / self._voxel_size))
            box = [[z, z + 1], [0, 1 << 30], [0, 1 << 30]]
        try:
            result = self.painting.accept(
                self.revision,
                self.proposal,
                self.proposal_label.value(),
                self.accept_label.value(),
                self.proposal_digest,
                box=box,
            )
        except PaintingError as error:
            self._report(f"Not accepted: {error}")
            return None
        self._reload()
        self._report(
            f"Accepted {result['changed_voxels']} unknown voxels; kept {result['kept_human_voxels']} human labels.",
        )
        return result
