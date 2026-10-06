"""Dialog classes for the napari-copick plugin."""

from typing import Any, Dict, List, Optional, Tuple

import copick
from copick_shared_ui.core.types import is_filament_object
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class DatasetIdDialog(QDialog):
    """Dialog for loading from dataset IDs."""

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load from Dataset IDs")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Dataset IDs input
        form_layout = QFormLayout()
        self.dataset_ids_input = QLineEdit()
        self.dataset_ids_input.setPlaceholderText("10000, 10001, ...")
        form_layout.addRow("Dataset IDs (comma separated):", self.dataset_ids_input)

        # Overlay root input
        self.overlay_root_input = QLineEdit()
        self.overlay_root_input.setText("/tmp/overlay_root")
        form_layout.addRow("Overlay Root:", self.overlay_root_input)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_values(self) -> Tuple[List[int], str]:
        """Get the values from the dialog."""
        dataset_ids_text = self.dataset_ids_input.text()
        dataset_ids = [int(id.strip()) for id in dataset_ids_text.split(",") if id.strip()]
        overlay_root = self.overlay_root_input.text()
        return dataset_ids, overlay_root


class SaveLayerDialog(QDialog):
    """Unified dialog for saving segmentation, points and filament layers to copick."""

    SEGMENTATION_TYPES = [
        ("Binary", "binary"),
        ("Multilabel", "multilabel"),
        ("Instance", "instance"),
        ("Panoptic (display only)", "panoptic"),
    ]

    def __init__(
        self,
        parent: Any,
        layers: List[Any],
        available_runs: Dict[str, copick.models.CopickRun],
        pickable_objects: List[copick.models.PickableObject],
        layer_type: str = "segmentation",  # "segmentation", "picks" or "filaments"
        preset_layer: Optional[Any] = None,
        preset_object_name: Optional[str] = None,
        preset_overwrite: bool = False,
        preset_run: Optional[copick.models.CopickRun] = None,
        preset_session_id: Optional[str] = None,
        preset_user_id: Optional[str] = None,
        preset_segmentation_type: Optional[str] = None,
        preset_segmentation_name: Optional[str] = None,
        preset_voxel_size: Optional[float] = None,
    ) -> None:
        super().__init__(parent)

        self.layer_type = layer_type
        layer_name = {"segmentation": "Segmentation", "picks": "Picks", "filaments": "Filaments"}[layer_type]
        self.setWindowTitle(f"Save {layer_name}")
        self.setMinimumWidth(420)

        self.layers = layers
        self.available_runs = available_runs
        self.pickable_objects = pickable_objects

        # Store preset values
        self.preset_layer = preset_layer
        self.preset_object_name = preset_object_name
        self.preset_overwrite = preset_overwrite
        self.preset_run = preset_run
        self.preset_session_id = preset_session_id
        self.preset_user_id = preset_user_id
        self.preset_segmentation_type = preset_segmentation_type
        self.preset_segmentation_name = preset_segmentation_name
        self.preset_voxel_size = preset_voxel_size

        layout = QVBoxLayout()
        form_layout = QFormLayout()
        self._form = form_layout

        # Layer selection
        self.layer_combo = QComboBox()
        for layer in layers:
            self.layer_combo.addItem(layer.name, layer)
        layer_label = {"segmentation": "Segmentation Layer:", "picks": "Points Layer:", "filaments": "Filament Layer:"}
        form_layout.addRow(layer_label[layer_type], self.layer_combo)

        # Run selection
        self.run_combo = QComboBox()
        for run_name, run in available_runs.items():
            self.run_combo.addItem(run_name, run)
        form_layout.addRow("Run:", self.run_combo)

        # Voxel spacing selection (segmentations; filaments record the spacing they were traced at)
        if layer_type in ("segmentation", "filaments"):
            self.voxel_spacing_combo = QComboBox()
            self.run_combo.currentIndexChanged.connect(self.update_voxel_spacings)
            form_layout.addRow("Voxel Spacing:", self.voxel_spacing_combo)

        # Segmentation type (replaces the former multilabel checkbox)
        if layer_type == "segmentation":
            self.type_combo = QComboBox()
            for text, value in self.SEGMENTATION_TYPES:
                self.type_combo.addItem(text, value)
            panoptic_item = self.type_combo.model().item(self.type_combo.count() - 1)
            panoptic_item.setEnabled(False)
            panoptic_item.setToolTip("Panoptic segmentations can be viewed but not written from napari.")
            self.type_combo.setToolTip(
                "Binary: one object, voxels 0/1.\n"
                "Multilabel: several objects, voxel = object label.\n"
                "Instance: one object, voxel = instance ID (0 = background); IDs match picks and filaments.",
            )
            form_layout.addRow("Type:", self.type_combo)

        # Object type selection (combo box for binary/instance/picks/filaments)
        self.object_combo = QComboBox()
        for obj in pickable_objects:
            if layer_type == "filaments" and not is_filament_object(obj):
                continue
            self.object_combo.addItem(obj.name, obj)
        self.object_label = QLabel("Object Type:")
        form_layout.addRow(self.object_label, self.object_combo)

        # Segmentation name input (multilabel)
        if layer_type == "segmentation":
            self.segmentation_name_input = QLineEdit()
            self.segmentation_name_input.setPlaceholderText("Enter segmentation name...")
            self.segmentation_name_label = QLabel("Segmentation Name:")
            form_layout.addRow(self.segmentation_name_label, self.segmentation_name_input)

        # Session ID
        self.session_input = QLineEdit(self.preset_session_id or "manual")
        form_layout.addRow("Session ID:", self.session_input)

        # User ID
        self.user_input = QLineEdit(self.preset_user_id or "napari")
        form_layout.addRow("User ID:", self.user_input)

        if layer_type == "segmentation":
            # Legacy binary-only processing options
            self.split_instances_checkbox = QCheckBox("Split labels into separate binary segmentations (legacy)")
            self.split_instances_checkbox.setChecked(False)
            self.split_instances_checkbox.setToolTip(
                "Writes one binary segmentation per label, with session IDs '<session>-<n>'.\n"
                "Prefer the 'Instance' type, which keeps all instances in one volume.",
            )
            form_layout.addRow("", self.split_instances_checkbox)

            self.convert_to_binary_checkbox = QCheckBox("Convert to binary (set all non-zero labels to 1)")
            self.convert_to_binary_checkbox.setChecked(True)  # Default ON for binary segmentations
            form_layout.addRow("", self.convert_to_binary_checkbox)

            # Make the checkboxes mutually exclusive
            self.split_instances_checkbox.toggled.connect(self._on_split_instances_toggled)
            self.convert_to_binary_checkbox.toggled.connect(self._on_convert_to_binary_toggled)
            self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        if layer_type == "filaments":
            self.polarity_checkbox = QCheckBox("Point order follows the polarity (all filaments)")
            self.polarity_checkbox.setChecked(False)
            self.polarity_checkbox.setTristate(False)
            form_layout.addRow("", self.polarity_checkbox)

            self.picks_checkbox = QCheckBox("Also write picks sampled along the filaments")
            self.picks_checkbox.setChecked(False)
            form_layout.addRow("", self.picks_checkbox)
            self.pick_spacing = QDoubleSpinBox()
            self.pick_spacing.setRange(1.0, 100000.0)
            self.pick_spacing.setDecimals(1)
            self.pick_spacing.setSuffix(" Å")
            self.pick_spacing.setToolTip("Distance between sampled picks along each filament")
            self.pick_spacing.setEnabled(False)
            form_layout.addRow("Pick spacing:", self.pick_spacing)
            self.picks_warning = QLabel(
                "Sampled picks replace the whole picks set of this object / user / session.",
            )
            self.picks_warning.setWordWrap(True)
            self.picks_warning.setStyleSheet("color: #d08000; font-size: 11px;")
            self.picks_warning.setVisible(False)
            form_layout.addRow("", self.picks_warning)
            self.picks_checkbox.toggled.connect(self.pick_spacing.setEnabled)
            self.picks_checkbox.toggled.connect(self.picks_warning.setVisible)
            self.object_combo.currentIndexChanged.connect(self._update_default_pick_spacing)
            self._update_default_pick_spacing()

        # Overwrite checkbox
        overwrite_label = f"Overwrite existing {layer_type}"
        self.overwrite_checkbox = QCheckBox(overwrite_label)
        self.overwrite_checkbox.setChecked(False)
        form_layout.addRow("", self.overwrite_checkbox)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        # Initialize voxel spacings
        if layer_type in ("segmentation", "filaments"):
            self.update_voxel_spacings()

        # Apply presets if provided
        if self.preset_layer:
            # Find and select the preset layer
            for i in range(self.layer_combo.count()):
                if self.layer_combo.itemData(i) is self.preset_layer:
                    self.layer_combo.setCurrentIndex(i)
                    break

        if self.preset_run:
            # Find and select the preset run
            for i in range(self.run_combo.count()):
                if self.run_combo.itemData(i) == self.preset_run:
                    self.run_combo.setCurrentIndex(i)
                    break

        if self.preset_object_name:
            # Find and select the preset object
            for i in range(self.object_combo.count()):
                obj = self.object_combo.itemData(i)
                if obj and obj.name == self.preset_object_name:
                    self.object_combo.setCurrentIndex(i)
                    break

        if self.preset_voxel_size is not None and layer_type in ("segmentation", "filaments"):
            for i in range(self.voxel_spacing_combo.count()):
                vs = self.voxel_spacing_combo.itemData(i)
                if vs is not None and abs(vs.voxel_size - self.preset_voxel_size) < 1e-6:
                    self.voxel_spacing_combo.setCurrentIndex(i)
                    break

        if layer_type == "segmentation":
            preset_type = self.preset_segmentation_type if self.preset_segmentation_type != "panoptic" else None
            if preset_type:
                self.type_combo.setCurrentIndex(self.type_combo.findData(preset_type))
            if self.preset_segmentation_name:
                self.segmentation_name_input.setText(self.preset_segmentation_name)
            self._on_type_changed()

        if self.preset_overwrite:
            self.overwrite_checkbox.setChecked(True)

    def update_voxel_spacings(self) -> None:
        """Update voxel spacing combo based on selected run (segmentations and filaments)."""
        if self.layer_type not in ("segmentation", "filaments"):
            return

        self.voxel_spacing_combo.clear()

        if self.run_combo.currentData():
            run = self.run_combo.currentData()
            for voxel_spacing in run.voxel_spacings:
                self.voxel_spacing_combo.addItem(
                    f"{voxel_spacing.voxel_size} Å",
                    voxel_spacing,
                )

    def segmentation_type(self) -> str:
        return self.type_combo.currentData() if self.layer_type == "segmentation" else ""

    def get_values(self) -> Dict[str, Any]:
        """Get the values from the dialog."""
        base_values = {
            "layer": self.layer_combo.currentData(),
            "run": self.run_combo.currentData(),
            "session_id": self.session_input.text(),
            "user_id": self.user_input.text(),
            "exist_ok": self.overwrite_checkbox.isChecked(),
        }

        if self.layer_type == "segmentation":
            seg_type = self.segmentation_type()
            is_binary = seg_type == "binary"
            base_values["voxel_spacing"] = self.voxel_spacing_combo.currentData()
            base_values["segmentation_type"] = seg_type
            base_values["is_multilabel"] = seg_type == "multilabel"
            base_values["split_instances"] = is_binary and self.split_instances_checkbox.isChecked()
            base_values["convert_to_binary"] = is_binary and self.convert_to_binary_checkbox.isChecked()

            # Multilabel segmentations have a free name; binary and instance ones are named after an object
            if seg_type == "multilabel":
                seg_name = self.segmentation_name_input.text()
                base_values["segmentation_name"] = seg_name
                base_values["object_name"] = seg_name  # For UI/logging compatibility
            else:
                base_values["object_name"] = self.object_combo.currentData().name
        elif self.layer_type == "filaments":
            vs = self.voxel_spacing_combo.currentData()
            base_values["voxel_spacing"] = vs.voxel_size if vs is not None else None
            base_values["object_name"] = self.object_combo.currentData().name if self.object_combo.count() else ""
            base_values["polarity_known"] = self.polarity_checkbox.isChecked()
            base_values["pick_spacing"] = self.pick_spacing.value() if self.picks_checkbox.isChecked() else None
        else:
            # For picks, always use object_name from combo
            base_values["object_name"] = self.object_combo.currentData().name

        return base_values

    def _update_default_pick_spacing(self) -> None:
        obj = self.object_combo.currentData()
        radius = getattr(obj, "radius", None) if obj is not None else None
        # Never the helical rise: that is descriptive, not a sampling distance.
        self.pick_spacing.setValue(float(radius) if radius else 100.0)

    def _on_split_instances_toggled(self, checked: bool) -> None:
        """Handle split instances checkbox toggle - disable convert to binary when checked."""
        if checked:
            self.convert_to_binary_checkbox.setChecked(False)

    def _on_convert_to_binary_toggled(self, checked: bool) -> None:
        """Handle convert to binary checkbox toggle - disable split instances when checked."""
        if checked:
            self.split_instances_checkbox.setChecked(False)

    def _on_type_changed(self, *_args) -> None:
        """Show the name field for multilabel, the object combo otherwise; binary-only options for binary."""
        if self.layer_type != "segmentation":
            return
        seg_type = self.segmentation_type()
        multilabel = seg_type == "multilabel"
        self.object_combo.setVisible(not multilabel)
        self.object_label.setVisible(not multilabel)
        self.segmentation_name_input.setVisible(multilabel)
        self.segmentation_name_label.setVisible(multilabel)
        binary = seg_type == "binary"
        self.split_instances_checkbox.setVisible(binary)
        self.convert_to_binary_checkbox.setVisible(binary)
        if not binary:
            self.split_instances_checkbox.setChecked(False)
            self.convert_to_binary_checkbox.setChecked(False)
        elif not self.split_instances_checkbox.isChecked():
            self.convert_to_binary_checkbox.setChecked(True)


# Legacy aliases for backward compatibility
class SaveSegmentationDialog(SaveLayerDialog):
    """Legacy wrapper for segmentation saving."""

    def __init__(
        self,
        parent: Any,
        segmentation_layers: List[Any],
        available_runs: Dict[str, copick.models.CopickRun],
        pickable_objects: List[copick.models.PickableObject],
        preset_layer: Optional[Any] = None,
        preset_object_name: Optional[str] = None,
        preset_overwrite: bool = False,
        preset_session_id: Optional[str] = None,
        preset_user_id: Optional[str] = None,
        **presets: Any,
    ) -> None:
        super().__init__(
            parent=parent,
            layers=segmentation_layers,
            available_runs=available_runs,
            pickable_objects=pickable_objects,
            layer_type="segmentation",
            preset_layer=preset_layer,
            preset_object_name=preset_object_name,
            preset_overwrite=preset_overwrite,
            preset_session_id=preset_session_id,
            preset_user_id=preset_user_id,
            **presets,
        )


class SavePicksDialog(SaveLayerDialog):
    """Legacy wrapper for picks saving."""

    def __init__(
        self,
        parent: Any,
        points_layers: List[Any],
        available_runs: Dict[str, copick.models.CopickRun],
        pickable_objects: List[copick.models.PickableObject],
        preset_layer: Optional[Any] = None,
        preset_object_name: Optional[str] = None,
        preset_overwrite: bool = False,
        preset_run: Optional[copick.models.CopickRun] = None,
    ) -> None:
        super().__init__(
            parent=parent,
            layers=points_layers,
            available_runs=available_runs,
            pickable_objects=pickable_objects,
            layer_type="picks",
            preset_layer=preset_layer,
            preset_object_name=preset_object_name,
            preset_overwrite=preset_overwrite,
            preset_run=preset_run,
        )
