"""Dialog classes for the napari-copick plugin."""

from typing import Any, Dict, List, Optional, Tuple

import copick
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
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
    """Unified dialog for saving segmentation and points layers to copick."""

    def __init__(
        self,
        parent: Any,
        layers: List[Any],
        available_runs: Dict[str, copick.models.CopickRun],
        pickable_objects: List[copick.models.PickableObject],
        layer_type: str = "segmentation",  # "segmentation" or "picks"
        preset_layer: Optional[Any] = None,
        preset_object_name: Optional[str] = None,
        preset_overwrite: bool = False,
        preset_run: Optional[copick.models.CopickRun] = None,
    ) -> None:
        super().__init__(parent)

        self.layer_type = layer_type
        layer_name = "Segmentation" if layer_type == "segmentation" else "Picks"
        self.setWindowTitle(f"Save {layer_name}")
        self.setMinimumWidth(400)

        self.layers = layers
        self.available_runs = available_runs
        self.pickable_objects = pickable_objects

        # Store preset values
        self.preset_layer = preset_layer
        self.preset_object_name = preset_object_name
        self.preset_overwrite = preset_overwrite
        self.preset_run = preset_run

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Layer selection
        self.layer_combo = QComboBox()
        for layer in layers:
            self.layer_combo.addItem(layer.name, layer)
        layer_label = f"{layer_name} Layer:" if layer_type == "segmentation" else "Points Layer:"
        form_layout.addRow(layer_label, self.layer_combo)

        # Run selection
        self.run_combo = QComboBox()
        for run_name, run in available_runs.items():
            self.run_combo.addItem(run_name, run)
        form_layout.addRow("Run:", self.run_combo)

        # Voxel spacing selection (only for segmentations)
        if layer_type == "segmentation":
            self.voxel_spacing_combo = QComboBox()
            self.run_combo.currentIndexChanged.connect(self.update_voxel_spacings)
            form_layout.addRow("Voxel Spacing:", self.voxel_spacing_combo)

        # Object type selection
        self.object_combo = QComboBox()
        for obj in pickable_objects:
            self.object_combo.addItem(obj.name, obj)
        form_layout.addRow("Object Type:", self.object_combo)

        # Session ID
        self.session_input = QLineEdit("manual")
        form_layout.addRow("Session ID:", self.session_input)

        # User ID
        self.user_input = QLineEdit("napari")
        form_layout.addRow("User ID:", self.user_input)

        # Split instances checkbox (only for segmentations)
        if layer_type == "segmentation":
            self.split_instances_checkbox = QCheckBox("Split instances (create binary volumes for each label)")
            self.split_instances_checkbox.setChecked(False)
            form_layout.addRow("", self.split_instances_checkbox)

            # Convert to binary checkbox (only for segmentations)
            self.convert_to_binary_checkbox = QCheckBox("Convert to binary (set all non-zero labels to 1)")
            self.convert_to_binary_checkbox.setChecked(False)
            form_layout.addRow("", self.convert_to_binary_checkbox)

            # Make the checkboxes mutually exclusive
            self.split_instances_checkbox.toggled.connect(self._on_split_instances_toggled)
            self.convert_to_binary_checkbox.toggled.connect(self._on_convert_to_binary_toggled)

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

        # Initialize voxel spacings for segmentations
        if layer_type == "segmentation":
            self.update_voxel_spacings()

        # Apply presets if provided
        if self.preset_layer:
            # Find and select the preset layer
            for i in range(self.layer_combo.count()):
                if self.layer_combo.itemData(i) == self.preset_layer:
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

        if self.preset_overwrite:
            self.overwrite_checkbox.setChecked(True)

    def update_voxel_spacings(self) -> None:
        """Update voxel spacing combo based on selected run (segmentations only)."""
        if self.layer_type != "segmentation":
            return

        self.voxel_spacing_combo.clear()

        if self.run_combo.currentData():
            run = self.run_combo.currentData()
            for voxel_spacing in run.voxel_spacings:
                self.voxel_spacing_combo.addItem(
                    f"{voxel_spacing.voxel_size} Å",
                    voxel_spacing,
                )

    def get_values(self) -> Dict[str, Any]:
        """Get the values from the dialog."""
        base_values = {
            "layer": self.layer_combo.currentData(),
            "run": self.run_combo.currentData(),
            "object_name": self.object_combo.currentData().name,
            "session_id": self.session_input.text(),
            "user_id": self.user_input.text(),
            "exist_ok": self.overwrite_checkbox.isChecked(),
        }

        # Add voxel spacing and processing options for segmentations
        if self.layer_type == "segmentation":
            base_values["voxel_spacing"] = self.voxel_spacing_combo.currentData()
            base_values["split_instances"] = self.split_instances_checkbox.isChecked()
            base_values["convert_to_binary"] = self.convert_to_binary_checkbox.isChecked()

        return base_values

    def _on_split_instances_toggled(self, checked: bool) -> None:
        """Handle split instances checkbox toggle - disable convert to binary when checked."""
        if checked:
            self.convert_to_binary_checkbox.setChecked(False)

    def _on_convert_to_binary_toggled(self, checked: bool) -> None:
        """Handle convert to binary checkbox toggle - disable split instances when checked."""
        if checked:
            self.split_instances_checkbox.setChecked(False)


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
