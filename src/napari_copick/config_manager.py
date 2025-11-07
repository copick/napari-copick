"""Configuration management for napari-copick plugin."""

import json
import logging
from typing import List, Optional

import copick
from napari.utils import DirectLabelColormap
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox

from napari_copick.dialogs import DatasetIdDialog

# Import thumbnail cache setup
try:
    from copick_shared_ui.core.image_interface import get_image_interface
    from copick_shared_ui.core.thumbnail_cache import set_global_cache_config, set_global_cache_image_interface
except ImportError:
    set_global_cache_config = None
    set_global_cache_image_interface = None
    get_image_interface = None

# Import the shared EditObjectTypesDialog
try:
    from copick_shared_ui.ui.edit_object_types_dialog import EditObjectTypesDialog
except ImportError:
    EditObjectTypesDialog = None


class ConfigManager:
    """Handles configuration loading and management for the napari-copick plugin."""

    def __init__(self, parent_widget):
        """Initialize the config manager.

        Args:
            parent_widget: The main CopickPlugin widget instance
        """
        self.parent_widget = parent_widget
        self.logger = logging.getLogger("CopickPlugin.ConfigManager")

    def open_file_dialog(self) -> None:
        """Open file dialog to select a config file."""
        path, _ = QFileDialog.getOpenFileName(
            self.parent_widget,
            "Open Config",
            "",
            "JSON Files (*.json)",
        )
        if path:
            self.load_config(config_path=path)

    def open_dataset_dialog(self) -> None:
        """Open dialog to load from dataset IDs."""
        dialog = DatasetIdDialog(self.parent_widget)
        if dialog.exec_():
            dataset_ids, overlay_root = dialog.get_values()
            if dataset_ids:
                self.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def open_edit_objects_dialog(self) -> None:
        """Open the EditObjectTypesDialog to manage object types."""
        if not self.parent_widget.root:
            self.parent_widget.info_label.setText("No configuration loaded. Please load a config first.")
            return

        if EditObjectTypesDialog is None:
            self.parent_widget.info_label.setText(
                "EditObjectTypesDialog is not available. Shared component may not be installed.",
            )
            return

        try:
            # Use the raw PickableObject instances from the config instead of the overlay objects
            # This avoids the "model_copy" issue with CopickObjectCDP instances
            config_objects = self.parent_widget.root.config.pickable_objects

            dialog = EditObjectTypesDialog(self.parent_widget, config_objects)
            if dialog.exec_() == QDialog.Accepted:
                # Check if there are any changes
                if dialog.has_changes():
                    # Get the updated objects from the dialog
                    updated_objects = dialog.get_objects()

                    # Update the configuration with the new objects
                    self.parent_widget.root.config.pickable_objects = updated_objects

                    # Save the updated config to disk
                    self._save_config()

                    # Clear the cached overlay objects so they get recreated with the new config
                    self.parent_widget.root._objects = None

                    # Update any UI elements that depend on the object types
                    self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)

                    # Update any loaded segmentation layers with new colormap
                    for layer in self.parent_widget.viewer.layers:
                        if hasattr(layer, "colormap") and "Segmentation:" in layer.name:
                            layer.colormap = DirectLabelColormap(
                                color_dict=self.parent_widget.get_copick_colormap(),
                            )
                            layer.painting_labels = [obj.label for obj in updated_objects]

                    self.parent_widget.info_label.setText(
                        f"Updated and saved {len(updated_objects)} object types in configuration",
                    )
                    self.logger.info("Object types configuration updated and saved successfully")
                else:
                    self.parent_widget.info_label.setText("No changes made to object types")
                    self.logger.info("No changes made to object types")
        except Exception as e:
            self.parent_widget.info_label.setText(f"Error opening EditObjectTypesDialog: {str(e)}")

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file
        """
        if config_path:
            self.parent_widget.root = copick.from_file(config_path)
            self.parent_widget.config_path = config_path  # Store config path for saving

            # Initialize thumbnail cache with config file
            self._setup_thumbnail_cache(config_path)

            self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=False)
            self._update_gallery()
            self._enable_buttons()
            self.parent_widget.info_label.setText(f"Loaded config from {config_path}")

    def load_from_dataset_ids(
        self,
        dataset_ids: Optional[List[str]] = None,
        overlay_root: str = "/tmp/overlay_root",
    ) -> None:
        """Load configuration from dataset IDs.

        Args:
            dataset_ids: List of dataset IDs to load
            overlay_root: Root directory for overlay filesystem
        """
        if dataset_ids:
            self.parent_widget.root = copick.from_czcdp_datasets(
                dataset_ids=dataset_ids,
                overlay_root=overlay_root,
                overlay_fs_args={"auto_mkdir": True},
            )

            # Initialize thumbnail cache with dataset-based config
            self._setup_thumbnail_cache_for_datasets(dataset_ids)

            self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=False)
            self._update_gallery()
            self._enable_buttons()
            self.parent_widget.info_label.setText(
                f"Loaded project from dataset IDs: {', '.join(map(str, dataset_ids))}",
            )

    def _setup_thumbnail_cache(self, config_path: str) -> None:
        """Set up thumbnail cache for file-based config.

        Args:
            config_path: Path to the configuration file
        """
        if set_global_cache_config:
            set_global_cache_config(config_path, app_name="copick")
            self._setup_image_interface()

    def _setup_thumbnail_cache_for_datasets(self, dataset_ids: List[str]) -> None:
        """Set up thumbnail cache for dataset-based config.

        Args:
            dataset_ids: List of dataset IDs
        """
        if set_global_cache_config:
            # For dataset-based configs, use a unique cache key based on dataset IDs
            cache_key = f"datasets_{'-'.join(map(str, dataset_ids))}"
            set_global_cache_config(cache_key, app_name="copick")
            self._setup_image_interface()

    def _setup_image_interface(self) -> None:
        """Set up image interface for thumbnail cache."""
        if set_global_cache_image_interface and get_image_interface:
            image_interface = get_image_interface()
            if image_interface:
                set_global_cache_image_interface(image_interface, app_name="copick")

    def _update_gallery(self) -> None:
        """Update the gallery widget with current copick root."""
        if hasattr(self.parent_widget, "gallery_widget") and self.parent_widget.gallery_widget:
            self.parent_widget.gallery_widget.set_copick_root(self.parent_widget.root)

    def _enable_buttons(self) -> None:
        """Enable UI buttons after configuration is loaded."""
        self.parent_widget.edit_objects_button.setEnabled(True)
        self.parent_widget.save_segmentation_button.setEnabled(True)
        self.parent_widget.save_picks_button.setEnabled(True)

    def _save_config(self) -> None:
        """Save the current config to disk."""
        if not self.parent_widget.root or not self.parent_widget.config_path:
            QMessageBox.warning(
                self.parent_widget,
                "No Configuration",
                "No copick configuration file loaded. Cannot save changes.\n\n"
                "Note: Configurations loaded from dataset IDs cannot be saved.",
            )
            return

        try:
            with open(self.parent_widget.config_path, "w") as f:
                json.dump(self.parent_widget.root.config.model_dump(), f, indent=4)

            self.logger.info(f"Configuration saved to {self.parent_widget.config_path}")
            self.parent_widget.info_label.setText(
                f"Configuration saved to {self.parent_widget.config_path}",
            )

        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            QMessageBox.critical(
                self.parent_widget,
                "Error Saving Configuration",
                f"Failed to save configuration: {str(e)}",
            )
            raise
