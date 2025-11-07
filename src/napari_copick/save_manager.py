"""Save operations and dialogs for napari-copick plugin."""

import logging
from typing import Any, Dict, List

from napari.layers import Labels, Points
from qtpy.QtWidgets import QDialog

from napari_copick.async_loaders import save_segmentation_worker
from napari_copick.dialogs import SaveLayerDialog, SaveSegmentationDialog
from napari_copick.save_utils import get_runs_from_open_layers, save_picks_to_copick


class SaveManager:
    """Handles save operations and dialogs for the napari-copick plugin."""

    def __init__(self, parent_widget):
        """Initialize the save manager.

        Args:
            parent_widget: The main CopickPlugin widget instance
        """
        self.parent_widget = parent_widget
        self.logger = logging.getLogger("CopickPlugin.SaveManager")

    def open_save_segmentation_dialog(self) -> None:
        """Open dialog to save a segmentation layer to copick."""
        if not self.parent_widget.root:
            self.parent_widget.info_label.setText("No configuration loaded. Please load a config first.")
            return

        # Get available segmentation layers (Labels layers)
        segmentation_layers = [
            layer for layer in self.parent_widget.viewer.layers if isinstance(layer, Labels) and layer.data.ndim == 3
        ]

        if not segmentation_layers:
            self.parent_widget.info_label.setText("No segmentation layers found in the viewer.")
            return

        # Get runs from currently open image layers
        available_runs = get_runs_from_open_layers(self.parent_widget.viewer)

        if not available_runs:
            self.parent_widget.info_label.setText("No runs found from currently open image layers.")
            return

        # Check if there's a currently selected segmentation layer to preset dialog values
        selected_layer = None
        selected_object_name = None
        should_enable_overwrite = False

        # Look for the currently active layer or the first segmentation layer
        if self.parent_widget.viewer.layers.selection.active in segmentation_layers:
            selected_layer = self.parent_widget.viewer.layers.selection.active
        elif segmentation_layers:
            selected_layer = segmentation_layers[0]

        # Check if this layer was loaded from an existing segmentation
        if selected_layer and "copick_source_object_name" in selected_layer.metadata:
            selected_object_name = selected_layer.metadata["copick_source_object_name"]
            should_enable_overwrite = True

        dialog = SaveSegmentationDialog(
            self.parent_widget,
            segmentation_layers,
            available_runs,
            self.parent_widget.root.pickable_objects,
            preset_layer=selected_layer,
            preset_object_name=selected_object_name,
            preset_overwrite=should_enable_overwrite,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                self.save_segmentation_async(result)
            except Exception as e:
                self.parent_widget.info_label.setText(f"Error saving segmentation: {str(e)}")
                self.logger.exception(f"Error saving segmentation: {str(e)}")

    def save_segmentation_async(self, save_params: Dict[str, Any]) -> None:
        """Save segmentation asynchronously with loading indicator.

        Args:
            save_params: Dictionary containing save parameters
        """
        # Create a unique operation ID for this save operation
        operation_id = f"save_segmentation_{save_params['object_name']}_{id(save_params)}"

        # Add global loading indicator
        self.parent_widget._add_operation(operation_id, f"Saving segmentation '{save_params['object_name']}'...")

        # Create the save worker
        worker = save_segmentation_worker(save_params)

        # Connect signals
        worker.yielded.connect(lambda msg: self._on_progress(msg, save_params, "save_segmentation"))
        worker.returned.connect(lambda result: self._on_segmentation_saved(result, operation_id))
        worker.errored.connect(lambda e: self._on_save_error(str(e), save_params, operation_id))
        worker.finished.connect(lambda: self._cleanup_save_worker(operation_id))

        # Start the worker
        worker.start()

        # Store the worker to track it using operation_id as key
        self.parent_widget.loading_workers[operation_id] = worker
        self.parent_widget.info_label.setText(f"Saving segmentation '{save_params['object_name']}'...")

    def _on_segmentation_saved(self, result: Dict[str, Any], operation_id: str) -> None:
        """Handle successful segmentation save.

        Args:
            result: The save result
            operation_id: The operation ID for cleanup
        """
        if result.get("success", False):
            message = result.get("message", "Segmentation saved successfully")

            # Handle different save modes
            if result.get("split_instances", False):
                instance_count = result.get("instance_count", 0)
                self.parent_widget.info_label.setText(f"{message} ({instance_count} instances)")
                self.logger.info(
                    f"Successfully saved {instance_count} split instances for '{result.get('object_name')}'",
                )
            elif result.get("convert_to_binary", False):
                self.parent_widget.info_label.setText(f"{message} (converted to binary)")
                self.logger.info(f"Successfully saved binary segmentation '{result.get('object_name')}'")
            else:
                self.parent_widget.info_label.setText(message)
                self.logger.info(f"Successfully saved single segmentation '{result.get('object_name')}'")

            # Instead of rebuilding entire tree, just refresh the relevant voxel spacing
            # to show the new segmentation(s) while preserving expansion state
            self.parent_widget.tree_expansion_manager.refresh_tree_after_save(result)
        else:
            self.parent_widget.info_label.setText(
                f"Failed to save segmentation: {result.get('message', 'Unknown error')}",
            )

        # Remove global loading indicator
        self.parent_widget._remove_operation(operation_id)

    def _on_save_error(self, error_message: str, save_params: Dict[str, Any], operation_id: str) -> None:
        """Handle segmentation save error.

        Args:
            error_message: The error message
            save_params: The save parameters
            operation_id: The operation ID for cleanup
        """
        self.parent_widget.info_label.setText(f"Error saving segmentation: {error_message}")
        self.logger.exception(f"Error saving segmentation: {error_message}")

        # Remove global loading indicator
        self.parent_widget._remove_operation(operation_id)

    def _cleanup_save_worker(self, operation_id: str) -> None:
        """Clean up save worker.

        Args:
            operation_id: The operation ID to clean up
        """
        if operation_id in self.parent_widget.loading_workers:
            del self.parent_widget.loading_workers[operation_id]

    def open_save_picks_dialog(self) -> None:
        """Open dialog to save a points layer to copick."""
        if not self.parent_widget.root:
            self.parent_widget.info_label.setText("No configuration loaded. Please load a config first.")
            return

        # Get available points layers
        points_layers = [
            layer
            for layer in self.parent_widget.viewer.layers
            if isinstance(layer, Points) and layer.data.shape[1] == 3
        ]

        if not points_layers:
            self.parent_widget.info_label.setText("No points layers found in the viewer.")
            return

        # Get runs from currently open image layers
        available_runs = get_runs_from_open_layers(self.parent_widget.viewer)

        if not available_runs:
            self.parent_widget.info_label.setText("No runs found from currently open image layers.")
            return

        # Check if there's a currently selected points layer to preset dialog values
        selected_layer = None
        selected_object_name = None
        selected_run = None
        should_enable_overwrite = False

        # Look for the currently active layer or the first points layer
        if self.parent_widget.viewer.layers.selection.active in points_layers:
            selected_layer = self.parent_widget.viewer.layers.selection.active
        elif points_layers:
            selected_layer = points_layers[0]

        # Check if this layer was loaded from existing picks
        if selected_layer and "copick_source_object_name" in selected_layer.metadata:
            selected_object_name = selected_layer.metadata["copick_source_object_name"]
            selected_run = selected_layer.metadata.get("copick_run")
            should_enable_overwrite = True

        dialog = SaveLayerDialog(
            parent=self.parent_widget,
            layers=points_layers,
            available_runs=available_runs,
            pickable_objects=self.parent_widget.root.pickable_objects,
            layer_type="picks",
            preset_layer=selected_layer,
            preset_object_name=selected_object_name,
            preset_overwrite=should_enable_overwrite,
            preset_run=selected_run,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                success = save_picks_to_copick(result, self.parent_widget.info_label.setText)
                if success:
                    # Refresh tree while preserving expansion state
                    self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)
            except Exception as e:
                self.parent_widget.info_label.setText(f"Error saving picks: {str(e)}")
                self.logger.exception(f"Error saving picks: {str(e)}")

    def _on_progress(self, message: str, save_params: Dict[str, Any], save_type: str) -> None:
        """Handle progress updates from save workers.

        Args:
            message: Progress message
            save_params: Save parameters
            save_type: Type of save operation
        """
        self.parent_widget.info_label.setText(f"{message}")

    def delete_items_async(self, items: List[Dict[str, Any]]) -> None:
        """Delete items asynchronously with loading indicator.

        Args:
            items: List of items to delete
        """
        # Create a unique operation ID for this delete operation
        operation_id = f"delete_items_{id(items)}"

        # Add global loading indicator
        self.parent_widget._add_operation(operation_id, f"Deleting {len(items)} items...")

        # Perform deletion synchronously for now (could be made async later)
        try:
            deleted_count = 0
            affected_runs = set()

            self.logger.info(f"Starting deletion of {len(items)} item groups")

            for item in items:
                self.logger.info(f"Processing item: {item}")
                if item["type"] == "picks":
                    self.logger.info(f"Deleting {len(item['picks'])} picks")
                    for pick in item["picks"]:
                        self.logger.info(f"Deleting pick: {pick}")
                        affected_runs.add(pick.run)
                        pick.delete()
                        deleted_count += 1
                elif item["type"] in ["segmentations", "segmentation"]:  # Handle both singular and plural
                    self.logger.info(f"Deleting {len(item['segmentations'])} segmentations")
                    for segmentation in item["segmentations"]:
                        self.logger.info(f"Deleting segmentation: {segmentation}")
                        affected_runs.add(segmentation.run)
                        segmentation.delete()
                        deleted_count += 1

            # Update UI
            self.parent_widget.info_label.setText(f"Successfully deleted {deleted_count} items.")

            # Only refresh the runs that had items deleted from them
            for run in affected_runs:
                run.refresh_segmentations()
                run.refresh_picks()

            # Refresh tree to reflect changes
            self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)

        except Exception as e:
            self.parent_widget.info_label.setText(f"Error deleting items: {str(e)}")
            self.logger.exception(f"Error deleting items: {str(e)}")
        finally:
            # Remove global loading indicator
            self.parent_widget._remove_operation(operation_id)
