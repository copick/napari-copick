"""Save operations and dialogs for napari-copick plugin."""

import logging
from typing import Any, Dict, List

from napari.layers import Labels, Points
from qtpy.QtWidgets import QDialog

from napari_copick.async_loaders import save_filaments_worker, save_segmentation_worker
from napari_copick.dialogs import SaveLayerDialog, SaveSegmentationDialog
from napari_copick.filament_layers import CENTRELINE_KIND, CONTROLS_KIND, KIND_KEY, SESSION_KEY
from napari_copick.instance_layers import SEGMENTATION_TYPE_KEY
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
        self._last_session_id: str | None = None
        self._last_user_id: str | None = None

    def open_save_segmentation_dialog(self) -> None:
        """Open dialog to save a segmentation layer to copick."""
        if not self.parent_widget.root:
            self.parent_widget.info_label.setText("No configuration loaded. Please load a config first.")
            return

        # Get available segmentation layers (Labels layers), excluding nninteractive interaction layers and the
        # display-only layers of panoptic segmentations
        segmentation_layers = [
            layer
            for layer in self.parent_widget.viewer.layers
            if isinstance(layer, Labels)
            and layer.data.ndim == 3
            and not layer.metadata.get("nninteractive_interaction_layer", False)
            and layer.metadata.get(KIND_KEY) != "panoptic_segments"
            and layer.metadata.get(SEGMENTATION_TYPE_KEY) != "panoptic"
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

        # Prefer layers matching known prefixes, then fall back to active layer
        for layer in segmentation_layers:
            lower_name = layer.name.lower()
            if lower_name.startswith("object") or lower_name.startswith("semantic map"):
                selected_layer = layer
                break
        if selected_layer is None:
            if self.parent_widget.viewer.layers.selection.active in segmentation_layers:
                selected_layer = self.parent_widget.viewer.layers.selection.active
            elif segmentation_layers:
                selected_layer = segmentation_layers[0]

        # Check if this layer was loaded from (or created for) an existing segmentation
        presets = {}
        if selected_layer is not None:
            meta = selected_layer.metadata
            if "copick_source_object_name" in meta:
                selected_object_name = meta["copick_source_object_name"]
                should_enable_overwrite = "copick_segmentation" in meta
            seg = meta.get("copick_segmentation")
            seg_type = meta.get(SEGMENTATION_TYPE_KEY)
            if seg_type:
                presets["preset_segmentation_type"] = seg_type
            if seg_type == "multilabel" and seg is not None:
                presets["preset_segmentation_name"] = seg.name
            if meta.get("copick_voxel_size") is not None:
                presets["preset_voxel_size"] = meta["copick_voxel_size"]
            if meta.get("copick_run") is not None:
                presets["preset_run"] = meta["copick_run"]
            if seg is not None:
                presets["preset_session_id"] = seg.session_id
                presets["preset_user_id"] = seg.user_id

        dialog = SaveSegmentationDialog(
            self.parent_widget,
            segmentation_layers,
            available_runs,
            self.parent_widget.root.pickable_objects,
            preset_layer=selected_layer,
            preset_object_name=selected_object_name,
            preset_overwrite=should_enable_overwrite,
            preset_session_id=presets.pop("preset_session_id", self._last_session_id),
            preset_user_id=presets.pop("preset_user_id", self._last_user_id),
            **presets,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                self._last_session_id = result["session_id"]
                self._last_user_id = result["user_id"]
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

        # Get available points layers (filament layers are saved as filaments)
        points_layers = [
            layer
            for layer in self.parent_widget.viewer.layers
            if isinstance(layer, Points)
            and layer.data.shape[1] == 3
            and layer.metadata.get(KIND_KEY) not in (CENTRELINE_KIND, CONTROLS_KIND)
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
            preset_session_id=self._last_session_id,
            preset_user_id=self._last_user_id,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                self._last_session_id = result["session_id"]
                self._last_user_id = result["user_id"]
                success = save_picks_to_copick(result, self.parent_widget.info_label.setText)
                if success:
                    # Refresh tree while preserving expansion state
                    self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)
            except Exception as e:
                self.parent_widget.info_label.setText(f"Error saving picks: {str(e)}")
                self.logger.exception(f"Error saving picks: {str(e)}")

    def filament_layers(self) -> List[Any]:
        """One layer per open filament session (the centreline layer, else the control layer)."""
        layers, sessions = [], set()
        for layer in self.parent_widget.viewer.layers:
            session = layer.metadata.get(SESSION_KEY) if hasattr(layer, "metadata") else None
            if session is not None and id(session) not in sessions:
                sessions.add(id(session))
                layers.append(layer)
        return layers

    def open_save_filaments_dialog(self, preset_layer: Any = None) -> None:
        """Open dialog to save a traced filament set to copick."""
        if not self.parent_widget.root:
            self.parent_widget.info_label.setText("No configuration loaded. Please load a config first.")
            return
        layers = self.filament_layers()
        if not layers:
            self.parent_widget.info_label.setText("No filament layers found. Trace or load filaments first.")
            return
        if preset_layer is None:
            active = self.parent_widget.viewer.layers.selection.active
            preset_layer = next(
                (
                    layer
                    for layer in layers
                    if active is not None and layer.metadata.get(SESSION_KEY) is active.metadata.get(SESSION_KEY)
                ),
                layers[0],
            )
        session = preset_layer.metadata[SESSION_KEY]
        runs = get_runs_from_open_layers(self.parent_widget.viewer)
        runs.setdefault(session.run.name, session.run)
        read_only = session.read_only
        dialog = SaveLayerDialog(
            parent=self.parent_widget,
            layers=layers,
            available_runs=runs,
            pickable_objects=self.parent_widget.root.pickable_objects,
            layer_type="filaments",
            preset_layer=preset_layer,
            preset_object_name=session.object_name,
            preset_overwrite=not read_only and session.source is not None,
            preset_run=session.run,
            preset_session_id=self._last_session_id if read_only else session.session_id,
            preset_user_id=self._last_user_id if read_only else session.user_id,
            preset_voxel_size=session.step,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        values = dialog.get_values()
        session = values["layer"].metadata[SESSION_KEY]
        if str(values["session_id"]) == "0":
            self.parent_widget.info_label.setText("Session 0 is reserved for tool output; choose another session.")
            return
        if values["object_name"] and values["object_name"] != session.object_name:
            session.object_name = values["object_name"]
        if values.get("polarity_known"):
            for i in list(session.filaments):
                session.set_polarity_known(i, True)
        self._last_session_id = values["session_id"]
        self._last_user_id = values["user_id"]
        values["session"] = session
        operation_id = f"save_filaments_{id(values)}"
        self.parent_widget._add_operation(operation_id, f"Saving filaments '{session.object_name}'...")
        worker = save_filaments_worker(values)
        worker.yielded.connect(lambda msg: self._on_progress(msg, values, "save_filaments"))
        worker.returned.connect(lambda result: self._on_filaments_saved(result, operation_id))
        worker.errored.connect(lambda e: self._on_save_error(str(e), values, operation_id))
        worker.finished.connect(lambda: self._cleanup_save_worker(operation_id))
        worker.start()
        self.parent_widget.loading_workers[operation_id] = worker

    def _on_filaments_saved(self, result: Dict[str, Any], operation_id: str) -> None:
        self.parent_widget.info_label.setText(result.get("message", "Filaments saved"))
        self.parent_widget._remove_operation(operation_id)
        # Rename the session's layers to its new identity and refresh the tree
        for layer in self.filament_layers():
            session = layer.metadata.get(SESSION_KEY)
            if session is not None:
                from napari_copick.filament_layers import layer_name

                layer.name = layer_name(session, controls=layer.metadata.get(KIND_KEY) == CONTROLS_KIND)
        if hasattr(self.parent_widget, "annotate_widget") and self.parent_widget.annotate_widget is not None:
            self.parent_widget.annotate_widget.refresh()
        self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)

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
                elif item["type"] == "filaments":
                    for filaments in item["filaments"]:
                        self.logger.info(f"Deleting filaments: {filaments}")
                        affected_runs.add(filaments.run)
                        filaments.delete()
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
                if hasattr(run, "refresh_filaments"):
                    run.refresh_filaments()

            # Refresh tree to reflect changes
            self.parent_widget.tree_expansion_manager.populate_tree(preserve_expansion=True)

        except Exception as e:
            self.parent_widget.info_label.setText(f"Error deleting items: {str(e)}")
            self.logger.exception(f"Error deleting items: {str(e)}")
        finally:
            # Remove global loading indicator
            self.parent_widget._remove_operation(operation_id)
