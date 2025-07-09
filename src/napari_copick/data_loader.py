"""Data loading operations for napari-copick plugin."""

import logging
from typing import Any, Dict, Optional

import copick
import numpy as np
from napari.utils import DirectLabelColormap
from qtpy.QtWidgets import QTreeWidgetItem

from napari_copick.async_loaders import load_segmentation_worker, load_tomogram_worker


class DataLoader:
    """Handles async data loading operations for the napari-copick plugin."""

    def __init__(self, parent_widget):
        """Initialize the data loader.

        Args:
            parent_widget: The main CopickPlugin widget instance
        """
        self.parent_widget = parent_widget
        self.logger = logging.getLogger("CopickPlugin.DataLoader")

    def load_tomogram_async(self, tomogram: copick.models.CopickTomogram, item: Optional[QTreeWidgetItem]) -> None:
        """Load a tomogram asynchronously with loading indicator using napari's threading system.

        Args:
            tomogram: The tomogram to load
            item: The tree widget item associated with the tomogram (optional)
        """
        # Check if already loading
        if tomogram in self.parent_widget.loading_workers:
            self.logger.warning(f"Tomogram {tomogram.tomo_type} already loading, skipping")
            return

        # Add loading indicators
        if item is not None:
            # Add tree-specific loading indicator only if we have a tree item
            self.parent_widget.tree_view.add_loading_indicator(item)
            self.parent_widget.loading_items[tomogram] = item
        else:
            # For cases where no tree item is available (e.g., info widget clicks)
            self.parent_widget.loading_items[tomogram] = None

        # Add global loading indicator (always show this)
        operation_id = f"load_tomogram_{tomogram.tomo_type}_{id(tomogram)}"
        self.parent_widget._add_operation(operation_id, f"Loading tomogram: {tomogram.tomo_type}...")

        # Get selected resolution level
        resolution_level = self.parent_widget.resolution_combo.currentIndex()

        # Create worker using napari's threading system
        worker = load_tomogram_worker(tomogram, resolution_level)

        # Connect signals
        worker.yielded.connect(lambda msg: self._on_progress(msg, tomogram, "tomogram"))
        worker.returned.connect(lambda result: self._on_tomogram_loaded(result))
        worker.errored.connect(lambda e: self._on_error(str(e), tomogram, "tomogram"))
        worker.finished.connect(lambda: self._cleanup_worker(tomogram))

        # Start the worker
        worker.start()

        self.parent_widget.loading_workers[tomogram] = worker
        self.parent_widget.info_label.setText(f"Loading tomogram: {tomogram.tomo_type}...")

    def load_segmentation_async(self, segmentation: copick.models.CopickSegmentation, item: QTreeWidgetItem) -> None:
        """Load a segmentation asynchronously with loading indicator using napari's threading system.

        Args:
            segmentation: The segmentation to load
            item: The tree widget item associated with the segmentation
        """
        # Check if already loading
        if segmentation in self.parent_widget.loading_workers:
            self.logger.warning(f"Segmentation {segmentation.name} already loading, skipping")
            return

        # Add loading indicator
        self.parent_widget.tree_view.add_loading_indicator(item)
        self.parent_widget.loading_items[segmentation] = item

        # Add global loading indicator
        operation_id = f"load_segmentation_{segmentation.name}_{id(segmentation)}"
        self.parent_widget._add_operation(operation_id, f"Loading segmentation: {segmentation.name}...")

        # Get selected resolution level
        resolution_level = self.parent_widget.resolution_combo.currentIndex()

        # Create worker using napari's threading system
        worker = load_segmentation_worker(segmentation, resolution_level)

        # Connect signals
        worker.yielded.connect(lambda msg: self._on_progress(msg, segmentation, "segmentation"))
        worker.returned.connect(lambda result: self._on_segmentation_loaded(result))
        worker.errored.connect(lambda e: self._on_error(str(e), segmentation, "segmentation"))
        worker.finished.connect(lambda: self._cleanup_worker(segmentation))

        # Start the worker
        worker.start()

        self.parent_widget.loading_workers[segmentation] = worker
        self.parent_widget.info_label.setText(f"Loading segmentation: {segmentation.name}...")

    def load_picks(self, pick_set: copick.models.CopickPicks, parent_run: Optional[copick.models.CopickRun]) -> None:
        """Load picks into napari as a points layer.

        Args:
            pick_set: The picks data to load
            parent_run: The parent run containing the picks
        """
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [(p.location.z, p.location.y, p.location.x) for p in pick_set.points]

                    # Find the matching pickable object to get the correct color
                    pickable_object = None
                    for obj in self.parent_widget.root.pickable_objects:
                        if obj.name == pick_set.pickable_object_name:
                            pickable_object = obj
                            break

                    if pickable_object:
                        color = pickable_object.color
                    else:
                        color = (255, 255, 255, 255)  # Default to white if no matching object found

                    colors = np.tile(
                        np.array(
                            [
                                color[0] / 255.0,
                                color[1] / 255.0,
                                color[2] / 255.0,
                                color[3] / 255.0,
                            ],
                        ),
                        (len(points), 1),
                    )

                    # TODO hardcoded default point size
                    point_size = pickable_object.radius if pickable_object.radius else 50
                    points_layer = self.parent_widget.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.pickable_object_name} ({pick_set.user_id} | {pick_set.session_id})",
                        size=point_size,
                        face_color=colors,
                        out_of_slice_display=True,
                    )
                    points_layer.size = [200] * len(points_layer.size)  # Set a default size for all points

                    # Store copick metadata in the layer for later use in save dialog
                    points_layer.metadata["copick_run"] = parent_run
                    points_layer.metadata["copick_picks"] = pick_set
                    points_layer.metadata["copick_source_object_name"] = pick_set.pickable_object_name
                    points_layer.metadata["copick_session_id"] = pick_set.session_id
                    points_layer.metadata["copick_user_id"] = pick_set.user_id

                    self.parent_widget.info_label.setText(f"Loaded Picks: {pick_set.pickable_object_name}")
                else:
                    self.parent_widget.info_label.setText(f"No points found for Picks: {pick_set.pickable_object_name}")
            else:
                self.parent_widget.info_label.setText(f"No pick set found for Picks: {pick_set.pickable_object_name}")
        else:
            self.parent_widget.info_label.setText("No parent run found")

    def _on_progress(self, message: str, data_object: Any, data_type: str) -> None:
        """Handle progress updates from workers.

        Args:
            message: Progress message
            data_object: The data object being loaded
            data_type: Type of data being loaded
        """
        self.parent_widget.info_label.setText(f"{message}")

    def _on_tomogram_loaded(self, result: Dict[str, Any]) -> None:
        """Handle successful tomogram loading.

        Args:
            result: The loading result containing tomogram data
        """
        tomogram = result["tomogram"]
        loaded_data = result["data"]
        voxel_size = result["voxel_size"]
        name = result["name"]
        resolution_level = result["resolution_level"]

        # Remove loading indicator (only for tree items)
        if tomogram in self.parent_widget.loading_items:
            item = self.parent_widget.loading_items[tomogram]
            if item is not None:
                self.parent_widget.tree_view.remove_loading_indicator(item)

        # Remove global loading indicator
        operation_id = f"load_tomogram_{tomogram.tomo_type}_{id(tomogram)}"
        self.parent_widget._remove_operation(operation_id)

        # Add pre-loaded image to the viewer (should be fast!)
        try:
            layer = self.parent_widget.viewer.add_image(
                loaded_data,
                scale=voxel_size,
                name=name,
            )
            layer.reset_contrast_limits()

            # Store copick metadata in the layer
            layer.metadata["copick_run"] = tomogram.voxel_spacing.run
            layer.metadata["copick_voxel_spacing"] = tomogram.voxel_spacing
            layer.metadata["copick_tomogram"] = tomogram
            layer.metadata["copick_resolution_level"] = resolution_level

            self.parent_widget.info_label.setText(
                f"Loaded Tomogram: {tomogram.tomo_type} (Resolution Level {resolution_level})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding image to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying tomogram: {str(e)}")

    def _on_segmentation_loaded(self, result: Dict[str, Any]) -> None:
        """Handle successful segmentation loading.

        Args:
            result: The loading result containing segmentation data
        """
        segmentation = result["segmentation"]
        loaded_data = result["data"]
        voxel_size = result["voxel_size"]
        name = result["name"]
        resolution_level = result["resolution_level"]

        # Remove loading indicator
        if segmentation in self.parent_widget.loading_items:
            item = self.parent_widget.loading_items[segmentation]
            self.parent_widget.tree_view.remove_loading_indicator(item)

        # Remove global loading indicator
        operation_id = f"load_segmentation_{segmentation.name}_{id(segmentation)}"
        self.parent_widget._remove_operation(operation_id)

        # Add pre-loaded segmentation to the viewer (should be fast!)
        try:
            # Create a color map based on copick colors
            if segmentation.is_multilabel:
                # For multilabel segmentations, use full colormap
                colormap = self.parent_widget.get_copick_colormap()
                painting_labels = [obj.label for obj in self.parent_widget.root.pickable_objects]
                class_labels_mapping = {obj.label: obj.name for obj in self.parent_widget.root.pickable_objects}
            else:
                # For single label segmentations, find the matching pickable object
                matching_obj = None
                for obj in self.parent_widget.root.pickable_objects:
                    if obj.name == segmentation.name:
                        matching_obj = obj
                        break

                if matching_obj:
                    # Create a simple colormap: 0 = background (black), 1 = object color
                    colormap = {
                        0: np.array([0, 0, 0, 0]),  # Transparent background
                        1: np.array(matching_obj.color) / 255.0,  # Object color
                    }
                    painting_labels = [1]  # Only allow painting with label 1
                    class_labels_mapping = {1: matching_obj.name}
                else:
                    # Fallback to default if no matching object found
                    colormap = {0: np.array([0, 0, 0, 0]), 1: np.array([1, 1, 1, 1])}
                    painting_labels = [1]
                    class_labels_mapping = {1: segmentation.name}

            painting_layer = self.parent_widget.viewer.add_labels(loaded_data, name=name, scale=voxel_size)
            painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
            painting_layer.painting_labels = painting_labels
            self.parent_widget.class_labels_mapping = class_labels_mapping

            # Store copick metadata in the layer
            painting_layer.metadata["copick_run"] = segmentation.run
            painting_layer.metadata["copick_segmentation"] = segmentation
            painting_layer.metadata["copick_voxel_size"] = segmentation.voxel_size
            painting_layer.metadata["copick_resolution_level"] = resolution_level
            painting_layer.metadata["copick_source_object_name"] = segmentation.name

            self.parent_widget.info_label.setText(
                f"Loaded Segmentation: {segmentation.name} (Resolution Level {resolution_level})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding segmentation to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying segmentation: {str(e)}")

    def _on_error(self, error_msg: str, data_object: Any, data_type: str) -> None:
        """Handle errors for loading operations.

        Args:
            error_msg: The error message
            data_object: The data object that failed to load
            data_type: The type of data that failed to load
        """
        if data_type == "tomogram":
            self.logger.exception(f"Tomogram loading error for {data_object.tomo_type}: {error_msg}")
        elif data_type == "segmentation":
            self.logger.exception(f"Segmentation loading error for {data_object.name}: {error_msg}")
        elif data_type == "run":
            self.logger.exception(f"Run expansion error for {data_object.name}: {error_msg}")
        elif data_type == "voxel_spacing":
            self.logger.exception(f"Voxel spacing expansion error for {data_object.voxel_size}: {error_msg}")

        # Remove global loading indicator for errors
        if data_type == "tomogram":
            operation_id = f"load_tomogram_{data_object.tomo_type}_{id(data_object)}"
            self.parent_widget._remove_operation(operation_id)
        elif data_type == "segmentation":
            operation_id = f"load_segmentation_{data_object.name}_{id(data_object)}"
            self.parent_widget._remove_operation(operation_id)
        elif data_type == "run":
            operation_id = f"expand_run_{data_object.name}"
            self.parent_widget._remove_operation(operation_id)
        elif data_type == "voxel_spacing":
            operation_id = f"expand_voxel_spacing_{data_object.voxel_size}"
            self.parent_widget._remove_operation(operation_id)

        # Remove loading indicator and clean up workers properly
        if data_object in self.parent_widget.loading_items:
            item = self.parent_widget.loading_items[data_object]
            if item is not None:
                self.parent_widget.tree_view.remove_loading_indicator(item)
            # Clean up loading worker
            self._cleanup_worker(data_object)
        elif data_object in self.parent_widget.expansion_items:
            item = self.parent_widget.expansion_items[data_object]
            self.parent_widget.tree_view.remove_loading_indicator(item)
            # Clean up expansion worker
            self._cleanup_expansion_worker(data_object)

        self.parent_widget.info_label.setText(f"Error: {error_msg}")

    def _cleanup_worker(self, data_object: Any) -> None:
        """Clean up loading worker and associated data.

        Args:
            data_object: The data object whose worker should be cleaned up
        """
        if data_object in self.parent_widget.loading_workers:
            del self.parent_widget.loading_workers[data_object]

        if data_object in self.parent_widget.loading_items:
            del self.parent_widget.loading_items[data_object]

    def _cleanup_expansion_worker(self, data_object: Any) -> None:
        """Clean up expansion worker and associated data.

        Args:
            data_object: The data object whose expansion worker should be cleaned up
        """
        if data_object in self.parent_widget.expansion_workers:
            del self.parent_widget.expansion_workers[data_object]

        if data_object in self.parent_widget.expansion_items:
            del self.parent_widget.expansion_items[data_object]
