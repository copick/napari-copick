"""Data loading operations for napari-copick plugin."""

import logging
from typing import Any, Dict, Optional

import copick
import numpy as np
from napari.utils import DirectLabelColormap
from qtpy.QtWidgets import QTreeWidgetItem

from napari_copick.async_loaders import load_filaments_worker, load_segmentation_worker, load_tomogram_worker
from napari_copick.filament_layers import add_centreline_layer
from napari_copick.instance_layers import (
    BBOXES_KEY,
    COLOR_MODE_KEY,
    COUNTS_KEY,
    PANOPTIC_ROWS_KEY,
    SEGMENTATION_TYPE_KEY,
    instance_colormap,
    segment_colormap,
)
from napari_copick.pick_layers import (
    FEATURE_DEFAULTS,
    TRANSFORMS_KEY,
    instance_colors,
    is_filament,
    picks_to_layer_data,
    reset_added_features,
)
from napari_copick.slicing import points_slicing, show_points_within, single_slice


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

    def load_segmentation_async(
        self,
        segmentation: copick.models.CopickSegmentation,
        item: Optional[QTreeWidgetItem] = None,
    ) -> None:
        """Load a segmentation asynchronously with loading indicator using napari's threading system.

        Args:
            segmentation: The segmentation to load
            item: The tree widget item associated with the segmentation (optional)
        """
        # Check if already loading
        if segmentation in self.parent_widget.loading_workers:
            self.logger.warning(f"Segmentation {segmentation.name} already loading, skipping")
            return

        # Add loading indicator
        if item is not None:
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
                    # Points are drawn at the particle centre, location + t.
                    points, features, transforms = picks_to_layer_data(pick_set.points)

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
                    if pickable_object is not None and is_filament(pickable_object):
                        # One colour per filament, so neighbouring filaments can be told apart.
                        colors = instance_colors(features["instance_id"], colors[0])

                    # TODO hardcoded default point size
                    point_size = pickable_object.radius if pickable_object and pickable_object.radius else 50
                    points_layer = self.parent_widget.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.pickable_object_name} ({pick_set.user_id} | {pick_set.session_id})",
                        size=point_size,
                        face_color=colors,
                        features=features,
                        feature_defaults=FEATURE_DEFAULTS,
                        **points_slicing(near=True),
                    )
                    show_points_within(self.parent_widget.viewer, point_size)
                    # Points added later are new picks, whatever point was selected when they were placed.
                    points_layer.events.data.connect(lambda event: reset_added_features(points_layer, event))

                    # Store copick metadata in the layer for later use in save dialog
                    points_layer.metadata["copick_run"] = parent_run
                    points_layer.metadata["copick_picks"] = pick_set
                    points_layer.metadata["copick_source_object_name"] = pick_set.pickable_object_name
                    points_layer.metadata["copick_session_id"] = pick_set.session_id
                    points_layer.metadata["copick_user_id"] = pick_set.user_id
                    points_layer.metadata[TRANSFORMS_KEY] = transforms
                    points_layer.metadata["copick_kind"] = "picks"
                    points_layer.metadata["copick_base_color"] = tuple(np.asarray(color, dtype=float) / 255.0)
                    filament_obj = pickable_object is not None and is_filament(pickable_object)
                    points_layer.metadata[COLOR_MODE_KEY] = "instance" if filament_obj else "object"

                    self.parent_widget.info_label.setText(f"Loaded Picks: {pick_set.pickable_object_name}")
                else:
                    self.parent_widget.info_label.setText(f"No points found for Picks: {pick_set.pickable_object_name}")
            else:
                self.parent_widget.info_label.setText(f"No pick set found for Picks: {pick_set.pickable_object_name}")
        else:
            self.parent_widget.info_label.setText("No parent run found")

    def load_filaments_async(self, filaments: Any, item: Optional[QTreeWidgetItem] = None) -> None:
        """Load a filament set (traced centrelines) as a points layer.

        Args:
            filaments: The ``CopickFilaments`` to load
            item: The tree widget item associated with the filaments (optional)
        """
        if filaments in self.parent_widget.loading_workers:
            return
        if item is not None:
            self.parent_widget.tree_view.add_loading_indicator(item)
        self.parent_widget.loading_items[filaments] = item
        operation_id = f"load_filaments_{id(filaments)}"
        self.parent_widget._add_operation(operation_id, f"Loading filaments: {filaments.pickable_object_name}...")

        worker = load_filaments_worker(filaments)
        worker.yielded.connect(lambda msg: self._on_progress(msg, filaments, "filaments"))
        worker.returned.connect(lambda result: self._on_filaments_loaded(result))
        worker.errored.connect(lambda e: self._on_error(str(e), filaments, "filaments"))
        worker.finished.connect(lambda: self._cleanup_worker(filaments))
        worker.start()
        self.parent_widget.loading_workers[filaments] = worker

    def _on_filaments_loaded(self, result: Dict[str, Any]) -> None:
        filaments = result["filaments"]
        session = result["session"]
        item = self.parent_widget.loading_items.get(filaments)
        if item is not None:
            self.parent_widget.tree_view.remove_loading_indicator(item)
        self.parent_widget._remove_operation(f"load_filaments_{id(filaments)}")
        try:
            layer = add_centreline_layer(self.parent_widget.viewer, session, run=filaments.run)
            layer.metadata["copick_filaments"] = filaments
            self.parent_widget.info_label.setText(
                f"Loaded {len(session.to_list())} filaments: {filaments.pickable_object_name} "
                f"({filaments.user_id} | {filaments.session_id})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding filaments to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying filaments: {str(e)}")

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
            single_slice(layer)  # thick slices (for points) don't average the tomogram

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
        seg_type = result.get("segmentation_type", "multilabel" if segmentation.is_multilabel else "binary")

        # Remove loading indicator
        if segmentation in self.parent_widget.loading_items:
            item = self.parent_widget.loading_items[segmentation]
            if item is not None:
                self.parent_widget.tree_view.remove_loading_indicator(item)

        # Remove global loading indicator
        operation_id = f"load_segmentation_{segmentation.name}_{id(segmentation)}"
        self.parent_widget._remove_operation(operation_id)

        if seg_type == "instance":
            self._add_instance_segmentation(result)
            return
        if seg_type == "panoptic":
            self._add_panoptic_segmentation(result)
            return

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
            painting_layer.metadata[SEGMENTATION_TYPE_KEY] = seg_type

            self.parent_widget.info_label.setText(
                f"Loaded Segmentation: {segmentation.name} (Resolution Level {resolution_level})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding segmentation to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying segmentation: {str(e)}")

    def _common_segmentation_metadata(self, layer: Any, result: Dict[str, Any], seg_type: str) -> None:
        segmentation = result["segmentation"]
        layer.metadata["copick_run"] = segmentation.run
        layer.metadata["copick_segmentation"] = segmentation
        layer.metadata["copick_voxel_size"] = segmentation.voxel_size
        layer.metadata["copick_resolution_level"] = result["resolution_level"]
        layer.metadata["copick_source_object_name"] = segmentation.name
        layer.metadata[SEGMENTATION_TYPE_KEY] = seg_type

    def _add_instance_segmentation(self, result: Dict[str, Any]) -> None:
        """An instance segmentation: a labels layer of instance IDs, each its own colour (as its picks/filaments)."""
        segmentation = result["segmentation"]
        try:
            counts = result.get("counts", {})
            layer = self.parent_widget.viewer.add_labels(
                result["data"],
                name=result["name"],
                scale=result["voxel_size"],
            )
            layer.colormap = instance_colormap(counts.keys())
            self._common_segmentation_metadata(layer, result, "instance")
            layer.metadata[COUNTS_KEY] = counts
            layer.metadata[BBOXES_KEY] = result.get("bboxes", {})
            layer.selected_label = (max(counts) + 1) if counts else 1
            self.parent_widget.info_label.setText(
                f"Loaded instance segmentation: {segmentation.name}, {len(counts)} instances "
                f"(Resolution Level {result['resolution_level']})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding instance segmentation to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying instance segmentation: {str(e)}")

    def _add_panoptic_segmentation(self, result: Dict[str, Any]) -> None:
        """A panoptic segmentation (display only): its label channel with object colours, and a segment layer where
        each (object, instance) has its own colour and the status bar names it."""
        segmentation = result["segmentation"]
        try:
            viewer = self.parent_widget.viewer
            labels = viewer.add_labels(result["data"], name=result["name"], scale=result["voxel_size"])
            labels.colormap = DirectLabelColormap(color_dict=self.parent_widget.get_copick_colormap())
            labels.editable = False
            self._common_segmentation_metadata(labels, result, "panoptic")

            rows = result["rows"]
            segments = viewer.add_labels(
                result["segments"],
                name=result["name"].replace("Panoptic:", "Panoptic segments:", 1),
                scale=result["voxel_size"],
            )
            segments.colormap = segment_colormap(rows)
            segments.editable = False
            try:
                import pandas as pd

                segments.features = pd.DataFrame(
                    {
                        "index": [0] + [r.row_key for r in rows],
                        "object": ["background"] + [r.label for r in rows],
                        "instance_id": [0] + [r.instance_id for r in rows],
                    },
                )
            except Exception as e:  # features are a convenience for the status bar
                self.logger.warning(f"Could not set panoptic segment features: {e}")
            self._common_segmentation_metadata(segments, result, "panoptic")
            segments.metadata["copick_kind"] = "panoptic_segments"
            segments.metadata[PANOPTIC_ROWS_KEY] = rows
            segments.metadata[BBOXES_KEY] = result.get("bboxes", {})
            self.parent_widget.info_label.setText(
                f"Loaded panoptic segmentation: {segmentation.name}, {len(rows)} segments "
                f"(Resolution Level {result['resolution_level']})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding panoptic segmentation to viewer: {str(e)}")
            self.parent_widget.info_label.setText(f"Error displaying panoptic segmentation: {str(e)}")

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
        elif data_type == "filaments":
            self.logger.exception(f"Filaments loading error for {data_object.pickable_object_name}: {error_msg}")

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
        elif data_type == "filaments":
            self.parent_widget._remove_operation(f"load_filaments_{id(data_object)}")

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
