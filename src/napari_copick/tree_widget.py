"""Tree widget implementation for napari-copick plugin."""

import logging
from typing import Any, Dict, List, Optional

import copick
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAction,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from napari_copick.async_loaders import (
    expand_run_worker,
    expand_voxel_spacing_worker,
)

logger = logging.getLogger(__name__)


class CopickTreeWidget(QTreeWidget):
    """Custom tree widget for displaying copick data structure."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_widget = parent
        self.setHeaderLabel("Copick Project")

        # Track loading and expansion workers
        self.loading_workers: Dict[Any, Any] = {}
        self.loading_items: Dict[Any, Optional[QTreeWidgetItem]] = {}
        self.expansion_workers: Dict[Any, Any] = {}
        self.expansion_items: Dict[Any, QTreeWidgetItem] = {}

        # Connect signals
        self.itemExpanded.connect(self.handle_item_expand)
        self.itemClicked.connect(self.handle_item_click)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_context_menu)

    def populate_tree(self, root: copick.models.CopickRoot) -> None:
        """Populate the tree with runs from the copick root."""
        self.clear()
        for run in root.runs:
            run_item = QTreeWidgetItem(self, [run.name])
            run_item.setData(0, Qt.UserRole, run)
            run_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

    def handle_item_expand(self, item: QTreeWidgetItem) -> None:
        """Handle item expansion for runs and voxel spacings."""
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.expand_run_async(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing_async(item, data)

    def handle_item_click(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item clicks for different copick objects."""
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.parent_widget.info_label.setText(f"Run: {data.name}")
            self.parent_widget.selected_run = data
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.parent_widget.info_label.setText(f"Voxel Spacing: {data.voxel_size}")
            self.lazy_load_voxel_spacing(item, data)
        elif isinstance(data, copick.models.CopickTomogram):
            self.parent_widget.data_loader.load_tomogram_async(data, item)
        elif isinstance(data, copick.models.CopickSegmentation):
            self.parent_widget.data_loader.load_segmentation_async(data, item)
        elif isinstance(data, copick.models.CopickPicks):
            parent_run = self.get_parent_run(item)
            self.parent_widget.data_loader.load_picks(data, parent_run)

    def get_parent_run(self, item: QTreeWidgetItem) -> Optional[copick.models.CopickRun]:
        """Get the parent run for a given tree item."""
        while item:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, copick.models.CopickRun):
                return data
            item = item.parent()
        return None

    def lazy_load_voxel_spacing(self, item: QTreeWidgetItem, voxel_spacing: copick.models.CopickVoxelSpacing) -> None:
        """Lazy load voxel spacing if not already loaded."""
        if not item.childCount():
            self.expand_voxel_spacing_async(item, voxel_spacing)

    def expand_run_async(self, item: QTreeWidgetItem, run: copick.models.CopickRun) -> None:
        """Expand a run asynchronously with loading indicator."""
        # Skip if already expanded or currently expanding
        if item.childCount() > 0 or run in self.expansion_workers:
            return

        # Add loading indicators
        self.add_loading_indicator(item)
        self.expansion_items[run] = item

        # Add global loading indicator
        operation_id = f"expand_run_{run.name}"
        self.parent_widget._add_operation(operation_id, f"Expanding run: {run.name}...")

        # Create worker
        worker = expand_run_worker(run)

        # Connect signals
        worker.yielded.connect(lambda msg: self.parent_widget.data_loader._on_progress(msg, run, "run"))
        worker.returned.connect(lambda result: self.on_run_expanded(result))
        worker.errored.connect(lambda e: self.parent_widget.data_loader._on_error(str(e), run, "run"))
        worker.finished.connect(lambda: self.cleanup_expansion_worker(run))

        # Start the worker
        worker.start()
        self.expansion_workers[run] = worker
        self.parent_widget.info_label.setText(f"Expanding run: {run.name}...")

    def on_run_expanded(self, result: Dict[str, Any]) -> None:
        """Handle successful run expansion."""
        run = result["run"]
        voxel_spacings = result["voxel_spacings"]
        picks_data = result["picks_data"]

        # Remove loading indicator
        if run in self.expansion_items:
            item = self.expansion_items[run]
            self.remove_loading_indicator(item)

            # Check if item is still valid before proceeding
            try:
                item.text(0)  # Test if item is still valid
            except RuntimeError:
                # Item has been deleted, clean up and return
                self.cleanup_expansion_worker(run)
                return

            # Add voxel spacings
            for voxel_spacing in voxel_spacings:
                spacing_item = QTreeWidgetItem(item, [f"Voxel Spacing: {voxel_spacing.voxel_size}"])
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

            # Add picks nested by pickable_object_name, then "user_id | session_id"
            picks_item = QTreeWidgetItem(item, ["Picks"])

            # Group picks by object type first
            picks_by_object = {}
            for user_id, sessions in picks_data.items():
                for session_id, picks in sessions.items():
                    for pick in picks:
                        object_name = pick.pickable_object_name
                        if object_name not in picks_by_object:
                            picks_by_object[object_name] = []
                        picks_by_object[object_name].append((user_id, session_id, pick))

            # Create tree structure: Object Type > "User | Session"
            for object_name in sorted(picks_by_object.keys()):
                object_item = QTreeWidgetItem(picks_item, [object_name])

                # Group by user|session and sort by user
                user_session_picks = {}
                for user_id, session_id, pick in picks_by_object[object_name]:
                    user_session_key = f"{user_id} | {session_id}"
                    if user_session_key not in user_session_picks:
                        user_session_picks[user_session_key] = []
                    user_session_picks[user_session_key].append(pick)

                # Sort by user (first part before |)
                for user_session_key in sorted(user_session_picks.keys(), key=lambda x: x.split(" | ")[0]):
                    user_session_item = QTreeWidgetItem(object_item, [user_session_key])
                    # Set the first pick as the data (for backwards compatibility)
                    user_session_item.setData(0, Qt.UserRole, user_session_picks[user_session_key][0])

            item.addChild(picks_item)

            # Continue expansion restoration for newly created children
            self.parent_widget.tree_expansion_manager._force_expand_item_if_needed(item, "")

            self.parent_widget.info_label.setText(f"Expanded run: {run.name}")

        # Remove global loading indicator
        operation_id = f"expand_run_{run.name}"
        self.parent_widget._remove_operation(operation_id)

    def expand_voxel_spacing_async(
        self,
        item: QTreeWidgetItem,
        voxel_spacing: copick.models.CopickVoxelSpacing,
    ) -> None:
        """Expand a voxel spacing asynchronously with loading indicator."""
        # Skip if already expanded or currently expanding
        if item.childCount() > 0 or voxel_spacing in self.expansion_workers:
            return

        # Add loading indicator
        self.add_loading_indicator(item)
        self.expansion_items[voxel_spacing] = item

        # Add global loading indicator
        operation_id = f"expand_voxel_spacing_{voxel_spacing.voxel_size}"
        self.parent_widget._add_operation(operation_id, f"Expanding voxel spacing: {voxel_spacing.voxel_size}...")

        # Create worker
        worker = expand_voxel_spacing_worker(voxel_spacing)

        # Connect signals
        worker.yielded.connect(
            lambda msg: self.parent_widget.data_loader._on_progress(msg, voxel_spacing, "voxel_spacing"),
        )
        worker.returned.connect(lambda result: self.on_voxel_spacing_expanded(result))
        worker.errored.connect(
            lambda e: self.parent_widget.data_loader._on_error(str(e), voxel_spacing, "voxel_spacing"),
        )
        worker.finished.connect(lambda: self.cleanup_expansion_worker(voxel_spacing))

        # Start the worker
        worker.start()
        self.expansion_workers[voxel_spacing] = worker
        self.parent_widget.info_label.setText(f"Expanding voxel spacing: {voxel_spacing.voxel_size}...")

    def on_voxel_spacing_expanded(self, result: Dict[str, Any]) -> None:
        """Handle successful voxel spacing expansion."""
        voxel_spacing = result["voxel_spacing"]
        tomograms = result["tomograms"]
        segmentations = result["segmentations"]

        # Remove loading indicator
        if voxel_spacing in self.expansion_items:
            item = self.expansion_items[voxel_spacing]
            self.remove_loading_indicator(item)

            # Check if item is still valid before proceeding
            try:
                item.text(0)  # Test if item is still valid
            except RuntimeError:
                # Item has been deleted, clean up and return
                self.cleanup_expansion_worker(voxel_spacing)
                return

            # Add tomograms
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in tomograms:
                tomo_child = QTreeWidgetItem(tomogram_item, [tomogram.tomo_type])
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            # Add segmentations with object type > "user | session" structure
            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations_by_object = self.group_segmentations_by_object_type(segmentations)

            # Create tree structure: Object Type > "User | Session"
            for object_name in sorted(segmentations_by_object.keys()):
                object_item = QTreeWidgetItem(segmentation_item, [object_name])

                # Group by user|session and sort by user
                user_session_segmentations = {}
                for segmentation in segmentations_by_object[object_name]:
                    user_session_key = f"{segmentation.user_id} | {segmentation.session_id}"
                    if user_session_key not in user_session_segmentations:
                        user_session_segmentations[user_session_key] = []
                    user_session_segmentations[user_session_key].append(segmentation)

                # Sort by user (first part before |)
                for user_session_key in sorted(user_session_segmentations.keys(), key=lambda x: x.split(" | ")[0]):
                    user_session_item = QTreeWidgetItem(object_item, [user_session_key])
                    # Set the first segmentation as the data (for backwards compatibility)
                    user_session_item.setData(0, Qt.UserRole, user_session_segmentations[user_session_key][0])

            item.addChild(segmentation_item)

            # Continue expansion restoration for newly created children
            # Build the correct path prefix for the voxel spacing's parent
            parent_run = self.get_parent_run(item)
            if parent_run:
                # The path prefix should be the parent run name only
                # so that when _force_expand_item_if_needed processes the voxel spacing item,
                # it will create the correct path: "249/Voxel Spacing: 13.48"
                parent_path = parent_run.name
                self.parent_widget.tree_expansion_manager._force_expand_item_if_needed(item, parent_path)
            else:
                self.parent_widget.tree_expansion_manager._force_expand_item_if_needed(item, "")

            self.parent_widget.info_label.setText(f"Expanded voxel spacing: {voxel_spacing.voxel_size}")

        # Remove global loading indicator
        operation_id = f"expand_voxel_spacing_{voxel_spacing.voxel_size}"
        self.parent_widget._remove_operation(operation_id)

    def group_segmentations_by_object_type(
        self,
        segmentations: List[copick.models.CopickSegmentation],
    ) -> Dict[str, List[copick.models.CopickSegmentation]]:
        """Group segmentations by object type (name)."""
        grouped = {}
        for segmentation in segmentations:
            object_name = segmentation.name
            if object_name not in grouped:
                grouped[object_name] = []
            grouped[object_name].append(segmentation)
        return grouped

    def add_loading_indicator(self, item: QTreeWidgetItem) -> None:
        """Add a loading indicator to the tree item while preserving original text."""
        # Store original text
        original_text = item.text(0)
        item.setData(0, Qt.UserRole + 1, original_text)  # Store in custom role

        # Create a widget with text + progress bar
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)

        # Original text label
        text_label = QLabel(original_text)
        layout.addWidget(text_label)

        # Small progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate progress
        progress_bar.setMaximumHeight(12)
        progress_bar.setMaximumWidth(60)
        layout.addWidget(progress_bar)

        layout.addStretch()

        # Set the widget on the tree item
        self.setItemWidget(item, 0, widget)

    def remove_loading_indicator(self, item: QTreeWidgetItem) -> None:
        """Remove the loading indicator from the tree item and restore original text."""
        try:
            # Check if the item is still valid (not deleted)
            if item is None:
                return

            # Try to access the item to see if it's still valid
            item.text(0)

            self.setItemWidget(item, 0, None)

            # Restore original text if stored
            original_text = item.data(0, Qt.UserRole + 1)
            if original_text:
                item.setText(0, original_text)
                item.setData(0, Qt.UserRole + 1, None)  # Clear stored text
        except RuntimeError:
            # Item has been deleted, ignore
            pass

    def cleanup_expansion_worker(self, data_object: Any) -> None:
        """Clean up expansion worker and associated data."""
        if data_object in self.expansion_workers:
            del self.expansion_workers[data_object]

        if data_object in self.expansion_items:
            del self.expansion_items[data_object]

    def cleanup_workers(self) -> None:
        """Stop and clean up all active workers."""
        # Clean up expansion workers
        for worker in list(self.expansion_workers.values()):
            if hasattr(worker, "quit"):
                worker.quit()
        self.expansion_workers.clear()
        self.expansion_items.clear()

    def open_context_menu(self, position) -> None:
        """Open context menu for right-click on tree items."""
        item = self.itemAt(position)
        if not item:
            return

        data = item.data(0, Qt.UserRole)

        # Only show delete option for picks and segmentations
        if isinstance(data, (copick.models.CopickPicks, copick.models.CopickSegmentation)):
            menu = QMenu(self)

            # Delete action
            delete_action = QAction("🗑️ Delete", self)
            delete_action.triggered.connect(lambda: self.delete_item(item, data))
            menu.addAction(delete_action)

            # Show menu at cursor position
            menu.exec_(self.mapToGlobal(position))

    def delete_item(self, item: QTreeWidgetItem, data) -> None:
        """Delete a specific picks or segmentation item."""
        if isinstance(data, copick.models.CopickPicks):
            item_type = "picks"
            item_name = f"{data.pickable_object_name} picks ({data.user_id} | {data.session_id})"
        elif isinstance(data, copick.models.CopickSegmentation):
            item_type = "segmentation"
            item_name = f"{data.name} segmentation ({data.user_id} | {data.session_id})"
        else:
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the {item_type}:\n\n{item_name}\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                # Create item structure for delete_items_async
                item_data = {
                    "type": item_type,
                    "object_name": data.pickable_object_name if hasattr(data, "pickable_object_name") else data.name,
                    "user_session": f"{data.user_id} | {data.session_id}",
                }

                if item_type == "picks":
                    item_data["picks"] = [data]
                else:
                    item_data["segmentations"] = [data]

                # Call the delete function
                self.parent_widget.save_manager.delete_items_async([item_data])

            except Exception as e:
                self.parent_widget.info_label.setText(f"Error deleting {item_type}: {str(e)}")
                logger.exception(f"Error deleting {item_type}: {str(e)}")
