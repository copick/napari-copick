"""Tree widget implementation for napari-copick plugin."""

import logging
from typing import Any, Dict, List, Optional

import copick
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
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
        self.customContextMenuRequested.connect(self.parent_widget.open_context_menu)

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
            self.parent_widget.load_tomogram_async(data, item)
        elif isinstance(data, copick.models.CopickSegmentation):
            self.parent_widget.load_segmentation_async(data, item)
        elif isinstance(data, copick.models.CopickPicks):
            parent_run = self.get_parent_run(item)
            self.parent_widget.load_picks(data, parent_run)

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
        worker.yielded.connect(lambda msg: self.parent_widget.on_progress(msg, run, "run"))
        worker.returned.connect(lambda result: self.on_run_expanded(result))
        worker.errored.connect(lambda e: self.parent_widget.on_error(str(e), run, "run"))
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

            # Add voxel spacings
            for voxel_spacing in voxel_spacings:
                spacing_item = QTreeWidgetItem(item, [f"Voxel Spacing: {voxel_spacing.voxel_size}"])
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

            # Add picks nested by user_id, session_id, and pickable_object_name
            picks_item = QTreeWidgetItem(item, ["Picks"])
            for user_id, sessions in picks_data.items():
                user_item = QTreeWidgetItem(picks_item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(user_item, [f"Session: {session_id}"])
                    for pick in picks:
                        pick_child = QTreeWidgetItem(session_item, [pick.pickable_object_name])
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

            # Restore expansion state for this run's children
            self.parent_widget.restore_expansion_for_item(item, run.name)

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
        worker.yielded.connect(lambda msg: self.parent_widget.on_progress(msg, voxel_spacing, "voxel_spacing"))
        worker.returned.connect(lambda result: self.on_voxel_spacing_expanded(result))
        worker.errored.connect(lambda e: self.parent_widget.on_error(str(e), voxel_spacing, "voxel_spacing"))
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

            # Add tomograms
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in tomograms:
                tomo_child = QTreeWidgetItem(tomogram_item, [tomogram.tomo_type])
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            # Add segmentations with user/session structure like picks
            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations_by_user_session = self.group_segmentations_by_user_session(segmentations)

            for user_id, sessions in segmentations_by_user_session.items():
                user_item = QTreeWidgetItem(segmentation_item, [f"User: {user_id}"])
                for session_id, segmentations_in_session in sessions.items():
                    session_item = QTreeWidgetItem(user_item, [f"Session: {session_id}"])
                    for segmentation in segmentations_in_session:
                        seg_child = QTreeWidgetItem(session_item, [segmentation.name])
                        seg_child.setData(0, Qt.UserRole, segmentation)
            item.addChild(segmentation_item)

            # Restore expansion state for this voxel spacing's children
            run_name = voxel_spacing.run.name
            voxel_path = f"{run_name}/Voxel Spacing: {voxel_spacing.voxel_size}"
            self.parent_widget.restore_expansion_for_item(item, voxel_path)

            self.parent_widget.info_label.setText(f"Expanded voxel spacing: {voxel_spacing.voxel_size}")

        # Remove global loading indicator
        operation_id = f"expand_voxel_spacing_{voxel_spacing.voxel_size}"
        self.parent_widget._remove_operation(operation_id)

    def group_segmentations_by_user_session(
        self,
        segmentations: List[copick.models.CopickSegmentation],
    ) -> Dict[str, Dict[str, List[copick.models.CopickSegmentation]]]:
        """Group segmentations by user_id and session_id like picks."""
        grouped = {}
        for segmentation in segmentations:
            user_id = segmentation.user_id
            session_id = segmentation.session_id

            if user_id not in grouped:
                grouped[user_id] = {}
            if session_id not in grouped[user_id]:
                grouped[user_id][session_id] = []

            grouped[user_id][session_id].append(segmentation)

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
        self.setItemWidget(item, 0, None)

        # Restore original text if stored
        original_text = item.data(0, Qt.UserRole + 1)
        if original_text:
            item.setText(0, original_text)
            item.setData(0, Qt.UserRole + 1, None)  # Clear stored text

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
