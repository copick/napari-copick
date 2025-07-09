import logging
from typing import Any, Dict, List, Optional, Set, Union

import copick
import napari
import numpy as np
import zarr
from napari.layers import Labels, Points
from napari.utils import DirectLabelColormap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_copick.async_loaders import (
    expand_run_worker,
    load_segmentation_worker,
    load_tomogram_worker,
    save_segmentation_worker,
)
from napari_copick.dialogs import DatasetIdDialog, SaveLayerDialog, SaveSegmentationDialog
from napari_copick.save_utils import get_runs_from_open_layers, save_picks_to_copick
from napari_copick.tree_widget import CopickTreeWidget

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
    # Fallback if shared component is not available
    EditObjectTypesDialog = None

# Import the gallery widget
try:
    from napari_copick.gallery_widget import NapariCopickGalleryWidget

    GALLERY_AVAILABLE = True
except ImportError:
    GALLERY_AVAILABLE = False

# Import the info widget
try:
    from napari_copick.info_widget import NapariCopickInfoWidget

    INFO_AVAILABLE = True
except ImportError:
    INFO_AVAILABLE = False


class CopickPlugin(QWidget):
    def __init__(
        self,
        viewer: Optional[napari.viewer.Viewer] = None,
        config_path: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        overlay_root: str = "/tmp/overlay_root",
    ) -> None:
        super().__init__()

        # Setup logging
        self.logger = logging.getLogger("CopickPlugin")

        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()

        self.root: Optional[copick.models.CopickRoot] = None
        self.selected_run: Optional[copick.models.CopickRun] = None
        self.current_layer: Optional[Any] = None
        self.session_id: str = "17"
        self.loading_workers: Dict[Any, Any] = {}  # Track active loading workers
        self.loading_items: Dict[Any, Optional[QTreeWidgetItem]] = {}  # Track tree items being loaded
        self.expansion_workers: Dict[Any, Any] = {}  # Track active expansion workers
        self.expansion_items: Dict[Any, QTreeWidgetItem] = {}  # Track tree items being expanded
        self.tree_expansion_state: Dict[str, bool] = {}  # Track expanded items by path
        self.setup_ui()

        if config_path:
            self.load_config(config_path=config_path)
        elif dataset_ids:
            self.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Config loading options
        load_options_layout = QHBoxLayout()

        # Config file button
        self.load_config_button = QPushButton("Load Config File")
        self.load_config_button.clicked.connect(self.open_file_dialog)
        load_options_layout.addWidget(self.load_config_button)

        # Dataset IDs button
        self.load_dataset_button = QPushButton("Load from Dataset IDs")
        self.load_dataset_button.clicked.connect(self.open_dataset_dialog)
        load_options_layout.addWidget(self.load_dataset_button)

        layout.addLayout(load_options_layout)

        # Edit Object Types button
        self.edit_objects_button = QPushButton("✏️ Edit Object Types")
        self.edit_objects_button.clicked.connect(self.open_edit_objects_dialog)
        self.edit_objects_button.setEnabled(False)  # Disabled until config is loaded
        self.edit_objects_button.setToolTip("Edit or add new object types in the configuration")
        layout.addWidget(self.edit_objects_button)

        # Create tab widget for tree and gallery views
        self.tab_widget = QTabWidget()

        # Tree view tab
        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)

        # Hierarchical tree view
        self.tree_view = CopickTreeWidget(self)
        tree_layout.addWidget(self.tree_view)

        # Save buttons layout
        save_buttons_layout = QHBoxLayout()

        # Save segmentation button
        self.save_segmentation_button = QPushButton("💾 Save Segmentation")
        self.save_segmentation_button.clicked.connect(self.open_save_segmentation_dialog)
        self.save_segmentation_button.setEnabled(False)  # Disabled until config is loaded
        self.save_segmentation_button.setToolTip("Save a segmentation layer to copick")
        save_buttons_layout.addWidget(self.save_segmentation_button)

        # Save picks button
        self.save_picks_button = QPushButton("📍 Save Picks")
        self.save_picks_button.clicked.connect(self.open_save_picks_dialog)
        self.save_picks_button.setEnabled(False)  # Disabled until config is loaded
        self.save_picks_button.setToolTip("Save a points layer to copick")
        save_buttons_layout.addWidget(self.save_picks_button)

        tree_layout.addLayout(save_buttons_layout)

        self.tab_widget.addTab(tree_tab, "🌲 Tree View")

        # Gallery view tab
        if GALLERY_AVAILABLE:
            self.gallery_widget = NapariCopickGalleryWidget(self.viewer, self)
            self.tab_widget.addTab(self.gallery_widget, "📸 Gallery View")

            # Connect gallery signals to navigate to info view
            self.gallery_widget.info_requested.connect(self._on_info_requested)
        else:
            # Fallback if gallery is not available
            gallery_fallback = QWidget()
            fallback_layout = QVBoxLayout(gallery_fallback)
            fallback_label = QLabel("Gallery view not available\n\nThe copick-shared-ui package is required.")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
            fallback_layout.addWidget(fallback_label)
            self.tab_widget.addTab(gallery_fallback, "📸 Gallery View")

        # Info view tab
        if INFO_AVAILABLE:
            self.info_widget = NapariCopickInfoWidget(self.viewer, self)
            self.tab_widget.addTab(self.info_widget, "📋 Info View")
        else:
            # Fallback if info widget is not available
            info_fallback = QWidget()
            fallback_layout = QVBoxLayout(info_fallback)
            fallback_label = QLabel("Info view not available\n\nThe copick-shared-ui package is required.")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
            fallback_layout.addWidget(fallback_label)
            self.tab_widget.addTab(info_fallback, "📋 Info View")

        layout.addWidget(self.tab_widget)

        # Resolution level selector
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("Image Resolution:")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(
            ["0 - Highest (Full Resolution)", "1 - Medium (Binned by 2)", "2 - Lowest (Binned by 4)"],
        )
        self.resolution_combo.setCurrentIndex(1)  # Default to medium resolution
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        layout.addLayout(resolution_layout)

        # Info label with fixed width and word wrap
        self.info_label = QLabel("Select a pick to get started")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumWidth(600)
        self.info_label.setMinimumHeight(20)
        layout.addWidget(self.info_label)

        # Global loading indicator
        self.loading_widget = QWidget()
        loading_layout = QHBoxLayout(self.loading_widget)
        loading_layout.setContentsMargins(5, 5, 5, 5)

        self.loading_label = QLabel("Loading...")
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)  # Indeterminate progress
        self.loading_progress.setMaximumHeight(20)

        loading_layout.addWidget(self.loading_label)
        loading_layout.addWidget(self.loading_progress)

        # Initially hidden
        self.loading_widget.setVisible(False)
        layout.addWidget(self.loading_widget)

        self.setLayout(layout)

        # Track active loading operations
        self.active_operations: Set[str] = set()  # Set of operation identifiers

    def _add_operation(self, operation_id: str, description: str = "Loading...") -> None:
        """Add an operation to the active operations and show global loading indicator."""
        self.active_operations.add(operation_id)
        self.loading_label.setText(description)
        self.loading_widget.setVisible(True)

    def _remove_operation(self, operation_id: str) -> None:
        """Remove an operation from active operations and hide loading indicator if none remain."""
        self.active_operations.discard(operation_id)
        if not self.active_operations:
            self.loading_widget.setVisible(False)

    def _update_loading_status(self, description: str) -> None:
        """Update the loading status description if operations are active."""
        if self.active_operations:
            self.loading_label.setText(description)

    def closeEvent(self, event: Any) -> None:
        """Clean up workers when widget is closed."""
        self.cleanup_workers()
        super().closeEvent(event)

    def cleanup_workers(self) -> None:
        """Stop and clean up all active workers."""
        # Clean up loading workers
        for worker in list(self.loading_workers.values()):
            if hasattr(worker, "quit"):
                worker.quit()
        self.loading_workers.clear()
        self.loading_items.clear()

        # Clean up expansion workers
        for worker in list(self.expansion_workers.values()):
            if hasattr(worker, "quit"):
                worker.quit()
        self.expansion_workers.clear()
        self.expansion_items.clear()

        # Clean up tree widget workers
        if hasattr(self, "tree_view"):
            self.tree_view.cleanup_workers()

        # Clean up shared UI components' workers
        if GALLERY_AVAILABLE and hasattr(self, "gallery_widget"):
            try:
                # Access the worker interface through the gallery integration
                if hasattr(self.gallery_widget, "gallery_integration"):
                    worker_interface = self.gallery_widget.gallery_integration.worker_interface
                    if hasattr(worker_interface, "shutdown_workers"):
                        worker_interface.shutdown_workers(timeout_ms=1000)
            except Exception as e:
                print(f"Warning: Could not cleanup gallery workers: {e}")

        if INFO_AVAILABLE and hasattr(self, "info_widget"):
            try:
                # Access the worker interface through the info widget
                if hasattr(self.info_widget, "worker_interface"):
                    worker_interface = self.info_widget.worker_interface
                    if hasattr(worker_interface, "shutdown_workers"):
                        worker_interface.shutdown_workers(timeout_ms=1000)
            except Exception as e:
                print(f"Warning: Could not cleanup info workers: {e}")

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "JSON Files (*.json)")
        if path:
            self.load_config(config_path=path)

    def open_dataset_dialog(self) -> None:
        dialog = DatasetIdDialog(self)
        if dialog.exec_():
            dataset_ids, overlay_root = dialog.get_values()
            if dataset_ids:
                self.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def open_edit_objects_dialog(self) -> None:
        """Open the EditObjectTypesDialog to manage object types"""
        if not self.root or not self.root.config:
            self.info_label.setText("No configuration loaded. Please load a config first.")
            return

        if EditObjectTypesDialog is None:
            self.info_label.setText("EditObjectTypesDialog is not available. Shared component may not be installed.")
            return

        try:
            dialog = EditObjectTypesDialog(self, self.root.pickable_objects)
            if dialog.exec_() == QDialog.Accepted:
                # Get the updated objects from the dialog
                updated_objects = dialog.get_objects()

                # Update the configuration
                self.root.pickable_objects = updated_objects

                # Update any UI elements that depend on the object types
                self.populate_tree(preserve_expansion=True)  # Refresh the tree view

                # Update any loaded segmentation layers with new colormap
                for layer in self.viewer.layers:
                    if hasattr(layer, "colormap") and "Segmentation:" in layer.name:
                        layer.colormap = DirectLabelColormap(color_dict=self.get_copick_colormap())
                        layer.painting_labels = [obj.label for obj in self.root.pickable_objects]

                self.info_label.setText(f"Updated {len(updated_objects)} object types in configuration")
        except Exception as e:
            self.info_label.setText(f"Error opening EditObjectTypesDialog: {str(e)}")

    def load_config(self, config_path: Optional[str] = None) -> None:
        if config_path:
            self.root = copick.from_file(config_path)

            # Initialize thumbnail cache with config file
            if set_global_cache_config:
                set_global_cache_config(config_path, app_name="copick")

                # Set up image interface for thumbnail cache
                if set_global_cache_image_interface and get_image_interface:
                    image_interface = get_image_interface()
                    if image_interface:
                        set_global_cache_image_interface(image_interface, app_name="copick")

            self.populate_tree(preserve_expansion=False)  # Initial load, no state to preserve
            self._update_gallery()
            self.edit_objects_button.setEnabled(True)  # Enable the button when config is loaded
            self.save_segmentation_button.setEnabled(True)  # Enable save buttons when config is loaded
            self.save_picks_button.setEnabled(True)
            self.info_label.setText(f"Loaded config from {config_path}")

    def load_from_dataset_ids(
        self,
        dataset_ids: Optional[List[str]] = None,
        overlay_root: str = "/tmp/overlay_root",
    ) -> None:
        if dataset_ids:
            self.root = copick.from_czcdp_datasets(
                dataset_ids=dataset_ids,
                overlay_root=overlay_root,
                overlay_fs_args={"auto_mkdir": True},
            )

            # Initialize thumbnail cache with dataset-based config
            if set_global_cache_config:
                # For dataset-based configs, use a unique cache key based on dataset IDs
                cache_key = f"datasets_{'-'.join(map(str, dataset_ids))}"
                set_global_cache_config(cache_key, app_name="copick")

                # Set up image interface for thumbnail cache
                if set_global_cache_image_interface and get_image_interface:
                    image_interface = get_image_interface()
                    if image_interface:
                        set_global_cache_image_interface(image_interface, app_name="copick")

            self.populate_tree(preserve_expansion=False)  # Initial load, no state to preserve
            self._update_gallery()
            self.edit_objects_button.setEnabled(True)  # Enable the button when config is loaded
            self.save_segmentation_button.setEnabled(True)  # Enable save buttons when config is loaded
            self.save_picks_button.setEnabled(True)
            self.info_label.setText(f"Loaded project from dataset IDs: {', '.join(map(str, dataset_ids))}")

    def save_tree_expansion_state(self) -> None:
        """Save the current expansion state of the tree."""
        self.tree_expansion_state.clear()

        def save_item_state(item: QTreeWidgetItem, path: str = "") -> None:
            """Recursively save expansion state for an item and its children."""
            current_path = f"{path}/{item.text(0)}" if path else item.text(0)

            if item.isExpanded():
                self.tree_expansion_state[current_path] = True
                self.logger.debug(f"Saved expanded state for: {current_path}")

            # Save state for children
            for i in range(item.childCount()):
                child = item.child(i)
                save_item_state(child, current_path)

        # Save state for all top-level items
        for i in range(self.tree_view.topLevelItemCount()):
            item = self.tree_view.topLevelItem(i)
            save_item_state(item)

        self.logger.debug(f"Saved {len(self.tree_expansion_state)} expanded items")

    def restore_tree_expansion_state(self) -> None:
        """Restore the previously saved expansion state of the tree."""
        if not self.tree_expansion_state:
            return

        # Use a delayed restoration to ensure items are properly loaded
        def delayed_restore():
            self._restore_all_expansion_states()

        # Schedule restoration after a short delay
        from qtpy.QtCore import QTimer

        QTimer.singleShot(100, delayed_restore)

    def _restore_all_expansion_states(self) -> None:
        """Internal method to restore all expansion states."""

        def restore_item_recursive(item: QTreeWidgetItem, path: str = "") -> None:
            current_path = f"{path}/{item.text(0)}" if path else item.text(0)

            # Check if this item should be expanded
            if current_path in self.tree_expansion_state:
                item.setExpanded(True)
                self.logger.debug(f"Restored expansion for: {current_path}")

            # Recursively restore children
            for i in range(item.childCount()):
                child = item.child(i)
                restore_item_recursive(child, current_path)

        # Restore all top-level items and their children
        for i in range(self.tree_view.topLevelItemCount()):
            item = self.tree_view.topLevelItem(i)
            restore_item_recursive(item)

    def restore_expansion_for_item(self, item: QTreeWidgetItem, parent_path: str = "") -> None:
        """Restore expansion state for a specific item and its children."""
        if not self.tree_expansion_state:
            return

        def restore_item_state(current_item: QTreeWidgetItem, path: str = "") -> None:
            """Recursively restore expansion state for an item and its children."""
            current_path = f"{path}/{current_item.text(0)}" if path else current_item.text(0)

            # Check if this item should be expanded
            if current_path in self.tree_expansion_state:
                current_item.setExpanded(True)
                self.logger.debug(f"Restored expanded state for: {current_path}")

            # Restore state for children
            for i in range(current_item.childCount()):
                child = current_item.child(i)
                restore_item_state(child, current_path)

        # Start restoration from the given item
        restore_item_state(item, parent_path)

        # Also process events to ensure expansion is visible
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

    def refresh_tree_after_save(self, save_result: Dict[str, Any]) -> None:
        """Refresh tree after saving, attempting to preserve expansion state."""
        try:
            # For now, fall back to full tree refresh with expansion preservation
            # In the future, this could be optimized to only refresh specific branches
            self.populate_tree(preserve_expansion=True)
        except Exception as e:
            self.logger.warning(f"Could not preserve expansion state during refresh: {e}")
            # Fall back to regular tree population
            self.populate_tree(preserve_expansion=False)

    def populate_tree(self, preserve_expansion: bool = True) -> None:
        """Populate the tree, optionally preserving expansion state."""
        if preserve_expansion:
            # Save current expansion state before repopulating
            self.save_tree_expansion_state()

        # Populate the tree
        self.tree_view.populate_tree(self.root)

        if preserve_expansion:
            # Restore expansion state after populating
            self.restore_tree_expansion_state()

    def handle_item_expand(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.expand_run_async(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing_async(item, data)

    def expand_run_async(self, item: QTreeWidgetItem, run: copick.models.CopickRun) -> None:
        """
        Expand a run asynchronously with loading indicator.
        """
        # Skip if already expanded or currently expanding
        if item.childCount() > 0 or run in self.expansion_workers:
            return

        # Add loading indicators
        self.add_loading_indicator(item)
        self.expansion_items[run] = item

        # Add global loading indicator
        operation_id = f"expand_run_{run.name}"
        self._add_operation(operation_id, f"Expanding run: {run.name}...")

        # Create worker
        worker = expand_run_worker(run)

        # Connect signals
        worker.yielded.connect(lambda msg: self.on_progress(msg, run, "run"))
        worker.returned.connect(lambda result: self.on_run_expanded(result))
        worker.errored.connect(lambda e: self.on_error(str(e), run, "run"))
        worker.finished.connect(lambda: self.cleanup_expansion_worker(run))

        # Start the worker
        worker.start()
        self.expansion_workers[run] = worker
        self.info_label.setText(f"Expanding run: {run.name}...")

    def on_run_expanded(self, result: Dict[str, Any]) -> None:
        """
        Handle successful run expansion.
        """
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
            self.restore_expansion_for_item(item, run.name)

            self.info_label.setText(f"Expanded run: {run.name}")

        # Remove global loading indicator
        operation_id = f"expand_run_{run.name}"
        self._remove_operation(operation_id)

    def load_tomogram_async(self, tomogram: copick.models.CopickTomogram, item: Optional[QTreeWidgetItem]) -> None:
        """
        Load a tomogram asynchronously with loading indicator using napari's threading system.
        """

        # Check if already loading
        if tomogram in self.loading_workers:
            self.logger.warning(f"Tomogram {tomogram.tomo_type} already loading, skipping")
            return

        # Add loading indicators
        if item is not None:
            # Add tree-specific loading indicator only if we have a tree item
            self.add_loading_indicator(item)
            self.loading_items[tomogram] = item
        else:
            # For cases where no tree item is available (e.g., info widget clicks)
            self.loading_items[tomogram] = None

        # Add global loading indicator (always show this)
        operation_id = f"load_tomogram_{tomogram.tomo_type}_{id(tomogram)}"
        self._add_operation(operation_id, f"Loading tomogram: {tomogram.tomo_type}...")

        # Get selected resolution level
        resolution_level = self.resolution_combo.currentIndex()

        # Create worker using napari's threading system
        worker = load_tomogram_worker(tomogram, resolution_level)

        # Connect signals
        worker.yielded.connect(lambda msg: self.on_progress(msg, tomogram, "tomogram"))
        worker.returned.connect(lambda result: self.on_tomogram_loaded(result))
        worker.errored.connect(lambda e: self.on_error(str(e), tomogram, "tomogram"))
        worker.finished.connect(lambda: self.cleanup_worker(tomogram))

        # Start the worker
        worker.start()

        self.loading_workers[tomogram] = worker
        self.info_label.setText(f"Loading tomogram: {tomogram.tomo_type}...")

    def add_loading_indicator(self, item: QTreeWidgetItem) -> None:
        """
        Add a loading indicator to the tree item while preserving original text.
        """
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
        self.tree_view.setItemWidget(item, 0, widget)

    def remove_loading_indicator(self, item: QTreeWidgetItem) -> None:
        """
        Remove the loading indicator from the tree item and restore original text.
        """
        self.tree_view.setItemWidget(item, 0, None)

        # Restore original text if stored
        original_text = item.data(0, Qt.UserRole + 1)
        if original_text:
            item.setText(0, original_text)
            item.setData(0, Qt.UserRole + 1, None)  # Clear stored text

    def load_segmentation_async(self, segmentation: copick.models.CopickSegmentation, item: QTreeWidgetItem) -> None:
        """
        Load a segmentation asynchronously with loading indicator using napari's threading system.
        """

        # Check if already loading
        if segmentation in self.loading_workers:
            self.logger.warning(f"Segmentation {segmentation.name} already loading, skipping")
            return

        # Add loading indicator
        self.add_loading_indicator(item)
        self.loading_items[segmentation] = item

        # Add global loading indicator
        operation_id = f"load_segmentation_{segmentation.name}_{id(segmentation)}"
        self._add_operation(operation_id, f"Loading segmentation: {segmentation.name}...")

        # Get selected resolution level
        resolution_level = self.resolution_combo.currentIndex()

        # Create worker using napari's threading system
        worker = load_segmentation_worker(segmentation, resolution_level)

        # Connect signals
        worker.yielded.connect(lambda msg: self.on_progress(msg, segmentation, "segmentation"))
        worker.returned.connect(lambda result: self.on_segmentation_loaded(result))
        worker.errored.connect(lambda e: self.on_error(str(e), segmentation, "segmentation"))
        worker.finished.connect(lambda: self.cleanup_worker(segmentation))

        # Start the worker
        worker.start()

        self.loading_workers[segmentation] = worker
        self.info_label.setText(f"Loading segmentation: {segmentation.name}...")

    def on_progress(self, message: str, data_object: Any, data_type: str) -> None:
        """
        Handle progress updates from workers.
        """
        pass
        self.info_label.setText(f"{message}")

    def on_tomogram_loaded(self, result: Dict[str, Any]) -> None:
        """
        Handle successful tomogram loading.
        """
        tomogram = result["tomogram"]
        loaded_data = result["data"]
        voxel_size = result["voxel_size"]
        name = result["name"]
        resolution_level = result["resolution_level"]

        # Remove loading indicator (only for tree items)
        if tomogram in self.loading_items:
            item = self.loading_items[tomogram]
            if item is not None:
                self.remove_loading_indicator(item)

        # Remove global loading indicator
        operation_id = f"load_tomogram_{tomogram.tomo_type}_{id(tomogram)}"
        self._remove_operation(operation_id)

        # Add pre-loaded image to the viewer (should be fast!)
        try:
            layer = self.viewer.add_image(
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

            self.info_label.setText(f"Loaded Tomogram: {tomogram.tomo_type} (Resolution Level {resolution_level})")
        except Exception as e:
            self.logger.exception(f"Error adding image to viewer: {str(e)}")
            self.info_label.setText(f"Error displaying tomogram: {str(e)}")

    def on_segmentation_loaded(self, result: Dict[str, Any]) -> None:
        """
        Handle successful segmentation loading.
        """
        segmentation = result["segmentation"]
        loaded_data = result["data"]
        voxel_size = result["voxel_size"]
        name = result["name"]
        resolution_level = result["resolution_level"]

        # Remove loading indicator
        if segmentation in self.loading_items:
            item = self.loading_items[segmentation]
            self.remove_loading_indicator(item)

        # Remove global loading indicator
        operation_id = f"load_segmentation_{segmentation.name}_{id(segmentation)}"
        self._remove_operation(operation_id)

        # Add pre-loaded segmentation to the viewer (should be fast!)
        try:
            # Create a color map based on copick colors
            if segmentation.is_multilabel:
                # For multilabel segmentations, use full colormap
                colormap = self.get_copick_colormap()
                painting_labels = [obj.label for obj in self.root.pickable_objects]
                class_labels_mapping = {obj.label: obj.name for obj in self.root.pickable_objects}
            else:
                # For single label segmentations, find the matching pickable object
                matching_obj = None
                for obj in self.root.pickable_objects:
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

            painting_layer = self.viewer.add_labels(loaded_data, name=name, scale=voxel_size)
            painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
            painting_layer.painting_labels = painting_labels
            self.class_labels_mapping = class_labels_mapping

            # Store copick metadata in the layer
            painting_layer.metadata["copick_run"] = segmentation.run
            painting_layer.metadata["copick_segmentation"] = segmentation
            painting_layer.metadata["copick_voxel_size"] = segmentation.voxel_size
            painting_layer.metadata["copick_resolution_level"] = resolution_level
            painting_layer.metadata["copick_source_object_name"] = segmentation.name

            self.info_label.setText(
                f"Loaded Segmentation: {segmentation.name} (Resolution Level {resolution_level})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding segmentation to viewer: {str(e)}")
            self.info_label.setText(f"Error displaying segmentation: {str(e)}")

    def on_error(self, error_msg: str, data_object: Any, data_type: str) -> None:
        """
        Handle errors for loading and expansion operations.
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
            self._remove_operation(operation_id)
        elif data_type == "segmentation":
            operation_id = f"load_segmentation_{data_object.name}_{id(data_object)}"
            self._remove_operation(operation_id)
        elif data_type == "run":
            operation_id = f"expand_run_{data_object.name}"
            self._remove_operation(operation_id)
        elif data_type == "voxel_spacing":
            operation_id = f"expand_voxel_spacing_{data_object.voxel_size}"
            self._remove_operation(operation_id)

        # Remove loading indicator and clean up workers properly
        if data_object in self.loading_items:
            item = self.loading_items[data_object]
            if item is not None:
                self.remove_loading_indicator(item)
            # Clean up loading worker
            self.cleanup_worker(data_object)
        elif data_object in self.expansion_items:
            item = self.expansion_items[data_object]
            self.remove_loading_indicator(item)
            # Clean up expansion worker
            self.cleanup_expansion_worker(data_object)

        self.info_label.setText(f"Error: {error_msg}")

    def cleanup_worker(self, data_object: Any) -> None:
        """
        Clean up loading worker and associated data.
        """
        if data_object in self.loading_workers:
            del self.loading_workers[data_object]

        if data_object in self.loading_items:
            del self.loading_items[data_object]

    def cleanup_expansion_worker(self, data_object: Any) -> None:
        """
        Clean up expansion worker and associated data.
        """
        if data_object in self.expansion_workers:
            del self.expansion_workers[data_object]

        if data_object in self.expansion_items:
            del self.expansion_items[data_object]

    def get_copick_colormap(
        self,
        pickable_objects: Optional[List[copick.models.PickableObject]] = None,
    ) -> Dict[Union[int, None], np.ndarray]:
        if not pickable_objects:
            pickable_objects = self.root.pickable_objects

        colormap = {obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])

        return colormap

    def load_picks(self, pick_set: copick.models.CopickPicks, parent_run: Optional[copick.models.CopickRun]) -> None:
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [(p.location.z, p.location.y, p.location.x) for p in pick_set.points]

                    # Find the matching pickable object to get the correct color
                    pickable_object = None
                    for obj in self.root.pickable_objects:
                        if obj.name == pick_set.pickable_object_name:
                            pickable_object = obj
                            break

                    if pickable_object:  # noqa: SIM108
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
                    )  # Create an array with the correct shape
                    # TODO hardcoded default point size
                    point_size = pickable_object.radius if pickable_object.radius else 50  # Default point size
                    points_layer = self.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.pickable_object_name}",
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

                    self.info_label.setText(f"Loaded Picks: {pick_set.pickable_object_name}")
                else:
                    self.info_label.setText(f"No points found for Picks: {pick_set.pickable_object_name}")
            else:
                self.info_label.setText(f"No pick set found for Picks: {pick_set.pickable_object_name}")
        else:
            self.info_label.setText("No parent run found")

    def get_color(self, pick: copick.models.CopickPicks) -> str:
        for obj in self.root.pickable_objects:
            if obj.name == pick.object_name:
                return obj.color
        return "white"

    def get_run(self, name: str) -> Optional[copick.models.CopickRun]:
        return self.root.get_run(name)

    def open_context_menu(self, position: Any) -> None:
        item = self.tree_view.itemAt(position)
        if not item:
            return

        if self.is_segmentations_or_picks_item(item):
            context_menu = QMenu(self.tree_view)
            if item.text(0) == "Segmentations":
                run_name = item.parent().parent().text(0)
                run = self.root.get_run(run_name)
                self.show_segmentation_widget(run)
            elif item.text(0) == "Picks":
                run_name = item.parent().text(0)
                run = self.root.get_run(run_name)
                self.show_picks_widget(run)
            context_menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def is_segmentations_or_picks_item(self, item: QTreeWidgetItem) -> bool:
        if item.text(0) == "Segmentations" or item.text(0) == "Picks":  # noqa: SIM103
            return True
        return False

    def show_segmentation_widget(self, run: copick.models.CopickRun) -> None:
        widget = QWidget()
        widget.setWindowTitle("Create New Segmentation")

        layout = QFormLayout(widget)
        name_input = QLineEdit(widget)
        name_input.setText("segmentation")
        layout.addRow("Name:", name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCopick")
        layout.addRow("User ID:", user_input)

        voxel_size_input = QComboBox(widget)
        for voxel_spacing in run.voxel_spacings:
            voxel_size_input.addItem(str(voxel_spacing.voxel_size))
        layout.addRow("Voxel Size:", voxel_size_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_segmentation(
                widget,
                run,
                name_input.text(),
                session_input.value(),
                user_input.text(),
                float(voxel_size_input.currentText()),
            ),
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def show_picks_widget(self, run: copick.models.CopickRun) -> None:
        widget = QWidget()
        widget.setWindowTitle("Create New Picks")

        layout = QFormLayout(widget)
        object_name_input = QComboBox(widget)
        for obj in self.root.pickable_objects:
            object_name_input.addItem(obj.name)
        layout.addRow("Object Name:", object_name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCopick")
        layout.addRow("User ID:", user_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_picks(
                widget,
                run,
                object_name_input.currentText(),
                session_input.value(),
                user_input.text(),
            ),
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def create_segmentation(
        self,
        widget: QWidget,
        run: copick.models.CopickRun,
        name: str,
        session_id: int,
        user_id: str,
        voxel_size: float,
    ) -> None:
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=str(session_id),
            is_multilabel=True,
            user_id=user_id,
        )

        # Get tomogram shape from first available tomogram
        first_tomogram = run.voxel_spacings[0].tomograms[0]
        zarr_group = zarr.open(first_tomogram.zarr(), "r")
        # Get shape from the highest resolution level
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]  # noqa: SIM118
        scale_levels.sort(key=int)
        tomo_array = zarr_group[scale_levels[0]]
        shape = tomo_array.shape

        dtype = np.int32

        # Create an empty Zarr array for the segmentation
        zarr_file = zarr.open(seg.zarr(), mode="w")
        zarr_file.create_dataset(
            "data",
            shape=shape,
            dtype=dtype,
            chunks=(128, 128, 128),
            fill_value=0,
        )

        self.populate_tree()
        widget.close()

    def create_picks(
        self,
        widget: QWidget,
        run: copick.models.CopickRun,
        object_name: str,
        session_id: int,
        user_id: str,
    ) -> None:
        run.new_picks(
            object_name=object_name,
            session_id=str(session_id),
            user_id=user_id,
        )
        self.populate_tree()
        widget.close()

    def _update_gallery(self) -> None:
        """Update the gallery widget with current copick root."""
        if GALLERY_AVAILABLE and hasattr(self, "gallery_widget"):
            self.gallery_widget.set_copick_root(self.root)

    def switch_to_tree_view(self) -> None:
        """Switch to tree view tab."""
        self.tab_widget.setCurrentIndex(0)

    def switch_to_gallery_view(self) -> None:
        """Switch to gallery view tab."""
        self.tab_widget.setCurrentIndex(1)

    def switch_to_info_view(self) -> None:
        """Switch to info view tab."""
        # Find the info view tab index
        for i in range(self.tab_widget.count()):
            tab_text = self.tab_widget.tabText(i)
            if "Info View" in tab_text:
                self.tab_widget.setCurrentIndex(i)
                return

    def _on_info_requested(self, run: copick.models.CopickRun) -> None:
        """Handle info request from gallery widget."""
        try:
            # Switch to info view immediately for snappy response
            self.switch_to_info_view()

            # Process events to make the tab switch visible immediately
            from qtpy.QtWidgets import QApplication

            QApplication.processEvents()

            # Now load the data
            if INFO_AVAILABLE and hasattr(self, "info_widget"):
                try:
                    self.info_widget.set_run(run)
                except Exception:
                    import traceback

                    traceback.print_exc()
        except Exception:
            import traceback

            traceback.print_exc()

    def open_save_segmentation_dialog(self) -> None:
        """Open dialog to save a segmentation layer to copick."""
        if not self.root:
            self.info_label.setText("No configuration loaded. Please load a config first.")
            return

        # Get available segmentation layers (Labels layers)
        segmentation_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Labels) and layer.data.ndim == 3
        ]

        if not segmentation_layers:
            self.info_label.setText("No segmentation layers found in the viewer.")
            return

        # Get runs from currently open image layers
        available_runs = get_runs_from_open_layers(self.viewer)

        if not available_runs:
            self.info_label.setText("No runs found from currently open image layers.")
            return

        # Check if there's a currently selected segmentation layer to preset dialog values
        selected_layer = None
        selected_object_name = None
        should_enable_overwrite = False

        # Look for the currently active layer or the first segmentation layer
        if self.viewer.layers.selection.active in segmentation_layers:
            selected_layer = self.viewer.layers.selection.active
        elif segmentation_layers:
            selected_layer = segmentation_layers[0]

        # Check if this layer was loaded from an existing segmentation
        if selected_layer and "copick_source_object_name" in selected_layer.metadata:
            selected_object_name = selected_layer.metadata["copick_source_object_name"]
            should_enable_overwrite = True

        dialog = SaveSegmentationDialog(
            self,
            segmentation_layers,
            available_runs,
            self.root.pickable_objects,
            preset_layer=selected_layer,
            preset_object_name=selected_object_name,
            preset_overwrite=should_enable_overwrite,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                self.save_segmentation_async(result)
            except Exception as e:
                self.info_label.setText(f"Error saving segmentation: {str(e)}")
                self.logger.exception(f"Error saving segmentation: {str(e)}")

    def save_segmentation_async(self, save_params: Dict[str, Any]) -> None:
        """Save segmentation asynchronously with loading indicator."""
        # Create a unique operation ID for this save operation
        operation_id = f"save_segmentation_{save_params['object_name']}_{id(save_params)}"

        # Add global loading indicator
        self._add_operation(operation_id, f"Saving segmentation '{save_params['object_name']}'...")

        # Create the save worker
        worker = save_segmentation_worker(save_params)

        # Connect signals
        worker.yielded.connect(lambda msg: self.on_progress(msg, save_params, "save_segmentation"))
        worker.returned.connect(lambda result: self.on_segmentation_saved(result, operation_id))
        worker.errored.connect(lambda e: self.on_save_error(str(e), save_params, operation_id))
        worker.finished.connect(lambda: self.cleanup_save_worker(operation_id))

        # Start the worker
        worker.start()

        # Store the worker to track it using operation_id as key
        self.loading_workers[operation_id] = worker
        self.info_label.setText(f"Saving segmentation '{save_params['object_name']}'...")

    def on_segmentation_saved(self, result: Dict[str, Any], operation_id: str) -> None:
        """Handle successful segmentation save."""
        if result.get("success", False):
            self.info_label.setText(result.get("message", "Segmentation saved successfully"))
            # Instead of rebuilding entire tree, just refresh the relevant voxel spacing
            # to show the new segmentation while preserving expansion state
            self.refresh_tree_after_save(result)
        else:
            self.info_label.setText(f"Failed to save segmentation: {result.get('message', 'Unknown error')}")

        # Remove global loading indicator
        self._remove_operation(operation_id)

    def on_save_error(self, error_message: str, save_params: Dict[str, Any], operation_id: str) -> None:
        """Handle segmentation save error."""
        self.info_label.setText(f"Error saving segmentation: {error_message}")
        self.logger.exception(f"Error saving segmentation: {error_message}")

        # Remove global loading indicator
        self._remove_operation(operation_id)

    def cleanup_save_worker(self, operation_id: str) -> None:
        """Clean up save worker."""
        if operation_id in self.loading_workers:
            del self.loading_workers[operation_id]

    def open_save_picks_dialog(self) -> None:
        """Open dialog to save a points layer to copick."""
        if not self.root:
            self.info_label.setText("No configuration loaded. Please load a config first.")
            return

        # Get available points layers
        points_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Points) and layer.data.shape[1] == 3
        ]

        if not points_layers:
            self.info_label.setText("No points layers found in the viewer.")
            return

        # Get runs from currently open image layers
        available_runs = get_runs_from_open_layers(self.viewer)

        if not available_runs:
            self.info_label.setText("No runs found from currently open image layers.")
            return

        # Check if there's a currently selected points layer to preset dialog values
        selected_layer = None
        selected_object_name = None
        selected_run = None
        should_enable_overwrite = False

        # Look for the currently active layer or the first points layer
        if self.viewer.layers.selection.active in points_layers:
            selected_layer = self.viewer.layers.selection.active
        elif points_layers:
            selected_layer = points_layers[0]

        # Check if this layer was loaded from existing picks
        if selected_layer and "copick_source_object_name" in selected_layer.metadata:
            selected_object_name = selected_layer.metadata["copick_source_object_name"]
            selected_run = selected_layer.metadata.get("copick_run")
            should_enable_overwrite = True

        dialog = SaveLayerDialog(
            parent=self,
            layers=points_layers,
            available_runs=available_runs,
            pickable_objects=self.root.pickable_objects,
            layer_type="picks",
            preset_layer=selected_layer,
            preset_object_name=selected_object_name,
            preset_overwrite=should_enable_overwrite,
            preset_run=selected_run,
        )
        if dialog.exec_() == QDialog.Accepted:
            try:
                result = dialog.get_values()
                success = save_picks_to_copick(result, self.info_label.setText)
                if success:
                    # Refresh tree while preserving expansion state
                    self.populate_tree(preserve_expansion=True)
            except Exception as e:
                self.info_label.setText(f"Error saving picks: {str(e)}")
                self.logger.exception(f"Error saving picks: {str(e)}")
