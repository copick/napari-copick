import logging

import copick
import napari
import numpy as np
import zarr
from napari.utils import DirectLabelColormap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .async_loaders import (
    expand_run_worker,
    expand_voxel_spacing_worker,
    load_segmentation_worker,
    load_tomogram_worker,
)

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
    from copick_shared_ui.ui import EditObjectTypesDialog
except ImportError:
    # Fallback if shared component is not available
    EditObjectTypesDialog = None

# Import the gallery widget
try:
    from .gallery_widget import NapariCopickGalleryWidget

    GALLERY_AVAILABLE = True
except ImportError:
    GALLERY_AVAILABLE = False

# Import the info widget
try:
    from .info_widget import NapariCopickInfoWidget

    INFO_AVAILABLE = True
except ImportError:
    INFO_AVAILABLE = False


class DatasetIdDialog(QDialog):
    def __init__(self, parent=None):
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

    def get_values(self):
        dataset_ids_text = self.dataset_ids_input.text()
        dataset_ids = [int(id.strip()) for id in dataset_ids_text.split(",") if id.strip()]
        overlay_root = self.overlay_root_input.text()
        return dataset_ids, overlay_root


class CopickPlugin(QWidget):
    def __init__(self, viewer=None, config_path=None, dataset_ids=None, overlay_root="/tmp/overlay_root"):
        super().__init__()

        # Setup logging
        self.logger = logging.getLogger("CopickPlugin")

        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()

        self.root = None
        self.selected_run = None
        self.current_layer = None
        self.session_id = "17"
        self.loading_workers = {}  # Track active loading workers
        self.loading_items = {}  # Track tree items being loaded
        self.expansion_workers = {}  # Track active expansion workers
        self.expansion_items = {}  # Track tree items being expanded
        self.setup_ui()

        if config_path:
            self.load_config(config_path=config_path)
        elif dataset_ids:
            self.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def setup_ui(self):
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
        self.tree_view = QTreeWidget()
        self.tree_view.setHeaderLabel("Copick Project")
        self.tree_view.itemExpanded.connect(self.handle_item_expand)
        self.tree_view.itemClicked.connect(self.handle_item_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_context_menu)
        tree_layout.addWidget(self.tree_view)

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

        # Info label
        self.info_label = QLabel("Select a pick to get started")
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
        self.active_operations = set()  # Set of operation identifiers

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

    def closeEvent(self, event):
        """Clean up workers when widget is closed."""
        self.cleanup_workers()
        super().closeEvent(event)

    def cleanup_workers(self):
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

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "JSON Files (*.json)")
        if path:
            self.load_config(config_path=path)

    def open_dataset_dialog(self):
        dialog = DatasetIdDialog(self)
        if dialog.exec_():
            dataset_ids, overlay_root = dialog.get_values()
            if dataset_ids:
                self.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def open_edit_objects_dialog(self):
        """Open the EditObjectTypesDialog to manage object types"""
        if not self.root or not self.root.config:
            self.info_label.setText("No configuration loaded. Please load a config first.")
            return

        if EditObjectTypesDialog is None:
            self.info_label.setText("EditObjectTypesDialog is not available. Shared component may not be installed.")
            return

        try:
            dialog = EditObjectTypesDialog(self, self.root.config.pickable_objects)
            if dialog.exec_() == QDialog.Accepted:
                # Get the updated objects from the dialog
                updated_objects = dialog.get_objects()

                # Update the configuration
                self.root.config.pickable_objects = updated_objects

                # Update any UI elements that depend on the object types
                self.populate_tree()  # Refresh the tree view

                # Update any loaded segmentation layers with new colormap
                for layer in self.viewer.layers:
                    if hasattr(layer, "colormap") and "Segmentation:" in layer.name:
                        layer.colormap = DirectLabelColormap(color_dict=self.get_copick_colormap())
                        layer.painting_labels = [obj.label for obj in self.root.config.pickable_objects]

                self.info_label.setText(f"Updated {len(updated_objects)} object types in configuration")
        except Exception as e:
            self.info_label.setText(f"Error opening EditObjectTypesDialog: {str(e)}")

    def load_config(self, config_path=None):
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

            self.populate_tree()
            self._update_gallery()
            self.edit_objects_button.setEnabled(True)  # Enable the button when config is loaded
            self.info_label.setText(f"Loaded config from {config_path}")

    def load_from_dataset_ids(self, dataset_ids=None, overlay_root="/tmp/overlay_root"):
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

            self.populate_tree()
            self._update_gallery()
            self.edit_objects_button.setEnabled(True)  # Enable the button when config is loaded
            self.info_label.setText(f"Loaded project from dataset IDs: {', '.join(map(str, dataset_ids))}")

    def populate_tree(self):
        self.tree_view.clear()
        for run in self.root.runs:
            run_item = QTreeWidgetItem(self.tree_view, [run.meta.name])
            run_item.setData(0, Qt.UserRole, run)
            run_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

    def handle_item_expand(self, item):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.expand_run_async(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing_async(item, data)

    def expand_run_async(self, item, run):
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
        operation_id = f"expand_run_{run.meta.name}"
        self._add_operation(operation_id, f"Expanding run: {run.meta.name}...")

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
        self.info_label.setText(f"Expanding run: {run.meta.name}...")

    def on_run_expanded(self, result):
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
                spacing_item = QTreeWidgetItem(item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"])
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

            # Add picks nested by user_id, session_id, and pickable_object_name
            picks_item = QTreeWidgetItem(item, ["Picks"])
            for user_id, sessions in picks_data.items():
                user_item = QTreeWidgetItem(picks_item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(user_item, [f"Session: {session_id}"])
                    for pick in picks:
                        pick_child = QTreeWidgetItem(session_item, [pick.meta.pickable_object_name])
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

            self.info_label.setText(f"Expanded run: {run.meta.name}")

        # Remove global loading indicator
        operation_id = f"expand_run_{run.meta.name}"
        self._remove_operation(operation_id)

    def expand_voxel_spacing_async(self, item, voxel_spacing):
        """
        Expand a voxel spacing asynchronously with loading indicator.
        """
        # Skip if already expanded or currently expanding
        if item.childCount() > 0 or voxel_spacing in self.expansion_workers:
            return

        # Add loading indicator
        self.add_loading_indicator(item)
        self.expansion_items[voxel_spacing] = item

        # Add global loading indicator
        operation_id = f"expand_voxel_spacing_{voxel_spacing.meta.voxel_size}"
        self._add_operation(operation_id, f"Expanding voxel spacing: {voxel_spacing.meta.voxel_size}...")

        # Create worker
        worker = expand_voxel_spacing_worker(voxel_spacing)

        # Connect signals
        worker.yielded.connect(lambda msg: self.on_progress(msg, voxel_spacing, "voxel_spacing"))
        worker.returned.connect(lambda result: self.on_voxel_spacing_expanded(result))
        worker.errored.connect(lambda e: self.on_error(str(e), voxel_spacing, "voxel_spacing"))
        worker.finished.connect(lambda: self.cleanup_expansion_worker(voxel_spacing))

        # Start the worker
        worker.start()
        self.expansion_workers[voxel_spacing] = worker
        self.info_label.setText(f"Expanding voxel spacing: {voxel_spacing.meta.voxel_size}...")

    def on_voxel_spacing_expanded(self, result):
        """
        Handle successful voxel spacing expansion.
        """
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
                tomo_child = QTreeWidgetItem(tomogram_item, [tomogram.meta.tomo_type])
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            # Add segmentations
            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(segmentation_item, [segmentation.meta.name])
                seg_child.setData(0, Qt.UserRole, segmentation)
            item.addChild(segmentation_item)

            self.info_label.setText(f"Expanded voxel spacing: {voxel_spacing.meta.voxel_size}")

        # Remove global loading indicator
        operation_id = f"expand_voxel_spacing_{voxel_spacing.meta.voxel_size}"
        self._remove_operation(operation_id)

    def handle_item_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.info_label.setText(f"Run: {data.meta.name}")
            self.selected_run = data
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.info_label.setText(f"Voxel Spacing: {data.meta.voxel_size}")
            self.lazy_load_voxel_spacing(item, data)
        elif isinstance(data, copick.models.CopickTomogram):
            self.load_tomogram_async(data, item)
        elif isinstance(data, copick.models.CopickSegmentation):
            self.load_segmentation_async(data, item)
        elif isinstance(data, copick.models.CopickPicks):
            parent_run = self.get_parent_run(item)
            self.load_picks(data, parent_run)

    def get_parent_run(self, item):
        while item:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, copick.models.CopickRun):
                return data
            item = item.parent()
        return None

    def lazy_load_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            self.expand_voxel_spacing_async(item, voxel_spacing)

    def load_tomogram_async(self, tomogram, item):
        """
        Load a tomogram asynchronously with loading indicator using napari's threading system.
        """

        # Check if already loading
        if tomogram in self.loading_workers:
            self.logger.warning(f"Tomogram {tomogram.meta.tomo_type} already loading, skipping")
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
        operation_id = f"load_tomogram_{tomogram.meta.tomo_type}_{id(tomogram)}"
        self._add_operation(operation_id, f"Loading tomogram: {tomogram.meta.tomo_type}...")

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
        self.info_label.setText(f"Loading tomogram: {tomogram.meta.tomo_type}...")

    def add_loading_indicator(self, item):
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

    def remove_loading_indicator(self, item):
        """
        Remove the loading indicator from the tree item and restore original text.
        """
        self.tree_view.setItemWidget(item, 0, None)

        # Restore original text if stored
        original_text = item.data(0, Qt.UserRole + 1)
        if original_text:
            item.setText(0, original_text)
            item.setData(0, Qt.UserRole + 1, None)  # Clear stored text

    def load_segmentation_async(self, segmentation, item):
        """
        Load a segmentation asynchronously with loading indicator using napari's threading system.
        """

        # Check if already loading
        if segmentation in self.loading_workers:
            self.logger.warning(f"Segmentation {segmentation.meta.name} already loading, skipping")
            return

        # Add loading indicator
        self.add_loading_indicator(item)
        self.loading_items[segmentation] = item

        # Add global loading indicator
        operation_id = f"load_segmentation_{segmentation.meta.name}_{id(segmentation)}"
        self._add_operation(operation_id, f"Loading segmentation: {segmentation.meta.name}...")

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
        self.info_label.setText(f"Loading segmentation: {segmentation.meta.name}...")

    def on_progress(self, message, data_object, data_type):
        """
        Handle progress updates from workers.
        """
        pass
        self.info_label.setText(f"{message}")

    def on_tomogram_loaded(self, result):
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
        operation_id = f"load_tomogram_{tomogram.meta.tomo_type}_{id(tomogram)}"
        self._remove_operation(operation_id)

        # Add pre-loaded image to the viewer (should be fast!)
        try:
            layer = self.viewer.add_image(
                loaded_data,
                scale=voxel_size,
                name=name,
            )
            layer.reset_contrast_limits()
            self.info_label.setText(f"Loaded Tomogram: {tomogram.meta.tomo_type} (Resolution Level {resolution_level})")
        except Exception as e:
            self.logger.exception(f"Error adding image to viewer: {str(e)}")
            self.info_label.setText(f"Error displaying tomogram: {str(e)}")

    def on_segmentation_loaded(self, result):
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
        operation_id = f"load_segmentation_{segmentation.meta.name}_{id(segmentation)}"
        self._remove_operation(operation_id)

        # Add pre-loaded segmentation to the viewer (should be fast!)
        try:
            # Create a color map based on copick colors
            colormap = self.get_copick_colormap()
            painting_layer = self.viewer.add_labels(loaded_data, name=name, scale=voxel_size)
            painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
            painting_layer.painting_labels = [obj.label for obj in self.root.config.pickable_objects]
            self.class_labels_mapping = {obj.label: obj.name for obj in self.root.config.pickable_objects}

            self.info_label.setText(
                f"Loaded Segmentation: {segmentation.meta.name} (Resolution Level {resolution_level})",
            )
        except Exception as e:
            self.logger.exception(f"Error adding segmentation to viewer: {str(e)}")
            self.info_label.setText(f"Error displaying segmentation: {str(e)}")

    def on_error(self, error_msg, data_object, data_type):
        """
        Handle errors for loading and expansion operations.
        """
        if data_type == "tomogram":
            self.logger.exception(f"Tomogram loading error for {data_object.meta.tomo_type}: {error_msg}")
        elif data_type == "segmentation":
            self.logger.exception(f"Segmentation loading error for {data_object.meta.name}: {error_msg}")
        elif data_type == "run":
            self.logger.exception(f"Run expansion error for {data_object.meta.name}: {error_msg}")
        elif data_type == "voxel_spacing":
            self.logger.exception(f"Voxel spacing expansion error for {data_object.meta.voxel_size}: {error_msg}")

        # Remove global loading indicator for errors
        if data_type == "tomogram":
            operation_id = f"load_tomogram_{data_object.meta.tomo_type}_{id(data_object)}"
            self._remove_operation(operation_id)
        elif data_type == "segmentation":
            operation_id = f"load_segmentation_{data_object.meta.name}_{id(data_object)}"
            self._remove_operation(operation_id)
        elif data_type == "run":
            operation_id = f"expand_run_{data_object.meta.name}"
            self._remove_operation(operation_id)
        elif data_type == "voxel_spacing":
            operation_id = f"expand_voxel_spacing_{data_object.meta.voxel_size}"
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

    def cleanup_worker(self, data_object):
        """
        Clean up loading worker and associated data.
        """
        if data_object in self.loading_workers:
            del self.loading_workers[data_object]

        if data_object in self.loading_items:
            del self.loading_items[data_object]

    def cleanup_expansion_worker(self, data_object):
        """
        Clean up expansion worker and associated data.
        """
        if data_object in self.expansion_workers:
            del self.expansion_workers[data_object]

        if data_object in self.expansion_items:
            del self.expansion_items[data_object]

    def get_copick_colormap(self, pickable_objects=None):
        if not pickable_objects:
            pickable_objects = self.root.config.pickable_objects
        colormap = {obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def load_picks(self, pick_set, parent_run):
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [(p.location.z, p.location.y, p.location.x) for p in pick_set.points]
                    color = (
                        pick_set.color if pick_set.color else (255, 255, 255, 255)
                    )  # Default to white if color is not set
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
                    pickable_object = [
                        obj for obj in self.root.pickable_objects if obj.name == pick_set.pickable_object_name
                    ][0]
                    # TODO hardcoded default point size
                    point_size = pickable_object.radius if pickable_object.radius else 50
                    self.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.meta.pickable_object_name}",
                        size=point_size,
                        face_color=colors,
                        out_of_slice_display=True,
                    )
                    self.info_label.setText(f"Loaded Picks: {pick_set.meta.pickable_object_name}")
                else:
                    self.info_label.setText(f"No points found for Picks: {pick_set.meta.pickable_object_name}")
            else:
                self.info_label.setText(f"No pick set found for Picks: {pick_set.meta.pickable_object_name}")
        else:
            self.info_label.setText("No parent run found")

    def get_color(self, pick):
        for obj in self.root.pickable_objects:
            if obj.name == pick.meta.object_name:
                return obj.color
        return "white"

    def get_run(self, name):
        return self.root.get_run(name)

    def open_context_menu(self, position):
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

    def is_segmentations_or_picks_item(self, item):
        if item.text(0) == "Segmentations" or item.text(0) == "Picks":  # noqa: SIM103
            return True
        return False

    def show_segmentation_widget(self, run):
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
            voxel_size_input.addItem(str(voxel_spacing.meta.voxel_size))
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

    def show_picks_widget(self, run):
        widget = QWidget()
        widget.setWindowTitle("Create New Picks")

        layout = QFormLayout(widget)
        object_name_input = QComboBox(widget)
        for obj in self.root.config.pickable_objects:
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

    def create_segmentation(self, widget, run, name, session_id, user_id, voxel_size):
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

    def create_picks(self, widget, run, object_name, session_id, user_id):
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

    def _on_info_requested(self, run) -> None:
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
