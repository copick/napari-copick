import logging
from typing import Any, Dict, List, Optional, Set, Union

import copick
import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_copick.config_manager import ConfigManager
from napari_copick.data_loader import DataLoader
from napari_copick.save_manager import SaveManager
from napari_copick.tree_expansion_manager import TreeExpansionManager
from napari_copick.tree_widget import CopickTreeWidget

# Import the gallery widget
try:
    from napari_copick.gallery_widget import NapariCopickGalleryWidget

    GALLERY_AVAILABLE = True
except ImportError:
    GALLERY_AVAILABLE = False
    NapariCopickGalleryWidget = None

# Import the info widget
try:
    from napari_copick.info_widget import NapariCopickInfoWidget

    INFO_AVAILABLE = True
except ImportError:
    INFO_AVAILABLE = False
    NapariCopickInfoWidget = None


class CopickPlugin(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
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
        self.config_path: Optional[str] = None  # Store config file path for saving
        self.selected_run: Optional[copick.models.CopickRun] = None
        self.current_layer: Optional[Any] = None
        self.session_id: str = "17"
        self.loading_workers: Dict[Any, Any] = {}  # Track active loading workers
        self.loading_items: Dict[Any, Optional[QTreeWidgetItem]] = {}  # Track tree items being loaded
        self.expansion_workers: Dict[Any, Any] = {}  # Track active expansion workers
        self.expansion_items: Dict[Any, QTreeWidgetItem] = {}  # Track tree items being expanded

        # Initialize manager classes
        self.config_manager = ConfigManager(self)
        self.tree_expansion_manager = TreeExpansionManager(self)
        self.data_loader = DataLoader(self)
        self.save_manager = SaveManager(self)

        self.setup_ui()

        if config_path:
            self.config_manager.load_config(config_path=config_path)
        elif dataset_ids:
            self.config_manager.load_from_dataset_ids(dataset_ids=dataset_ids, overlay_root=overlay_root)

    def setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Config loading options
        load_options_layout = QHBoxLayout()

        # Config file button
        self.load_config_button = QPushButton("Load Config File")
        self.load_config_button.clicked.connect(self.config_manager.open_file_dialog)
        load_options_layout.addWidget(self.load_config_button)

        # Dataset IDs button
        self.load_dataset_button = QPushButton("Load from Dataset IDs")
        self.load_dataset_button.clicked.connect(self.config_manager.open_dataset_dialog)
        load_options_layout.addWidget(self.load_dataset_button)

        layout.addLayout(load_options_layout)

        # Edit Object Types button
        self.edit_objects_button = QPushButton("✏️ Edit Object Types")
        self.edit_objects_button.clicked.connect(self.config_manager.open_edit_objects_dialog)
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
        self.save_segmentation_button.clicked.connect(self.save_manager.open_save_segmentation_dialog)
        self.save_segmentation_button.setEnabled(False)  # Disabled until config is loaded
        self.save_segmentation_button.setToolTip("Save a segmentation layer to copick")
        save_buttons_layout.addWidget(self.save_segmentation_button)

        # Save picks button
        self.save_picks_button = QPushButton("📍 Save Picks")
        self.save_picks_button.clicked.connect(self.save_manager.open_save_picks_dialog)
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

    def get_copick_colormap(
        self,
        pickable_objects: Optional[List[copick.models.PickableObject]] = None,
    ) -> Dict[Union[int, None], np.ndarray]:
        if not pickable_objects:
            pickable_objects = self.root.pickable_objects

        colormap = {obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])

        return colormap

    def get_run(self, name: str) -> Optional[copick.models.CopickRun]:
        return self.root.get_run(name)

    def open_context_menu(self, position: Any) -> None:
        # Context menu functionality has been simplified - most creation is now handled through save dialogs
        pass

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
