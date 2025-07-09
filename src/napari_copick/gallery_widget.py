"""napari-specific gallery widget implementation using shared copick-shared-ui."""

from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

try:
    # Import directly from module to avoid __init__.py issues
    import copick_shared_ui.platform.napari_integration as napari_integration_module

    NapariGalleryIntegration = napari_integration_module.NapariGalleryIntegration
    SHARED_UI_AVAILABLE = True
except ImportError as e:
    print(f"❌ DEBUG: Gallery import failed - {e}")
    import traceback

    traceback.print_exc()
    SHARED_UI_AVAILABLE = False

if TYPE_CHECKING:
    import napari
    from copick.models import CopickRun, CopickTomogram


class NapariCopickGalleryWidget(QWidget):
    """napari-specific implementation of the copick gallery widget."""

    # Define signals
    info_requested = Signal(object)  # Emits CopickRun when info is requested

    def __init__(self, viewer: "napari.Viewer", parent: Optional[QWidget] = None) -> None:
        self.original_parent = parent
        super().__init__(parent)
        self.viewer = viewer
        self.copick_root = None

        if not SHARED_UI_AVAILABLE:
            print("⚠️ DEBUG: Using fallback gallery UI - shared UI not available")
            self._setup_fallback_ui()
            return

        # Initialize the shared UI integration
        self.gallery_integration = NapariGalleryIntegration(viewer)
        self.gallery_widget = self.gallery_integration.create_gallery_widget(self)

        # Setup UI
        self._setup_ui()

        # Connect signals
        self.gallery_widget.run_selected.connect(self._on_run_selected)
        self.gallery_widget.info_requested.connect(self._on_info_requested)

    def _setup_ui(self) -> None:
        """Setup the widget UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with controls
        header_layout = QHBoxLayout()

        # Back to tree button
        self.back_button = QPushButton("← Back to Tree View")
        self.back_button.clicked.connect(self._on_back_clicked)
        header_layout.addWidget(self.back_button)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Add the gallery widget
        layout.addWidget(self.gallery_widget)

    def _setup_fallback_ui(self) -> None:
        """Setup fallback UI when shared components are not available."""
        layout = QVBoxLayout(self)

        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel

        label = QLabel(
            "Gallery widget not available\n\nThe copick-shared-ui package is required for the gallery feature.",
        )
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
        layout.addWidget(label)

    def set_copick_root(self, copick_root: Any) -> None:
        """Set the copick root for the gallery."""
        self.copick_root = copick_root

        if SHARED_UI_AVAILABLE and hasattr(self, "gallery_integration"):
            self.gallery_integration.set_copick_root(copick_root)
            self.gallery_widget.set_copick_root(copick_root)

    def apply_search_filter(self, filter_text: str) -> None:
        """Apply search filter to the gallery."""
        if SHARED_UI_AVAILABLE and hasattr(self, "gallery_widget"):
            self.gallery_widget.apply_search_filter(filter_text)

    @Slot(object)
    def _on_run_selected(self, run: "CopickRun") -> None:
        """Handle run selection from gallery."""
        try:
            # Use the shared system's best tomogram selection and caching
            # This will trigger thumbnail generation and save best tomogram info
            if SHARED_UI_AVAILABLE and hasattr(self, "gallery_integration"):
                # Let the shared gallery system handle the selection - this will trigger
                # the thumbnail worker which properly caches the best tomogram info

                # Use the shared system to find the best tomogram
                from copick_shared_ui.workers.base import AbstractThumbnailWorker

                class BestTomogramFinder(AbstractThumbnailWorker):
                    def __init__(self, run: Any, callback: Any) -> None:
                        self.run = run
                        self.callback = callback
                        super().__init__(run, run.name, callback, force_regenerate=False)

                    def start(self) -> None:
                        try:
                            best_tomogram = self._select_best_tomogram(self.run)
                            if best_tomogram:
                                self.callback(None, best_tomogram, None)
                            else:
                                self.callback(None, None, "No suitable tomogram found")
                        except Exception as e:
                            self.callback(None, None, str(e))

                    def cancel(self) -> None:
                        pass

                    def _array_to_pixmap(self, array: Any) -> Any:
                        return None  # Not needed for tomogram finding

                def on_best_tomogram_found(thumbnail_id: Any, best_tomogram: Any, error: Any) -> None:
                    if best_tomogram and not error:
                        self._load_tomogram_from_gallery(best_tomogram)
                    else:
                        print(f"Could not find best tomogram for run {run.name}: {error}")

                finder = BestTomogramFinder(run, on_best_tomogram_found)
                finder.start()

        except Exception as e:
            print(f"Error loading tomogram from gallery: {e}")

    def _load_tomogram_from_gallery(self, tomogram: "CopickTomogram") -> None:
        """Load tomogram from gallery using the data loader manager."""
        try:
            # Find the parent CopickPlugin widget to access the data loader
            parent_widget = self.parent()
            while parent_widget and not hasattr(parent_widget, "data_loader"):
                parent_widget = parent_widget.parent()

            if parent_widget and hasattr(parent_widget, "data_loader"):
                # Use the data loader manager to load the tomogram
                # Pass None for the tree item since this is from the gallery
                parent_widget.data_loader.load_tomogram_async(tomogram, None)
            else:
                print("Could not find parent widget with data_loader")

        except Exception as e:
            print(f"Error loading tomogram from gallery: {e}")

    @Slot(object)
    def _on_info_requested(self, run: "CopickRun") -> None:
        """Handle info request from gallery."""
        # Emit the signal that the main widget will connect to
        self.info_requested.emit(run)

    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        # Find the parent CopickPlugin widget to switch back to tree view
        parent_widget = self.parent()
        while parent_widget and not hasattr(parent_widget, "switch_to_tree_view"):
            parent_widget = parent_widget.parent()

        if parent_widget and hasattr(parent_widget, "switch_to_tree_view"):
            parent_widget.switch_to_tree_view()
        else:
            print("Could not find parent widget with switch_to_tree_view method")
