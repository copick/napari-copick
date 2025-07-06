"""napari-specific gallery widget implementation using shared copick-shared-ui."""

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

try:
    # Import directly from module to avoid __init__.py issues
    print("🔍 DEBUG: Attempting to import napari gallery integration...")
    import copick_shared_ui.platform.napari_integration as napari_integration_module

    NapariGalleryIntegration = napari_integration_module.NapariGalleryIntegration
    print("✅ DEBUG: Successfully imported NapariGalleryIntegration")
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

    def __init__(self, viewer: "napari.Viewer", parent: Optional[QWidget] = None):
        self.original_parent = parent
        super().__init__(parent)
        self.viewer = viewer
        self.copick_root = None

        print(f"📊 DEBUG: Gallery SHARED_UI_AVAILABLE = {SHARED_UI_AVAILABLE}")
        if not SHARED_UI_AVAILABLE:
            print("⚠️ DEBUG: Using fallback gallery UI - shared UI not available")
            self._setup_fallback_ui()
            return

        print("✅ DEBUG: Using full gallery widget implementation")

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

    def set_copick_root(self, copick_root) -> None:
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
            # Select the best tomogram from the run
            best_tomogram = self._select_best_tomogram(run)
            if best_tomogram:
                # Load the tomogram using the same async mechanism as the tree view
                self._load_tomogram_async(best_tomogram)

        except Exception as e:
            print(f"Error loading tomogram from gallery: {e}")

    def _load_tomogram_async(self, tomogram: "CopickTomogram") -> None:
        """Load tomogram asynchronously using the same mechanism as the tree view."""
        try:
            # Import the async loader
            from .async_loaders import load_tomogram_worker

            # Get the resolution level from the parent widget's resolution combo
            # Find the parent CopickPlugin widget to access resolution_combo
            parent_widget = self.parent()
            while parent_widget and not hasattr(parent_widget, "resolution_combo"):
                parent_widget = parent_widget.parent()

            if parent_widget and hasattr(parent_widget, "resolution_combo"):
                resolution_level = parent_widget.resolution_combo.currentIndex()

                # Add global loading indicator
                operation_id = f"gallery_load_{tomogram.tomo_type}_{id(tomogram)}"
                parent_widget._add_operation(operation_id, f"Loading {tomogram.tomo_type} from gallery...")

            else:
                # Fallback to medium resolution if we can't find the combo
                resolution_level = 1
                operation_id = None

            # Create worker using napari's threading system
            worker = load_tomogram_worker(tomogram, resolution_level)

            # Store operation info for cleanup
            self._current_operation = (operation_id, parent_widget, tomogram)

            # Connect signals
            worker.yielded.connect(lambda msg: self._on_progress(msg, tomogram))
            worker.returned.connect(lambda result: self._on_tomogram_loaded(result))
            worker.errored.connect(lambda e: self._on_error(str(e), tomogram))

            # Start the worker
            worker.start()

        except Exception as e:
            print(f"Error starting async tomogram load: {e}")

    def _on_progress(self, message: str, tomogram: "CopickTomogram") -> None:
        """Handle loading progress updates."""
        pass

    def _on_tomogram_loaded(self, result: dict) -> None:
        """Handle successful tomogram loading."""
        try:
            data = result["data"]
            voxel_size = result["voxel_size"]
            name = result["name"]

            # Add to napari viewer with the same naming convention as tree view
            self.viewer.add_image(
                data,
                name=name,  # Use the same name format as tree view
                scale=voxel_size,
                opacity=0.8,
                blending="additive",
            )

            # Clean up global loading indicator
            self._cleanup_operation()

        except Exception:
            # Clean up global loading indicator even on error
            self._cleanup_operation()

    def _on_error(self, error: str, tomogram: "CopickTomogram") -> None:
        """Handle tomogram loading errors."""
        print(f"Gallery: Error loading {tomogram.tomo_type}: {error}")
        # Clean up global loading indicator
        self._cleanup_operation()

    def _cleanup_operation(self) -> None:
        """Clean up the current loading operation."""
        if hasattr(self, "_current_operation"):
            operation_id, parent_widget, tomogram = self._current_operation
            if operation_id and parent_widget and hasattr(parent_widget, "_remove_operation"):
                parent_widget._remove_operation(operation_id)
            delattr(self, "_current_operation")

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
            pass

    def _select_best_tomogram(self, run: "CopickRun"):
        """Select the best tomogram from a run (prefer denoised, highest voxel spacing)."""
        try:
            all_tomograms = []

            # Collect all tomograms from all voxel spacings
            for vs in run.voxel_spacings:
                for tomo in vs.tomograms:
                    all_tomograms.append(tomo)

            if not all_tomograms:
                return None

            # Preference order for tomogram types (denoised first)
            preferred_types = ["denoised", "wbp"]

            # Group by voxel spacing (highest first)
            voxel_spacings = sorted({tomo.voxel_spacing.voxel_size for tomo in all_tomograms}, reverse=True)

            # Try each voxel spacing, starting with highest
            for vs_size in voxel_spacings:
                vs_tomograms = [tomo for tomo in all_tomograms if tomo.voxel_spacing.voxel_size == vs_size]

                # Try preferred types in order
                for preferred_type in preferred_types:
                    for tomo in vs_tomograms:
                        if preferred_type.lower() in tomo.tomo_type.lower():
                            return tomo

                # If no preferred type found, return the first tomogram at this voxel spacing
                if vs_tomograms:
                    return vs_tomograms[0]

            # Fallback: return any tomogram
            return all_tomograms[0] if all_tomograms else None

        except Exception as e:
            print(f"Error selecting best tomogram: {e}")
            return None
