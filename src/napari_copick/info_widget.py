"""napari-specific implementation of the copick info widget."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

try:
    from napari.qt.threading import create_worker, thread_worker  # noqa: F401

    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

# Import shared components
try:
    print("🔍 DEBUG: Attempting to import from copick_shared_ui.core.models...")
    from copick_shared_ui.core.models import (
        AbstractImageInterface,
        AbstractInfoSessionInterface,
        AbstractThemeInterface,
        AbstractWorkerInterface,
    )
    print("✅ DEBUG: Successfully imported core models")
    
    print("🔍 DEBUG: Attempting to import theming components...")
    from copick_shared_ui.theming.colors import get_color_scheme
    from copick_shared_ui.theming.styles import (
        generate_button_stylesheet,
        generate_input_stylesheet,
        generate_stylesheet,
    )
    from copick_shared_ui.theming.theme_detection import detect_napari_theme
    print("✅ DEBUG: Successfully imported theming components")
    
    print("🔍 DEBUG: Attempting to import info widget...")
    from copick_shared_ui.widgets.info.info_widget import CopickInfoWidget
    print("✅ DEBUG: Successfully imported CopickInfoWidget")

    print("✅ DEBUG: All imports successful - SHARED_UI_AVAILABLE = True")
    SHARED_UI_AVAILABLE = True
except ImportError as e:
    print(f"❌ DEBUG: Import failed - {e}")
    print(f"❌ DEBUG: SHARED_UI_AVAILABLE = False")
    import traceback
    traceback.print_exc()
    SHARED_UI_AVAILABLE = False

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickTomogram

print(f"📊 DEBUG: NAPARI_AVAILABLE = {NAPARI_AVAILABLE}, SHARED_UI_AVAILABLE = {SHARED_UI_AVAILABLE}")

if NAPARI_AVAILABLE and SHARED_UI_AVAILABLE:
    print("✅ DEBUG: Using full info widget implementation")

    class NapariInfoSessionInterface(AbstractInfoSessionInterface):
        """napari-specific session interface for info widget."""

        def __init__(self, viewer, plugin_widget):
            self.viewer = viewer
            self.plugin_widget = plugin_widget

        def load_tomogram_and_switch_view(self, tomogram: "CopickTomogram") -> None:
            """Load the tomogram and switch to main napari view."""
            # Switch to tree view and load tomogram
            self.plugin_widget.switch_to_tree_view()

            # Find the tomogram in the tree and load it
            # For now, just use the plugin's async loading mechanism
            # This could be enhanced to find the exact tree item
            self.plugin_widget.load_tomogram_async(tomogram, None)

        def navigate_to_gallery(self) -> None:
            """Navigate back to gallery view."""
            self.plugin_widget.switch_to_gallery_view()

        def expand_run_in_tree(self, run: "CopickRun") -> None:
            """Expand the run in the tree view."""
            # Switch to tree view first
            self.plugin_widget.switch_to_tree_view()
            # The tree expansion is handled by the main plugin


    class NapariThemeInterface(AbstractThemeInterface):
        """napari-specific theme interface."""

        def __init__(self, viewer):
            self.viewer = viewer
            self._theme_change_callbacks: List[callable] = []

        def get_theme_colors(self) -> Dict[str, str]:
            """Get color scheme for current napari theme."""
            theme = detect_napari_theme(self.viewer)
            return get_color_scheme(theme)

        def get_theme_stylesheet(self) -> str:
            """Get base stylesheet for current napari theme."""
            theme = detect_napari_theme(self.viewer)
            return generate_stylesheet(theme)

        def get_button_stylesheet(self, button_type: str = "primary") -> str:
            """Get button stylesheet for current napari theme."""
            theme = detect_napari_theme(self.viewer)
            return generate_button_stylesheet(button_type, theme)

        def get_input_stylesheet(self) -> str:
            """Get input field stylesheet for current napari theme."""
            theme = detect_napari_theme(self.viewer)
            return generate_input_stylesheet(theme)

        def connect_theme_changed(self, callback: callable) -> None:
            """Connect to napari theme change events."""
            self._theme_change_callbacks.append(callback)
            # napari doesn't have a direct theme change signal
            # This could be enhanced to listen for theme changes

        def _emit_theme_changed(self) -> None:
            """Emit theme changed to all callbacks."""
            for callback in self._theme_change_callbacks:
                callback()

    class NapariImageInterface(AbstractImageInterface):
        """napari-specific image/pixmap interface."""

        # create_pixmap_from_array is now inherited from AbstractImageInterface

        def scale_pixmap(self, pixmap: Any, size: tuple, smooth: bool = True) -> Any:
            """Scale a pixmap to the specified size."""
            from qtpy.QtCore import Qt

            if pixmap is None:
                return None

            width, height = size
            transform_mode = Qt.SmoothTransformation if smooth else Qt.FastTransformation
            return pixmap.scaled(width, height, Qt.KeepAspectRatio, transform_mode)

        def save_pixmap(self, pixmap: Any, path: str) -> bool:
            """Save pixmap to file."""
            try:
                if pixmap is None:
                    return False
                return pixmap.save(path)
            except Exception:
                return False

        def load_pixmap(self, path: str) -> Optional[Any]:
            """Load pixmap from file."""
            try:
                from qtpy.QtGui import QPixmap

                pixmap = QPixmap(path)
                return pixmap if not pixmap.isNull() else None
            except Exception:
                return None

    class NapariWorkerInterface(AbstractWorkerInterface):
        """napari-specific worker interface using @thread_worker."""

        def __init__(self):
            self._active_workers = []

        def start_thumbnail_worker(
            self,
            item: Union["CopickRun", "CopickTomogram"],
            thumbnail_id: str,
            callback: callable,
            force_regenerate: bool = False,
        ) -> None:
            """Start a thumbnail loading worker using napari's @thread_worker."""
            if not NAPARI_AVAILABLE:
                callback(thumbnail_id, None, "napari not available")
                return

            @thread_worker
            def load_thumbnail():
                try:
                    from copick_shared_ui.workers.napari import NapariThumbnailWorker

                    # Create the worker with the item (run or tomogram)
                    def dummy_callback(tid, pixmap, error):
                        # This callback will be overridden by the worker's internal handling
                        pass
                    
                    worker = NapariThumbnailWorker(item, thumbnail_id, dummy_callback, force_regenerate)
                    
                    # Generate thumbnail using unified system
                    pixmap, error = worker.generate_thumbnail_pixmap()
                    
                    if error:
                        return None, error
                    else:
                        return pixmap, None

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return None, str(e)

            # Create and connect the worker
            worker = load_thumbnail()
            worker.returned.connect(lambda result: callback(thumbnail_id, result[0], result[1]))
            worker.errored.connect(lambda error: callback(thumbnail_id, None, str(error)))

            # Store reference and start
            self._active_workers.append(worker)
            worker.start()

        def clear_workers(self) -> None:
            """Clear all pending workers."""
            for worker in self._active_workers:
                if hasattr(worker, "quit"):
                    worker.quit()
            self._active_workers.clear()

        def shutdown_workers(self, timeout_ms: int = 3000) -> None:
            """Shutdown all workers with timeout."""
            self.clear_workers()
            # napari workers don't have a shutdown mechanism like QThreadPool

    class NapariCopickInfoWidget(CopickInfoWidget):
        """napari-specific copick info widget."""

        def __init__(self, viewer, plugin_widget, parent: Optional[QWidget] = None):
            self.viewer = viewer
            self.plugin_widget = plugin_widget

            # Create platform interfaces
            session_interface = NapariInfoSessionInterface(viewer, plugin_widget)
            theme_interface = NapariThemeInterface(viewer)
            worker_interface = NapariWorkerInterface()
            image_interface = NapariImageInterface()

            super().__init__(
                session_interface=session_interface,
                theme_interface=theme_interface,
                worker_interface=worker_interface,
                image_interface=image_interface,
                parent=parent,
            )

            # Connect tomogram clicked signal to load tomogram
            self.tomogram_clicked.connect(self._on_tomogram_clicked)

        def _on_tomogram_clicked(self, tomogram: "CopickTomogram") -> None:
            """Handle tomogram click by loading it in napari."""
            # Use the plugin's loading mechanism
            self.plugin_widget.load_tomogram_async(tomogram, None)

else:
    # Fallback if dependencies are not available
    print("⚠️ DEBUG: Using fallback info widget - dependencies not available")
    class NapariCopickInfoWidget(QWidget):
        """Fallback info widget when dependencies are not available."""

        def __init__(self, viewer, plugin_widget, parent: Optional[QWidget] = None):
            super().__init__(parent)
            from qtpy.QtWidgets import QLabel, QVBoxLayout

            layout = QVBoxLayout()
            label = QLabel("Info widget not available\n\nThe copick-shared-ui package is required.")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
            layout.addWidget(label)
            self.setLayout(layout)

        def set_run(self, run) -> None:
            """Dummy method for compatibility."""
            pass

        def delete(self) -> None:
            """Dummy method for compatibility."""
            pass
