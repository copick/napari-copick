"""napari-specific implementation of the copick info widget."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

try:
    from napari.qt.threading import create_worker, thread_worker  # noqa: F401

    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

# Import shared components
try:
    from copick_shared_ui.gallery.core.models import (
        AbstractImageInterface,
        AbstractThemeInterface,
        AbstractWorkerInterface,
    )
    from copick_shared_ui.gallery.theming.colors import get_color_scheme
    from copick_shared_ui.gallery.theming.styles import (
        generate_button_stylesheet,
        generate_input_stylesheet,
        generate_stylesheet,
    )
    from copick_shared_ui.gallery.theming.theme_detection import detect_napari_theme
    from copick_shared_ui.info.core.info_widget import CopickInfoWidget
    from copick_shared_ui.info.core.models import AbstractInfoSessionInterface

    SHARED_UI_AVAILABLE = True
except ImportError:
    SHARED_UI_AVAILABLE = False

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickTomogram

if NAPARI_AVAILABLE and SHARED_UI_AVAILABLE:

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

        def get_portal_link(self, item) -> Optional[str]:
            """Get CryoET Data Portal link for an item if applicable."""
            try:
                # Import here to avoid circular imports
                from copick.impl.cryoet_data_portal import CopickRunCDP

                # Check if this is a CryoET Data Portal project
                if hasattr(item, "run") and isinstance(item.run, CopickRunCDP):
                    run_id = item.run.portal_run_id

                    if hasattr(item, "meta") and hasattr(item.meta, "portal_tomo_id"):
                        # Tomogram link
                        return f"https://cryoetdataportal.czscience.com/runs/{run_id}?table-tab=Tomograms"
                    elif hasattr(item, "meta") and hasattr(item.meta, "portal_annotation_id"):
                        # Annotation link (picks, segmentations)
                        return f"https://cryoetdataportal.czscience.com/runs/{run_id}?table-tab=Annotations"
                    elif (
                        hasattr(item, "voxel_spacing")
                        and hasattr(item.voxel_spacing, "run")
                        and isinstance(item.voxel_spacing.run, CopickRunCDP)
                    ):
                        # Voxel spacing or tomogram via voxel spacing
                        run_id = item.voxel_spacing.run.portal_run_id
                        return f"https://cryoetdataportal.czscience.com/runs/{run_id}"
                    else:
                        # General run link
                        return f"https://cryoetdataportal.czscience.com/runs/{run_id}"

                return None
            except Exception:
                return None

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

        def create_pixmap_from_array(self, array: Any) -> Any:
            """Create a QPixmap from numpy array."""
            import numpy as np
            from qtpy.QtGui import QImage, QPixmap

            if array.ndim == 2:
                # Grayscale image
                height, width = array.shape
                bytes_per_line = width

                # Ensure array is uint8
                if array.dtype != np.uint8:
                    # Normalize to 0-255 range
                    array_min, array_max = array.min(), array.max()
                    if array_max > array_min:
                        array = ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
                    else:
                        array = np.zeros_like(array, dtype=np.uint8)

                # Create QImage from array
                qimage = QImage(array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                # Convert to QPixmap
                return QPixmap.fromImage(qimage)

            elif array.ndim == 3 and array.shape[2] == 3:
                # RGB image
                height, width, channels = array.shape
                bytes_per_line = width * channels

                # Ensure array is uint8
                if array.dtype != np.uint8:
                    array = (array * 255).astype(np.uint8)

                # Create QImage from array
                qimage = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Convert to QPixmap
                return QPixmap.fromImage(qimage)

            else:
                return None

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
            run: "CopickRun",
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
                    from copick_shared_ui.gallery.workers.base_workers import AbstractThumbnailWorker

                    # Create a dummy worker to access the shared methods
                    class ThumbnailHelper(AbstractThumbnailWorker):
                        def start(self):
                            pass

                        def cancel(self):
                            pass

                    helper = ThumbnailHelper(run, thumbnail_id, callback, force_regenerate)

                    # Select best tomogram
                    tomogram = helper._select_best_tomogram(run)
                    if not tomogram:
                        return None, "No tomogram found"

                    # Generate thumbnail array
                    thumbnail_array = helper._generate_thumbnail_array(tomogram)
                    if thumbnail_array is None:
                        return None, "Failed to generate thumbnail"

                    # Convert to QPixmap
                    import numpy as np
                    from qtpy.QtGui import QImage, QPixmap

                    # Ensure array is uint8
                    if thumbnail_array.dtype != np.uint8:
                        array_min, array_max = thumbnail_array.min(), thumbnail_array.max()
                        if array_max > array_min:
                            thumbnail_array = ((thumbnail_array - array_min) / (array_max - array_min) * 255).astype(
                                np.uint8,
                            )
                        else:
                            thumbnail_array = np.zeros_like(thumbnail_array, dtype=np.uint8)

                    height, width = thumbnail_array.shape
                    bytes_per_line = width
                    qimage = QImage(thumbnail_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)

                    return pixmap, None

                except Exception as e:
                    pass
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
