"""Napari-specific CLI tools widget.

Provides the platform-specific interface implementations and integration
factory for embedding the shared ClickCommandBrowser in napari-copick.
"""

import json
import logging
import tempfile
from typing import TYPE_CHECKING, Any, List, Optional

try:
    from qtpy.QtWidgets import QVBoxLayout, QWidget
except ImportError:
    from Qt.QtWidgets import QVBoxLayout, QWidget

from copick_shared_ui.core.models import (
    AbstractCLIContextInterface,
    AbstractCLIRefreshInterface,
)
from copick_shared_ui.widgets.cli.command_browser import ClickCommandBrowser

if TYPE_CHECKING:
    import napari

    from napari_copick.widget import CopickPlugin

logger = logging.getLogger("NapariCLIWidget")

# Categories that require a view refresh after command execution
_REFRESH_CATEGORIES = {"Data Management", "Data Processing", "Download"}


class NapariCLIContextInterface(AbstractCLIContextInterface):
    """Reads CLI context from the napari CopickPlugin widget."""

    def __init__(self, plugin: "CopickPlugin"):
        self._plugin = plugin
        self._temp_config_path: Optional[str] = None

    def get_config_path(self) -> Optional[str]:
        if self._plugin.config_path:
            return self._plugin.config_path

        # Fallback: serialize in-memory config to temp file
        if self._plugin.root is not None:
            try:
                config_dict = self._plugin.root.config.model_dump()
                if self._temp_config_path is None:
                    f = tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".json",
                        prefix="copick_config_",
                        delete=False,
                    )
                    self._temp_config_path = f.name
                    f.close()
                with open(self._temp_config_path, "w") as f:
                    json.dump(config_dict, f)
                return self._temp_config_path
            except Exception as e:
                logger.warning(f"Failed to serialize config: {e}")

        return None

    def get_copick_root(self) -> Optional[Any]:
        return self._plugin.root

    def get_run_names(self) -> List[str]:
        if self._plugin.root:
            return [r.name for r in self._plugin.root.runs]
        return []

    def get_object_names(self) -> List[str]:
        if self._plugin.root:
            return [obj.name for obj in self._plugin.root.pickable_objects]
        return []

    def get_voxel_spacings(self) -> List[float]:
        if self._plugin.root and self._plugin.root.runs:
            # Collect unique voxel spacings from first run
            spacings = set()
            for run in self._plugin.root.runs[:5]:  # Sample first 5 runs
                for vs in run.voxel_spacings:
                    spacings.add(vs.voxel_size)
            return sorted(spacings)
        return []

    def get_user_ids(self) -> List[str]:
        if self._plugin.root:
            user_ids = set()
            for run in self._plugin.root.runs[:5]:
                for picks in run.picks:
                    user_ids.add(picks.user_id)
            return sorted(user_ids)
        return []

    def get_session_ids(self) -> List[str]:
        if self._plugin.root:
            session_ids = set()
            for run in self._plugin.root.runs[:5]:
                for picks in run.picks:
                    session_ids.add(picks.session_id)
            return sorted(session_ids)
        return []

    def get_tomo_types(self) -> List[str]:
        if self._plugin.root and self._plugin.root.runs:
            tomo_types = set()
            for run in self._plugin.root.runs[:5]:
                for vs in run.voxel_spacings:
                    for tomo in vs.tomograms:
                        tomo_types.add(tomo.tomo_type)
            return sorted(tomo_types)
        return []

    def get_selected_copick_object(self) -> Optional[Any]:
        try:
            from qtpy.QtCore import Qt

            tree = self._plugin.tree_view
            items = tree.selectedItems()
            if items:
                return items[0].data(0, Qt.UserRole)
        except Exception:
            pass
        return None

    def connect_selection_changed(self, callback) -> None:
        try:
            from qtpy.QtCore import Qt

            self._plugin.tree_view.itemClicked.connect(
                lambda item, col: callback(item.data(0, Qt.UserRole)),
            )
        except Exception:
            pass

    def disconnect_selection_changed(self, callback) -> None:
        # Disconnecting lambdas is not straightforward; the form cleanup
        # handles this by being destroyed, which disconnects all signals.
        pass


class NapariCLIRefreshInterface(AbstractCLIRefreshInterface):
    """Refreshes napari views after CLI command execution."""

    def __init__(self, plugin: "CopickPlugin"):
        self._plugin = plugin

    def refresh_after_command(self, command_category: str) -> None:
        if command_category not in _REFRESH_CATEGORIES:
            return

        try:
            self._plugin.tree_expansion_manager.populate_tree(preserve_expansion=True)
        except Exception as e:
            logger.warning(f"Failed to refresh tree: {e}")

        try:
            self._plugin._update_gallery()
        except Exception as e:
            logger.warning(f"Failed to refresh gallery: {e}")


class NapariCopickCLIWidget(QWidget):
    """Wrapper widget for embedding the CLI tool browser in napari-copick."""

    def __init__(
        self,
        viewer: "napari.Viewer",
        plugin: "CopickPlugin",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._plugin = plugin
        self._viewer = viewer
        self._browser: Optional[ClickCommandBrowser] = None

        # Create platform interfaces
        self._context = NapariCLIContextInterface(plugin)
        self._refresh = NapariCLIRefreshInterface(plugin)

        # Import shared-ui theme and worker interfaces
        from copick_shared_ui.platform.napari_integration import (
            NapariThemeInterface,
            NapariWorkerInterface,
        )

        self._theme = NapariThemeInterface(viewer)
        self._worker = NapariWorkerInterface()

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._browser = ClickCommandBrowser(
            context_interface=self._context,
            theme_interface=self._theme,
            worker_interface=self._worker,
            refresh_interface=self._refresh,
            parent=self,
        )
        layout.addWidget(self._browser)

    def populate_commands(self) -> None:
        """Discover and display CLI commands. Call after config is loaded."""
        if self._browser is not None:
            self._browser.populate_commands()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._browser is not None:
            self._browser.cleanup()
