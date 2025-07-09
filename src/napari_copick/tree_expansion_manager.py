"""Tree expansion state management for napari-copick plugin."""

import logging
from typing import Any, Dict, Set

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QTreeWidgetItem


class TreeExpansionManager:
    """Manages tree expansion state preservation and restoration."""

    def __init__(self, parent_widget):
        """Initialize the tree expansion manager.

        Args:
            parent_widget: The main CopickPlugin widget instance
        """
        self.parent_widget = parent_widget
        self.logger = logging.getLogger("CopickPlugin.TreeExpansionManager")

        # Tree expansion state tracking
        self.tree_expansion_state: Dict[str, bool] = {}
        self.pending_expansions: Set[str] = set()

    def save_tree_expansion_state(self) -> None:
        """Save the current expansion state of the tree."""
        self.tree_expansion_state.clear()

        def save_item_state(item: QTreeWidgetItem, path: str = "") -> None:
            """Recursively save expansion state for an item and its children."""
            current_path = f"{path}/{item.text(0)}" if path else item.text(0)

            if item.isExpanded():
                self.tree_expansion_state[current_path] = True

            # Save state for children
            for i in range(item.childCount()):
                child = item.child(i)
                save_item_state(child, current_path)

        # Save state for all top-level items
        for i in range(self.parent_widget.tree_view.topLevelItemCount()):
            item = self.parent_widget.tree_view.topLevelItem(i)
            save_item_state(item)

    def restore_tree_expansion_state(self) -> None:
        """Restore the previously saved expansion state of the tree."""
        if not self.tree_expansion_state:
            return

        # Clear pending expansions from previous restoration
        self.pending_expansions.clear()

        # Add all previously expanded paths to pending expansions
        for path in self.tree_expansion_state.keys():
            self.pending_expansions.add(path)

        # Use a delayed restoration to ensure items are properly loaded
        def delayed_restore():
            self._force_expansion_restoration()

        # Schedule restoration after a short delay
        QTimer.singleShot(100, delayed_restore)

    def _force_expansion_restoration(self) -> None:
        """Force expansion of items that were previously expanded."""
        # Start with top-level items (runs)
        for i in range(self.parent_widget.tree_view.topLevelItemCount()):
            item = self.parent_widget.tree_view.topLevelItem(i)
            self._force_expand_item_if_needed(item)

    def _force_expand_item_if_needed(self, item: QTreeWidgetItem, path: str = "") -> None:
        """Force expand an item if it was previously expanded.

        Args:
            item: The tree widget item to check and potentially expand
            path: The current path to the item
        """
        current_path = f"{path}/{item.text(0)}" if path else item.text(0)

        # Check if this item should be expanded
        if current_path in self.pending_expansions:
            # Remove from pending since we're handling it
            self.pending_expansions.discard(current_path)

            # Force expansion by triggering the expand handler
            if not item.isExpanded():
                item.setExpanded(True)

                # Trigger the expansion handler to load children asynchronously
                self.parent_widget.tree_view.handle_item_expand(item)

        # Process existing children
        for i in range(item.childCount()):
            child = item.child(i)
            self._force_expand_item_if_needed(child, current_path)

    def refresh_tree_after_save(self, save_result: Dict[str, Any]) -> None:
        """Refresh tree after saving, attempting to preserve expansion state.

        Args:
            save_result: The result from the save operation
        """
        try:
            # For now, fall back to full tree refresh with expansion preservation
            # In the future, this could be optimized to only refresh specific branches
            self.populate_tree(preserve_expansion=True)
        except Exception as e:
            self.logger.warning(f"Could not preserve expansion state during refresh: {e}")
            # Fall back to regular tree population
            self.populate_tree(preserve_expansion=False)

    def populate_tree(self, preserve_expansion: bool = True) -> None:
        """Populate the tree, optionally preserving expansion state.

        Args:
            preserve_expansion: Whether to preserve the current expansion state
        """

        if preserve_expansion:
            # Save current expansion state before repopulating
            self.save_tree_expansion_state()

        # Populate the tree
        self.parent_widget.tree_view.populate_tree(self.parent_widget.root)

        if preserve_expansion:
            # Restore expansion state after populating
            self.restore_tree_expansion_state()
