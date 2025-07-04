import sys

import copick
import dask.array as da
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
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


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
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()

        self.root = None
        self.selected_run = None
        self.current_layer = None
        self.session_id = "17"
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

        # Hierarchical tree view
        self.tree_view = QTreeWidget()
        self.tree_view.setHeaderLabel("Copick Project")
        self.tree_view.itemExpanded.connect(self.handle_item_expand)
        self.tree_view.itemClicked.connect(self.handle_item_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_context_menu)
        layout.addWidget(self.tree_view)

        # Info label
        self.info_label = QLabel("Select a pick to get started")
        layout.addWidget(self.info_label)

        self.setLayout(layout)

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

    def load_config(self, config_path=None):
        if config_path:
            self.root = copick.from_file(config_path)
            self.populate_tree()
            self.info_label.setText(f"Loaded config from {config_path}")

    def load_from_dataset_ids(self, dataset_ids=None, overlay_root="/tmp/overlay_root"):
        if dataset_ids:
            self.root = copick.from_czcdp_datasets(
                dataset_ids=dataset_ids,
                overlay_root=overlay_root,
                overlay_fs_args={"auto_mkdir": True},
            )
            self.populate_tree()
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
            self.expand_run(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing(item, data)

    def expand_run(self, item, run):
        if not item.childCount():
            for voxel_spacing in run.voxel_spacings:
                spacing_item = QTreeWidgetItem(item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"])
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

            # Add picks nested by user_id, session_id, and pickable_object_name
            picks = run.picks
            picks_item = QTreeWidgetItem(item, ["Picks"])
            user_dict = {}
            for pick in picks:
                if pick.meta.user_id not in user_dict:
                    user_dict[pick.meta.user_id] = {}
                if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                    user_dict[pick.meta.user_id][pick.meta.session_id] = []
                user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)

            for user_id, sessions in user_dict.items():
                user_item = QTreeWidgetItem(picks_item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(user_item, [f"Session: {session_id}"])
                    for pick in picks:
                        pick_child = QTreeWidgetItem(session_item, [pick.meta.pickable_object_name])
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

    def expand_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in voxel_spacing.tomograms:
                tomo_child = QTreeWidgetItem(tomogram_item, [tomogram.meta.tomo_type])
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations = voxel_spacing.run.get_segmentations(voxel_size=voxel_spacing.meta.voxel_size)
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(segmentation_item, [segmentation.meta.name])
                seg_child.setData(0, Qt.UserRole, segmentation)
            item.addChild(segmentation_item)

    def handle_item_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.info_label.setText(f"Run: {data.meta.name}")
            self.selected_run = data
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.info_label.setText(f"Voxel Spacing: {data.meta.voxel_size}")
            self.lazy_load_voxel_spacing(item, data)
        elif isinstance(data, copick.models.CopickTomogram):
            self.load_tomogram(data)
        elif isinstance(data, copick.models.CopickSegmentation):
            self.load_segmentation(data)
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
            self.expand_voxel_spacing(item, voxel_spacing)

    def load_tomogram(self, tomogram):
        """
        Load a tomogram directly using napari's multiscale API instead of using napari-ome-zarr.
        This handles the multiscale zarr arrays directly.
        """
        zarr_path = tomogram.zarr()
        zarr_group = zarr.open(zarr_path, "r")

        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]  # noqa: SIM118
        scale_levels.sort(key=int)

        if not scale_levels:
            self.info_label.setText(f"Error: No scale levels found in tomogram: {tomogram.meta.tomo_type}")
            return

        # Calculate scaling factors between resolution levels
        all_arrays = []
        all_data = []

        # Get the highest resolution data
        base_array = zarr_group[scale_levels[0]]
        base_shape = base_array.shape

        # Calculate voxel size from metadata or fallback to uniform scaling
        voxel_size = [tomogram.voxel_spacing.meta.voxel_size] * 3

        # Collect all scale levels and calculate scale factors
        scales = []
        for level in scale_levels:
            array = zarr_group[level]
            # Create Dask array for lazy loading
            dask_array = da.from_array(array, chunks=array.chunks)
            all_arrays.append(array)
            all_data.append(dask_array)

            # Calculate scale relative to the base level
            scale_factor = [bs / s for bs, s in zip(base_shape, array.shape)]
            scales.append(scale_factor)

        # Add multiscale image to the viewer
        _ = self.viewer.add_image(
            all_data,
            scale=voxel_size,
            multiscale=True,
            name=f"Tomogram: {tomogram.meta.tomo_type}",
            contrast_limits=[0, 1],
        )

        self.info_label.setText(f"Loaded Tomogram: {tomogram.meta.tomo_type} with {len(scale_levels)} scale levels")

    def load_segmentation(self, segmentation):
        zarr_data = zarr.open(segmentation.zarr(), "r+")
        data = zarr_data["data"] if "data" in zarr_data else zarr_data["0"]

        scale = [segmentation.meta.voxel_size] * 3

        # Create a color map based on copick colors
        colormap = self.get_copick_colormap()
        painting_layer = self.viewer.add_labels(data, name=f"Segmentation: {segmentation.meta.name}", scale=scale)
        painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        painting_layer.painting_labels = [obj.label for obj in self.root.config.pickable_objects]
        self.class_labels_mapping = {obj.label: obj.name for obj in self.root.config.pickable_objects}

        self.info_label.setText(f"Loaded Segmentation: {segmentation.meta.name}")

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
        print("Opening context menu")
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
        if item.text(0) == "Segmentations" or item.text(0) == "Picks":
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

        tomo = zarr.open(run.voxel_spacings[0].tomograms[0].zarr(), "r")["0"]

        shape = tomo.shape
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copick Plugin")
    parser.add_argument("--config_path", type=str, help="Path to the copick config file", default=None)
    parser.add_argument(
        "--dataset_ids",
        type=int,
        nargs="+",
        help="Dataset IDs to include in the project (space separated)",
        default=None,
    )
    parser.add_argument(
        "--overlay_root",
        type=str,
        default="/tmp/overlay_root",
        help="Root URL for the overlay storage when using dataset IDs",
    )
    args = parser.parse_args()

    if not args.config_path and not args.dataset_ids:
        print("Either --config_path or --dataset_ids must be provided")
        sys.exit(1)
    elif args.config_path and args.dataset_ids:
        print("Only one of --config_path or --dataset_ids should be provided, not both")
        sys.exit(1)

    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(
        viewer,
        config_path=args.config_path,
        dataset_ids=args.dataset_ids,
        overlay_root=args.overlay_root,
    )
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()
