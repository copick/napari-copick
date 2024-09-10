import dask.array as da
import numpy as np
import copick
import zarr
import napari
import sys
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QLineEdit,
    QMenu,
    QAction,
    QFormLayout,
    QComboBox,
    QSpinBox,
)
from qtpy.QtCore import Qt, QPoint
from napari.utils import DirectLabelColormap


class CopickPlugin(QWidget):
    def __init__(self, viewer, config_path=None):
        super().__init__()
        self.viewer = viewer
        self.root = None
        self.selected_run = None
        self.current_layer = None
        self.session_id = "17"
        self.setup_ui()
        if config_path:
            self.load_config(config_path)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Config loading button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # Hierarchical tree view
        self.tree_view = QTreeWidget()
        self.tree_view.setHeaderLabel("Copick Project")
        self.tree_view.itemExpanded.connect(self.handle_item_expand)
        self.tree_view.itemClicked.connect(self.handle_item_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
            self.open_context_menu
        )
        layout.addWidget(self.tree_view)

        # Info label
        self.info_label = QLabel("Select a pick to get started")
        layout.addWidget(self.info_label)

        self.setLayout(layout)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config", "", "JSON Files (*.json)"
        )
        if path:
            self.load_config(path)

    def load_config(self, path=None):
        if path:
            self.root = copick.from_file(path)
            self.populate_tree()

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
                spacing_item = QTreeWidgetItem(
                    item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"]
                )
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(
                    QTreeWidgetItem.ShowIndicator
                )

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
                    session_item = QTreeWidgetItem(
                        user_item, [f"Session: {session_id}"]
                    )
                    for pick in picks:
                        pick_child = QTreeWidgetItem(
                            session_item, [pick.meta.pickable_object_name]
                        )
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

    def expand_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in voxel_spacing.tomograms:
                tomo_child = QTreeWidgetItem(
                    tomogram_item, [tomogram.meta.tomo_type]
                )
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(
                    segmentation_item, [segmentation.meta.name]
                )
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
        zarr_path = tomogram.zarr()
        zarr_group = zarr.open(zarr_path, "r")

        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]
        scale_levels.sort(key=int)

        data = [zarr_group[level] for level in scale_levels]

        # data = [da.from_zarr(str(zarr_path), level) * (int(level) + 1) / 2 for level in scale_levels]

        # Highest scale level only
        # data = zarr.open(tomogram.zarr(), 'r')["0"]

        # TODO scale needs to account for scale pyramid (4x for the lowest scale in this case)
        
        scale = [tomogram.voxel_spacing.meta.voxel_size] * 3
        self.viewer.add_image(
            data, name=f"Tomogram: {tomogram.meta.tomo_type}", scale=scale
        )
        self.info_label.setText(
            f"Loaded Tomogram: {tomogram.meta.tomo_type} with num scales = {len(scale_levels)}"
        )

    def load_segmentation(self, segmentation):
        zarr_data = zarr.open(segmentation.zarr().path, "r")
        if "data" in zarr_data:
            data = zarr_data["data"]
        else:
            data = zarr_data[:]

        scale = [segmentation.meta.voxel_size] * 3

        # Create a color map based on copick colors
        colormap = self.get_copick_colormap()
        painting_layer = self.viewer.add_labels(
            data, name=f"Segmentation: {segmentation.meta.name}", scale=scale
        )
        painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        painting_layer.painting_labels = [
            obj.label for obj in self.root.config.pickable_objects
        ]
        self.class_labels_mapping = {
            obj.label: obj.name for obj in self.root.config.pickable_objects
        }

        self.info_label.setText(
            f"Loaded Segmentation: {segmentation.meta.name}"
        )

    def get_copick_colormap(self, pickable_objects=None):
        if not pickable_objects:
            pickable_objects = self.root.config.pickable_objects
        colormap = {
            obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects
        }
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def load_picks(self, pick_set, parent_run):
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [
                        (p.location.z, p.location.y, p.location.x)
                        for p in pick_set.points
                    ]
                    color = (
                        pick_set.color
                        if pick_set.color
                        else (255, 255, 255, 255)
                    )  # Default to white if color is not set
                    colors = np.tile(
                        np.array(
                            [
                                color[0] / 255.0,
                                color[1] / 255.0,
                                color[2] / 255.0,
                                color[3] / 255.0,
                            ]
                        ),
                        (len(points), 1),
                    )  # Create an array with the correct shape
                    pickable_object = [
                        obj
                        for obj in self.root.pickable_objects
                        if obj.name == pick_set.pickable_object_name
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
                    self.info_label.setText(
                        f"Loaded Picks: {pick_set.meta.pickable_object_name}"
                    )
                else:
                    self.info_label.setText(
                        f"No points found for Picks: {pick_set.meta.pickable_object_name}"
                    )
            else:
                self.info_label.setText(
                    f"No pick set found for Picks: {pick_set.meta.pickable_object_name}"
                )
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
            )
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
            )
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def create_segmentation(
        self, widget, run, name, session_id, user_id, voxel_size
    ):
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=str(session_id),
            is_multilabel=True,
            user_id=user_id,
        )

        tomo = zarr.open(run.voxel_spacings[0].tomograms[0].zarr().path, "r")[
            "0"
        ]

        shape = tomo.shape
        dtype = np.int32

        # Create an empty Zarr array for the segmentation
        zarr_file = zarr.open(seg.zarr().path, mode="w")
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
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the copick config file",
    )
    args = parser.parse_args()

    config_path = None
    # config_path = "/Users/kharrington/Data/copick/chlamy_10301.json"
    # config_path = "/Volumes/CZII_A/cellcanvas_tutorial/copick-local.json"
    # config_path = "/Users/kharrington/Data/copick/synthetic_data_10439.json"

    viewer = napari.Viewer()
    # copick_plugin = CopickPlugin(viewer, config_path=config_path)
    copick_plugin = CopickPlugin(viewer, config_path=(args.config_path if config_path is None else config_path))
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()
