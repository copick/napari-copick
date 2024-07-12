import copick
import json
import zarr
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QSlider, QFileDialog
from qtpy.QtCore import Qt
from copick.impl.filesystem import CopickRootFSSpec

class CopickPlugin(QWidget):
    # CONFIG_PATH = "/Volumes/CZII_A/cellcanvas_tutorial/copick.json"
    CONFIG_PATH = None

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.root = None
        self.selected_run = None
        self.current_layer = None
        self.current_pick = None
        self.pick_index = 0
        self.picks = None
        self.voxel_spacing = 10
        self.setup_ui()
        self.session_id = "0"

        # Load config from static path if provided
        if CopickPlugin.CONFIG_PATH:
            self.load_config(CopickPlugin.CONFIG_PATH)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Config loading button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # Dropdown for available runs
        self.run_dropdown = QComboBox()
        self.run_dropdown.currentIndexChanged.connect(self.load_voxel_spacing)
        layout.addWidget(self.run_dropdown)

        # Dropdown for voxel spacing
        self.voxel_spacing_dropdown = QComboBox()
        self.voxel_spacing_dropdown.currentIndexChanged.connect(self.update_voxel_spacing)
        layout.addWidget(self.voxel_spacing_dropdown)

        # Pickable object dropdown
        self.object_dropdown = QComboBox()
        self.object_dropdown.currentIndexChanged.connect(self.load_tomograms)
        layout.addWidget(self.object_dropdown)

        # Tomogram type dropdown
        self.tomogram_type_dropdown = QComboBox()
        layout.addWidget(self.tomogram_type_dropdown)
        
        # Crop size slider label
        self.crop_size_label = QLabel("Crop Radius")
        layout.addWidget(self.crop_size_label)
        
        # Crop size slider
        self.crop_size_slider = QSlider()
        self.crop_size_slider.setMinimum(10)
        self.crop_size_slider.setMaximum(100)
        self.crop_size_slider.setValue(50)
        self.crop_size_slider.setOrientation(Qt.Horizontal)
        self.crop_size_slider.valueChanged.connect(self.update_crop)
        layout.addWidget(self.crop_size_slider)

        # Accept and Reject buttons
        buttons_layout = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.reject_button = QPushButton("Reject")
        self.accept_button.clicked.connect(self.handle_accept)
        self.reject_button.clicked.connect(self.handle_reject)
        buttons_layout.addWidget(self.accept_button)
        buttons_layout.addWidget(self.reject_button)
        layout.addLayout(buttons_layout)

        # Info label
        self.info_label = QLabel("Select a pick to get started")
        layout.addWidget(self.info_label)

        self.setLayout(layout)

        # Setup keybinds
        @self.viewer.bind_key('q')
        def accept_pick(viewer):
            self.handle_accept()

        @self.viewer.bind_key('w')
        def reject_pick(viewer):
            self.handle_reject()

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "JSON Files (*.json)")
        if path:
            self.load_config(path)

    def load_config(self, path=None):
        if path:
            self.root = CopickRootFSSpec.from_file(path)
            self.populate_run_dropdown()
            self.run_dropdown.setCurrentIndex(0)
            self.load_voxel_spacing()

    def populate_run_dropdown(self):
        self.run_dropdown.clear()
        for run in self.root.runs:
            self.run_dropdown.addItem(run.name)

    def get_run(self, name):
        rm = copick.models.CopickRunMeta(name=name)
        return copick.impl.filesystem.CopickRunFSSpec(root=self.root, meta=rm)

    def load_voxel_spacing(self):
        run_name = self.run_dropdown.currentText()
        self.selected_run = self.root.get_run(run_name)

        if self.selected_run:
            voxel_spacings = self.selected_run.voxel_spacings
            self.voxel_spacing_dropdown.clear()
            self.voxel_spacing_dropdown.addItems([str(vs.voxel_size) for vs in voxel_spacings])

            if self.voxel_spacing_dropdown.count() > 0:
                self.voxel_spacing_dropdown.setCurrentIndex(0)
                self.update_voxel_spacing()

    def update_voxel_spacing(self):
        try:
            self.voxel_spacing = float(self.voxel_spacing_dropdown.currentText())
        except ValueError:
            self.voxel_spacing = 10
        self.load_objects()

    def load_objects(self):
        if not self.selected_run:
            return
        
        self.object_dropdown.clear()
        existing_items = []
        for obj in self.root.config.pickable_objects:
            if obj.name not in existing_items:
                self.object_dropdown.addItem(obj.name)
                existing_items.append(obj.name)

        if self.object_dropdown.count() > 0:
            self.object_dropdown.setCurrentIndex(0)
            self.load_tomograms()

    def load_tomograms(self):
        if not self.selected_run:
            return

        object_name = self.object_dropdown.currentText()
        user_id = "albumImportFromCryoETDataPortal"

        self.picks = self.selected_run.get_picks(object_name, user_id=user_id, session_id=self.session_id)
        if not self.picks:
            self.selected_run.new_picks(object_name, session_id=self.session_id, user_id=user_id)
            self.picks = self.selected_run.get_picks(object_name, user_id=user_id, session_id=self.session_id)

        if self.picks:
            self.picks = self.picks[0]

        self.tomogram_type_dropdown.clear()
        available_tomograms = self.selected_run.get_voxel_spacing(self.voxel_spacing).tomograms
        self.tomogram_type_dropdown.addItems([tomo.tomo_type for tomo in available_tomograms])

        self.pick_index = 0
        self.load_next_pick()

    def update_crop(self):
        if self.current_pick:
            self.load_next_pick()
        
    def update_info_label(self):
        if self.current_pick:
            self.info_label.setText(f"Object: {self.picks.pickable_object_name}, Run Name: {self.picks.run.name}, Location: {self.current_pick.location}, Number {self.pick_index} of {len(self.picks.points)}")

    def handle_accept(self):
        if self.current_pick:
            print(f"Accept, Object Type: {self.picks.pickable_object_name}, Run Name: {self.picks.run.name}, Location: {self.current_pick.location}")
            self.remove_current_layers()
            pick_set = self.selected_run.get_picks(self.picks.pickable_object_name, user_id=self.root.user_id, session_id=self.session_id)[0]
            if pick_set.points is None:
                pick_set.points = []
            
            pick_set.points = pick_set.points + [self.current_pick]
            pick_set.store()
            self.pick_index += 1
            self.load_next_pick()

    def handle_reject(self):
        if self.current_pick:
            print(f"Reject, Object Type: {self.picks.pickable_object_name}, Run Name: {self.picks.run.name}, Location: {self.current_pick.location}")
            self.remove_current_layers()
            self.pick_index += 1
            self.load_next_pick()

    def remove_current_layers(self):
        if self.current_layer:
            self.viewer.layers.remove(self.current_layer)
            self.current_layer = None
        for layer in list(self.viewer.layers):
            if "Pick -" in layer.name:
                self.viewer.layers.remove(layer)

    def load_next_pick(self):
        print("Loading next pick")
        self.remove_current_layers()

        if self.picks and self.picks.points and self.pick_index < len(self.picks.points):
            self.current_pick = self.picks.points[self.pick_index]
            obj_name, location = self.picks.pickable_object_name, self.current_pick.location
            crop_size = self.crop_size_slider.value()
            tomogram_type = self.tomogram_type_dropdown.currentText()

            data = zarr.open(self.picks.run.get_voxel_spacing(self.voxel_spacing).get_tomogram(tomogram_type).zarr(), 'r')["0"]
            z = int(location.z / self.voxel_spacing)
            y = int(location.y / self.voxel_spacing)
            x = int(location.x / self.voxel_spacing)
            z_min, z_max = max(0, z - crop_size), min(data.shape[0], z + crop_size)
            y_min, y_max = max(0, y - crop_size), min(data.shape[1], y + crop_size)
            x_min, x_max = max(0, x - crop_size), min(data.shape[2], x + crop_size)
            crop = data[z_min:z_max, y_min:y_max, x_min:x_max]

            self.current_layer = self.viewer.add_image(crop, name=f"{obj_name} - Run: {self.picks.run.name}, Loc: ({x}, {y}, {z})")
            self.points_layer = self.viewer.add_points([[z - z_min, y - y_min, x - x_min]], size=10, name=f"Pick - {obj_name}", visible=True, out_of_slice_display=True)
            self.update_info_label()
        else:
            self.info_label.setText("No more picks to process.")

if __name__ == "__main__":    
    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(viewer)
    viewer.window.add_dock_widget(copick_plugin, area='bottom')
