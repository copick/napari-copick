"""
"""


import copick
import json
import zarr
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QSlider, QFileDialog, QLineEdit
from qtpy.QtCore import Qt
from copick.impl.filesystem import CopickRootFSSpec

class CopickPlugin(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.root = None
        self.current_layer = None
        self.current_pick = None
        self.pick_index = 0
        self.picks = []
        self.voxel_spacing = 10
        self.setup_ui()
        self.session_id = "17"

    def setup_ui(self):
        layout = QVBoxLayout()

        # Config loading button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # Run selection text input
        self.run_entry = QLineEdit()
        self.run_entry.setPlaceholderText("Enter run name (e.g. TS_100_1)")
        layout.addWidget(self.run_entry)

        # Button to apply run names
        self.apply_button = QPushButton("Start copicking!")
        self.apply_button.clicked.connect(self.apply_run_names)
        layout.addWidget(self.apply_button)
        
        # Pickable object dropdown
        self.object_dropdown = QComboBox()
        self.object_dropdown.currentIndexChanged.connect(self.load_objects)
        layout.addWidget(self.object_dropdown)

        # Tomogram type dropdown
        self.tomogram_type_dropdown = QComboBox()
        self.tomogram_type_dropdown.addItems(["denoised", "ctfdeconvolved", "isonetcorrected", "wbp"])
        self.tomogram_type_dropdown.currentIndexChanged.connect(self.update_crop)
        layout.addWidget(self.tomogram_type_dropdown)
        
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
            # Initialize pickable object dropdown after config is loaded
            self.object_dropdown.clear()
            for obj in self.root.pickable_objects:
                self.object_dropdown.addItem(obj.name)            

    def get_run(self, name):
        rm = copick.models.CopickRunMeta(name=name)
        return copick.impl.filesystem.CopickRunFSSpec(root=self.root, meta=rm)
            
    def apply_run_names(self):
        run_names = self.run_entry.text().split(',')
        # TODO Utz :D
        self.root._runs = [self.get_run(name) for name in run_names]
        self.load_runs(run_names)

    def load_runs(self, run_names):
        self.object_dropdown.clear()
        existing_items = []
        for run_name in run_names:
            run_name = run_name.strip()
            # TODO this is very bad it means this will only work with run
            self.selected_run = self.root.get_run(run_name)
            if self.selected_run:
                for obj in self.root.config.pickable_objects:
                    if obj.name not in existing_items:
                        self.object_dropdown.addItem(obj.name)
                        existing_items.append(obj.name)
                    # Check for picks and initialize as needed
                    picks = self.selected_run.get_picks(obj.name, user_id=self.root.user_id, session_id=self.session_id)
                    if not picks:
                        picks = self.selected_run.new_picks(obj.name, self.session_id, self.root.user_id)
                        picks = self.selected_run.get_picks(obj.name, user_id=self.root.user_id, session_id=self.session_id)
        self.pick_index = 0
        self.load_objects()
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
            # Save pick
            # TODO hard coded again
            pick_set = self.selected_run.get_picks(self.picks.pickable_object_name, user_id=self.root.user_id, session_id=self.session_id)[0]
            if pick_set.points is None:
                pick_set.points = []
            
            pick_set.points = pick_set.points + [self.current_pick]
            pick_set.store()
            # Move to next
            self.pick_index += 1
            self.load_next_pick()

    def handle_reject(self):
        if self.current_pick:
            print(f"Reject, Object Type: {self.picks.pickable_object_name}, Run Name: {self.picks.run.name}, Location: {self.current_pick.location}")
            self.remove_current_layers()
            # Move to next
            self.pick_index += 1
            self.load_next_pick()

    def remove_current_layers(self):
        if self.current_layer:
            self.viewer.layers.remove(self.current_layer)
            self.current_layer = None
        for layer in list(self.viewer.layers):
            if "Pick -" in layer.name:
                self.viewer.layers.remove(layer)

    def load_objects(self):
        # TODO this is hard coded to use prepicks and the first sessionid in liset
        self.picks = self.selected_run.get_picks(self.object_dropdown.currentText(), user_id="prepick")[0]
        self.pick_index = 0
        self.load_next_pick()

    def load_next_pick(self):
        print("Loading next pick")
        self.remove_current_layers()        
        if self.pick_index < len(self.picks.points):
            self.current_pick = self.picks.points[self.pick_index]
            obj_name, location = self.picks.pickable_object_name, self.current_pick.location
            crop_size = self.crop_size_slider.value()
            tomogram_type = self.tomogram_type_dropdown.currentText()            

            # TODO use LRUCacheStore to help with crop resizing
            data = zarr.open(self.picks.run.get_voxel_spacing(self.voxel_spacing).get_tomogram(tomogram_type).zarr(), 'r')["0"]
            # Scale points
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
