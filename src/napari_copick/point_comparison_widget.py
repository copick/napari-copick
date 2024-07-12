import numpy as np
from scipy.spatial import cKDTree
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QLineEdit
from qtpy.QtCore import Qt
from copick.impl.filesystem import CopickRootFSSpec
import napari
from napari.utils import DirectLabelColormap
import zarr
import os

voxel_spacing = 10
REFERENCE_SEGMENTATION_NAME = "multifeatures006XGBoost002"
REFERENCE_LAYER_NAME = "Reference Segmentation"

class PickComparisonPlugin(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.root = None
        self.reference_picks = None
        self.candidate_picks = None
        self.distance_threshold = 10  # Default distance threshold
        self.reference_segmentation_dir = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Config loading button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)
        
        # Run name input
        self.run_entry = QLineEdit()
        self.run_entry.setPlaceholderText("Enter run name (e.g. TS_100_1)")
        self.run_entry.setText("TS_100_4")
        layout.addWidget(self.run_entry)

        # User and session selection
        self.reference_user_entry = QLineEdit()
        self.reference_user_entry.setPlaceholderText("Enter reference user ID")
        self.reference_user_entry.setText("curated")
        layout.addWidget(self.reference_user_entry)

        self.reference_session_entry = QLineEdit()
        self.reference_session_entry.setPlaceholderText("Enter reference session ID")
        self.reference_session_entry.setText("19")
        layout.addWidget(self.reference_session_entry)

        self.candidate_user_entry = QLineEdit()
        self.candidate_user_entry.setPlaceholderText("Enter candidate user ID")
        self.candidate_user_entry.setText("candidate")
        layout.addWidget(self.candidate_user_entry)

        self.candidate_session_entry = QLineEdit()
        self.candidate_session_entry.setPlaceholderText("Enter candidate session ID")
        self.candidate_session_entry.setText("cellcanvasCandidates006")
        layout.addWidget(self.candidate_session_entry)

        # Distance threshold input
        self.threshold_entry = QLineEdit()
        self.threshold_entry.setPlaceholderText("Enter distance threshold")
        self.threshold_entry.setText("1000")
        layout.addWidget(self.threshold_entry)

        # Load picks button
        self.load_picks_button = QPushButton("Load Picks")
        self.load_picks_button.clicked.connect(self.load_picks)
        layout.addWidget(self.load_picks_button)

        # Compute metrics button
        self.compute_button = QPushButton("Compute Metrics")
        self.compute_button.clicked.connect(self.compute_metrics)
        layout.addWidget(self.compute_button)

        # Info label
        self.info_label = QLabel("Load config to get started")
        layout.addWidget(self.info_label)

        self.load_config("/Volumes/CZII_A/cellcanvas_tutorial/copick.json")
        
        self.setLayout(layout)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "JSON Files (*.json)")
        if path:
            self.load_config(path)

    def load_config(self, path):
        self.root = CopickRootFSSpec.from_file(path)
        self.info_label.setText("Config loaded. Enter details and load picks.")

    def load_picks(self):
        if not self.root:
            self.info_label.setText("Please load a config file first.")
            return

        run_name = self.run_entry.text()
        reference_user_id = self.reference_user_entry.text()
        reference_session_id = self.reference_session_entry.text()
        candidate_user_id = self.candidate_user_entry.text()
        candidate_session_id = self.candidate_session_entry.text()
        distance_threshold = self.threshold_entry.text()

        if not run_name or not reference_user_id or not reference_session_id or not candidate_user_id or not candidate_session_id or not distance_threshold:
            self.info_label.setText("Please fill in all details.")
            return

        self.distance_threshold = float(distance_threshold)
        run = self.root.get_run(run_name)

        self.reference_picks = self.load_picks_for_user(run, reference_user_id, reference_session_id)
        self.candidate_picks = self.load_picks_for_user(run, candidate_user_id, candidate_session_id)

        self.overlay_picks()
        self.load_reference_layer(run_name)

    def load_picks_for_user(self, run, user_id, session_id):
        picks = run.get_picks(user_id=user_id, session_id=session_id)
        pick_points = {}
        for pick in picks:
            points = pick.points
            object_name = pick.pickable_object_name
            pick_points[object_name] = np.array([[p.location.z, p.location.y, p.location.x] for p in points]) / voxel_spacing
        return pick_points

    def overlay_picks(self):
        self.viewer.layers.clear()
        colormap = self.get_copick_colormap_names()
        if self.reference_picks:
            for obj_name, points in self.reference_picks.items():
                face_color = colormap[obj_name] if obj_name in colormap else 'red'
                self.viewer.add_points(points, size=10, name=f"Reference - {obj_name}", face_color=face_color, out_of_slice_display=True)
        if self.candidate_picks:
            for obj_name, points in self.candidate_picks.items():
                face_color = colormap[obj_name] if obj_name in colormap else 'blue'
                self.viewer.add_points(points, size=10, name=f"Candidate - {obj_name}", face_color=face_color, out_of_slice_display=True)
        
        # Display the tomogram
        tomogram_type = "denoised"  # Change as necessary
        data = zarr.open(self.root.get_run(self.run_entry.text()).get_voxel_spacing(voxel_spacing).get_tomogram(tomogram_type).zarr(), 'r')["0"]
        # self.viewer.add_image(data, name="Tomogram", scale=(voxel_spacing, voxel_spacing, voxel_spacing))
        self.viewer.add_image(data, name="Tomogram")

    def load_multilabel_segmentation(self, segmentation_dir, segmentation_name):
        seg_file = [f for f in os.listdir(segmentation_dir) if f.endswith('.zarr') and segmentation_name in f]
        if seg_file:
            seg_path = os.path.join(segmentation_dir, seg_file[0])
            return zarr.open(seg_path, mode='r')['data']
        return None

    def load_reference_layer(self, run_name):
        self.reference_segmentation_dir = f"/Volumes/CZII_A/cellcanvas_tutorial/ExperimentRuns/{run_name}/Segmentations"
        if self.reference_segmentation_dir:
            segmentation = np.asarray(self.load_multilabel_segmentation(self.reference_segmentation_dir, REFERENCE_SEGMENTATION_NAME)) + 1
            colormap = self.get_copick_colormap()
            # labels_layer = self.viewer.add_labels(segmentation, name=REFERENCE_LAYER_NAME, opacity=0.5, scale=(voxel_spacing, voxel_spacing, voxel_spacing), color=colormap)
            labels_layer = self.viewer.add_labels(segmentation, name=REFERENCE_LAYER_NAME, opacity=0.5)
            labels_layer.colormap = DirectLabelColormap(color_dict=colormap)
            
    def get_copick_colormap(self):
        pickable_objects = self.root.config.pickable_objects
        colormap = {obj.label: np.array(obj.color)/255.0 for obj in pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def get_copick_colormap_names(self):
        pickable_objects = self.root.config.pickable_objects
        colormap = {obj.name: np.array(obj.color)/255.0 for obj in pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def compute_metrics(self):
        if not self.reference_picks or not self.candidate_picks:
            self.info_label.setText("Please load reference and candidate picks.")
            return

        results = {}
        for particle_type in self.reference_picks:
            if particle_type in self.candidate_picks:
                metrics = self.calculate_metrics(self.reference_picks[particle_type], self.candidate_picks[particle_type], self.distance_threshold)
                results[particle_type] = metrics
            else:
                results[particle_type] = self.empty_metrics(len(self.reference_picks[particle_type]))

        self.display_metrics(results)

    def calculate_metrics(self, reference_points, candidate_points, threshold):
        if len(candidate_points) == 0:
            return self.empty_metrics(len(reference_points))

        ref_tree = cKDTree(reference_points)
        distances, indices = ref_tree.query(candidate_points)

        valid_distances = distances[distances != np.inf]
        average_distance = np.mean(valid_distances) if valid_distances.size > 0 else np.inf

        matches_within_threshold = distances <= threshold

        precision = np.sum(matches_within_threshold) / len(candidate_points)
        unique_matched_indices = np.unique(indices[matches_within_threshold])
        recall = len(unique_matched_indices) / len(reference_points)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        num_reference_particles = len(reference_points)
        num_candidate_particles = len(candidate_points)
        num_matched_particles = np.sum(matches_within_threshold)
        percent_matched_reference = (len(unique_matched_indices) / num_reference_particles) * 100
        percent_matched_candidate = (num_matched_particles / num_candidate_particles) * 100

        return {
            'average_distance': average_distance,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_reference_particles': num_reference_particles,
            'num_candidate_particles': num_candidate_particles,
            'num_matched_particles': num_matched_particles,
            'percent_matched_reference': percent_matched_reference,
            'percent_matched_candidate': percent_matched_candidate
        }

    def empty_metrics(self, num_reference_particles):
        return {
            'average_distance': np.inf,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'num_reference_particles': num_reference_particles,
            'num_candidate_particles': 0,
            'num_matched_particles': 0,
            'percent_matched_reference': 0.0,
            'percent_matched_candidate': 0.0
        }

    def display_metrics(self, results):
        for particle_type, metrics in results.items():
            print(f"Particle: {particle_type}")
            print(f"  Average Distance: {metrics['average_distance']}")
            print(f"  Precision: {metrics['precision']}")
            print(f"  Recall: {metrics['recall']}")
            print(f"  F1 Score: {metrics['f1_score']}")
            print(f"  Number of Reference Particles: {metrics['num_reference_particles']}")
            print(f"  Number of Candidate Particles: {metrics['num_candidate_particles']}")
            print(f"  Number of Matched Particles: {metrics['num_matched_particles']}")
            print(f"  Percent Matched (Reference): {metrics['percent_matched_reference']}%")
            print(f"  Percent Matched (Candidate): {metrics['percent_matched_candidate']}%")
        self.info_label.setText("Metrics computed. Check console output.")

if __name__ == "__main__":
    viewer = napari.Viewer()
    plugin = PickComparisonPlugin(viewer)
    viewer.window.add_dock_widget(plugin, area='right')
    # napari.run()
