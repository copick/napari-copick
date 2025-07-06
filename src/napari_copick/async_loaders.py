"""
Async data loaders for napari-copick using napari's threading system.
"""

import numpy as np
import zarr
from napari.qt.threading import thread_worker


@thread_worker
def load_tomogram_worker(tomogram, resolution_level=0):
    """Load tomogram data in background thread using napari's threading system."""
    try:
        zarr_path = tomogram.zarr()

        yield f"Opening zarr group for {tomogram.meta.tomo_type}..."
        zarr_group = zarr.open(zarr_path, "r")

        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]  # noqa: SIM118
        scale_levels.sort(key=int)

        if not scale_levels:
            raise ValueError(f"No scale levels found in tomogram: {tomogram.meta.tomo_type}")

        # Validate resolution level
        if resolution_level >= len(scale_levels):
            resolution_level = len(scale_levels) - 1

        # Load only the selected resolution level
        selected_level = scale_levels[resolution_level]

        yield f"Loading resolution level {resolution_level}..."
        array = zarr_group[selected_level]

        # Calculate voxel size from metadata, adjusting for resolution level
        base_voxel_size = tomogram.voxel_spacing.meta.voxel_size
        # Each resolution level is typically 2x binned
        scale_factor = 2**resolution_level
        voxel_size = [base_voxel_size * scale_factor] * 3

        # Actually load the data (not lazy!)
        yield "Loading image data into memory..."
        loaded_data = np.array(array)

        # Return the final result with pre-loaded data
        return {
            "tomogram": tomogram,
            "data": loaded_data,
            "voxel_size": voxel_size,
            "name": f"Tomogram: {tomogram.meta.tomo_type} (Level {resolution_level})",
            "resolution_level": resolution_level,
        }

    except Exception:
        raise


@thread_worker
def load_segmentation_worker(segmentation, resolution_level=0):
    """Load segmentation data in background thread using napari's threading system."""
    try:
        zarr_path = segmentation.zarr()

        yield f"Opening zarr group for {segmentation.meta.name}..."
        zarr_group = zarr.open(zarr_path, "r+")

        # Try to find data in zarr group
        if "data" in zarr_group:
            data_key = "data"
        elif "0" in zarr_group:
            # Handle multiscale segmentations
            scale_levels = [key for key in zarr_group.keys() if key.isdigit()]  # noqa: SIM118
            scale_levels.sort(key=int)

            # Validate resolution level for multiscale
            if resolution_level >= len(scale_levels):
                resolution_level = len(scale_levels) - 1

            data_key = scale_levels[resolution_level]
        else:
            # Fallback to first available key
            data_key = list(zarr_group.keys())[0]

        yield f"Loading segmentation data from level {resolution_level}..."
        array = zarr_group[data_key]

        # Calculate voxel size from metadata, adjusting for resolution level if multiscale
        base_voxel_size = segmentation.meta.voxel_size
        if data_key.isdigit():
            # Multiscale segmentation
            scale_factor = 2**resolution_level
            voxel_size = [base_voxel_size * scale_factor] * 3
        else:
            # Single scale segmentation
            voxel_size = [base_voxel_size] * 3
            resolution_level = 0  # Reset to 0 for display

        # Actually load the data (not lazy!)
        yield "Loading segmentation data into memory..."
        loaded_data = np.array(array)

        # Return the final result with pre-loaded data
        return {
            "segmentation": segmentation,
            "data": loaded_data,
            "voxel_size": voxel_size,
            "name": f"Segmentation: {segmentation.meta.name} (Level {resolution_level})",
            "resolution_level": resolution_level,
        }

    except Exception as e:
        error_msg = f"Error loading segmentation: {str(e)}"
        raise ValueError(error_msg) from e


@thread_worker
def expand_run_worker(run):
    """Expand a run in the tree by gathering voxel spacings and picks data."""
    try:
        yield f"Loading voxel spacings for {run.meta.name}..."

        # Get voxel spacings (usually fast)
        voxel_spacings = list(run.voxel_spacings)

        yield f"Loading picks for {run.meta.name}..."

        # Get picks (can be slow)
        picks = run.picks

        # Organize picks by user_id and session_id
        yield "Organizing picks data..."
        user_dict = {}
        for pick in picks:
            if pick.meta.user_id not in user_dict:
                user_dict[pick.meta.user_id] = {}
            if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                user_dict[pick.meta.user_id][pick.meta.session_id] = []
            user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)

        return {"run": run, "voxel_spacings": voxel_spacings, "picks_data": user_dict}

    except Exception as e:
        error_msg = f"Error expanding run: {str(e)}"
        raise ValueError(error_msg) from e


@thread_worker
def expand_voxel_spacing_worker(voxel_spacing):
    """Expand a voxel spacing in the tree by gathering tomograms and segmentations."""
    try:
        yield f"Loading tomograms for voxel size {voxel_spacing.meta.voxel_size}..."

        # Get tomograms (usually fast)
        tomograms = list(voxel_spacing.tomograms)
        yield f"Loading segmentations for voxel size {voxel_spacing.meta.voxel_size}..."

        # Get segmentations (can be slow)
        segmentations = voxel_spacing.run.get_segmentations(voxel_size=voxel_spacing.meta.voxel_size)

        return {"voxel_spacing": voxel_spacing, "tomograms": tomograms, "segmentations": segmentations}

    except Exception as e:
        error_msg = f"Error expanding voxel spacing: {str(e)}"
        raise ValueError(error_msg) from e
