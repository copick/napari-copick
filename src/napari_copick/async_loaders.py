"""
Async data loaders for napari-copick using napari's threading system.
"""

from typing import Any, Dict

import copick
import numpy as np
import zarr
from napari.qt.threading import thread_worker


@thread_worker
def load_tomogram_worker(tomogram: copick.models.CopickTomogram, resolution_level: int = 0):
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
def load_segmentation_worker(segmentation: copick.models.CopickSegmentation, resolution_level: int = 0):
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
            "name": f"Segmentation: {segmentation.meta.name} ({segmentation.user_id} | {segmentation.session_id}) (Level {resolution_level})",
            "resolution_level": resolution_level,
        }

    except Exception as e:
        error_msg = f"Error loading segmentation: {str(e)}"
        raise ValueError(error_msg) from e


@thread_worker
def expand_run_worker(run: copick.models.CopickRun):
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
def expand_voxel_spacing_worker(voxel_spacing: copick.models.CopickVoxelSpacing):
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


@thread_worker
def save_segmentation_worker(save_params: Dict[str, Any]):
    """Save segmentation data in background thread with scaling and saving operations."""
    try:
        layer = save_params["layer"]
        run = save_params["run"]
        voxel_spacing = save_params["voxel_spacing"]
        object_name = save_params["object_name"]
        session_id = save_params["session_id"]
        user_id = save_params["user_id"]
        exist_ok = save_params.get("exist_ok", False)
        split_instances = save_params.get("split_instances", False)
        convert_to_binary = save_params.get("convert_to_binary", False)

        yield f"Processing segmentation '{object_name}' for run '{run.name}'..."

        yield "Getting segmentation data from layer..."

        # Get the segmentation data
        seg_data = layer.data

        yield "Determining target shape from tomogram..."

        # Import the utility functions
        from napari_copick.save_utils import get_tomogram_shape_at_level_0, scale_segmentation_to_target_shape

        # Handle scaling to tomogram zarr layer '0' dimensions if needed
        target_shape = get_tomogram_shape_at_level_0(run, voxel_spacing)

        if seg_data.shape != target_shape:
            yield f"Scaling segmentation from {seg_data.shape} to {target_shape}..."
            seg_data = scale_segmentation_to_target_shape(seg_data, target_shape)
        else:
            yield "Segmentation shape matches target, no scaling needed..."

        yield "Converting data to uint8 format..."
        seg_data = seg_data.astype(np.uint8)

        # Handle different processing modes
        if split_instances:
            yield "Splitting segmentation into binary instances..."
            from napari_copick.save_utils import split_segmentation_into_instances

            instances = split_segmentation_into_instances(seg_data, session_id)

            if not instances:
                raise ValueError("No valid instances found in segmentation data")

            saved_segmentations = []

            for i, instance in enumerate(instances):
                yield f"Saving instance {i+1}/{len(instances)} (label {instance['label']}, session '{instance['session_id']}')"

                # Create new segmentation for this instance
                segmentation = run.new_segmentation(
                    voxel_size=voxel_spacing.voxel_size,
                    name=object_name,
                    session_id=instance["session_id"],
                    user_id=user_id,
                    is_multilabel=False,  # Binary segmentation
                    exist_ok=exist_ok,
                )

                # Save the binary instance data
                segmentation.from_numpy(instance["data"], levels=1, dtype=np.uint8)
                saved_segmentations.append(segmentation)

            yield f"Successfully saved {len(instances)} binary instances for '{object_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved {len(instances)} binary instances for '{object_name}' to run '{run.name}'",
                "segmentations": saved_segmentations,
                "object_name": object_name,
                "run_name": run.name,
                "split_instances": True,
                "instance_count": len(instances),
            }
        elif convert_to_binary:
            yield "Converting segmentation to binary..."
            from napari_copick.save_utils import convert_segmentation_to_binary

            # Convert to binary (all non-zero labels become 1)
            binary_data = convert_segmentation_to_binary(seg_data)

            yield "Creating binary segmentation..."

            # Create new segmentation
            segmentation = run.new_segmentation(
                voxel_size=voxel_spacing.voxel_size,
                name=object_name,
                session_id=session_id,
                user_id=user_id,
                is_multilabel=False,  # Binary segmentation
                exist_ok=exist_ok,
            )

            yield "Saving binary segmentation to copick using from_numpy method..."

            # Save using copick's from_numpy method which follows copick conventions
            segmentation.from_numpy(binary_data, levels=1, dtype=np.uint8)

            yield f"Successfully saved binary segmentation '{object_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved binary segmentation '{object_name}' to run '{run.name}'",
                "segmentation": segmentation,
                "object_name": object_name,
                "run_name": run.name,
                "convert_to_binary": True,
            }
        else:
            yield "Creating single segmentation..."

            # Create new segmentation
            segmentation = run.new_segmentation(
                voxel_size=voxel_spacing.voxel_size,
                name=object_name,
                session_id=session_id,
                user_id=user_id,
                is_multilabel=False,  # Single label segmentation
                exist_ok=exist_ok,
            )

            yield "Saving segmentation to copick using from_numpy method..."

            # Save using copick's from_numpy method which follows copick conventions
            segmentation.from_numpy(seg_data, levels=1, dtype=np.uint8)

            yield f"Successfully saved segmentation '{object_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved segmentation '{object_name}' to run '{run.name}'",
                "segmentation": segmentation,
                "object_name": object_name,
                "run_name": run.name,
            }

    except Exception as e:
        error_msg = f"Error saving segmentation: {str(e)}"
        raise ValueError(error_msg) from e
