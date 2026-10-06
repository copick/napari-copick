"""Utility functions for saving segmentations and picks to copick."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import copick
import numpy as np
from copick_shared_ui.core.types import segmentation_type_flags

from napari_copick.pick_layers import TRANSFORMS_KEY, layer_to_points
from napari_copick.storage import open_multiscale_level

logger = logging.getLogger(__name__)


def get_runs_from_open_layers(viewer: Any) -> Dict[str, copick.models.CopickRun]:
    """Get unique runs from currently open image layers with metadata."""
    runs = {}
    for layer in viewer.layers:
        if hasattr(layer, "metadata") and "copick_run" in layer.metadata:
            run = layer.metadata["copick_run"]
            if run.name not in runs:
                runs[run.name] = run
    return runs


def save_segmentation_to_copick(
    save_params: Dict[str, Any],
    info_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Save a segmentation layer to copick."""
    layer = save_params["layer"]
    run = save_params["run"]
    voxel_spacing = save_params["voxel_spacing"]
    session_id = save_params["session_id"]
    user_id = save_params["user_id"]
    is_multilabel = save_params.get("is_multilabel", False)
    seg_type = save_params.get("segmentation_type", "multilabel" if is_multilabel else "binary")

    # Use segmentation_name for multilabel, object_name for single-label
    segmentation_name = save_params.get("segmentation_name", save_params.get("object_name"))

    try:
        # Create new segmentation
        segmentation = run.new_segmentation(
            voxel_size=voxel_spacing.voxel_size,
            name=segmentation_name,
            session_id=session_id,
            user_id=user_id,
            **segmentation_type_flags(seg_type),
        )

        # Get the segmentation data
        seg_data = layer.data

        # Handle scaling to tomogram zarr layer '0' dimensions if needed
        target_shape = get_tomogram_shape_at_level_0(run, voxel_spacing)
        if seg_data.shape != target_shape:
            seg_data = scale_segmentation_to_target_shape(seg_data, target_shape)

        validate_segmentation_values(seg_data, seg_type)

        # copick chooses the smallest unsigned dtype that holds every value (uint16+ for instance IDs)
        segmentation.from_numpy(seg_data, levels=1)

        if info_callback:
            info_callback(f"Saved segmentation '{segmentation_name}' to run '{run.name}'")

        return True

    except Exception as e:
        logger.exception(f"Error saving segmentation: {str(e)}")
        if info_callback:
            info_callback(f"Error saving segmentation: {str(e)}")
        return False


def save_picks_to_copick(save_params: Dict[str, Any], info_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Save a points layer to copick."""
    layer = save_params["layer"]
    run = save_params["run"]
    object_name = save_params["object_name"]
    session_id = save_params["session_id"]
    user_id = save_params["user_id"]
    exist_ok = save_params.get("exist_ok", False)

    try:
        # Create new picks
        picks = run.new_picks(
            object_name=object_name,
            session_id=session_id,
            user_id=user_id,
            exist_ok=exist_ok,
        )

        # Loaded points keep their transform (shift included), instance id and score; new points get the identity.
        scale = getattr(layer, "scale", (1.0, 1.0, 1.0))
        points = layer_to_points(
            layer.data,
            scale,
            getattr(layer, "features", None),
            layer.metadata.get(TRANSFORMS_KEY) if hasattr(layer, "metadata") else None,
        )
        picks.points = points
        picks.store()

        if info_callback:
            info_callback(f"Saved {len(points)} picks for '{object_name}' to run '{run.name}'")

        return True

    except Exception as e:
        logger.exception(f"Error saving picks: {str(e)}")
        if info_callback:
            info_callback(f"Error saving picks: {str(e)}")
        return False


def get_tomogram_shape_at_level_0(
    run: copick.models.CopickRun,
    voxel_spacing: copick.models.CopickVoxelSpacing,
) -> Tuple[int, int, int]:
    """Get the shape of the tomogram at resolution level 0."""
    try:
        # Find a tomogram at this voxel spacing
        tomograms = voxel_spacing.tomograms
        if not tomograms:
            raise ValueError("No tomograms found at this voxel spacing")

        first_tomogram = tomograms[0]
        level = open_multiscale_level(first_tomogram, 0)
        return tuple(level.array.shape)

    except Exception as e:
        logger.exception(f"Error getting tomogram shape: {str(e)}")
        # Return a default shape if we can't determine it
        raise RuntimeError(f"Error getting tomogram shape: {str(e)}") from e


def scale_segmentation_to_target_shape(seg_data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Scale segmentation data to target shape using nearest neighbor interpolation."""
    try:
        from scipy.ndimage import zoom

        # Calculate zoom factors for each dimension
        zoom_factors = [target_shape[i] / seg_data.shape[i] for i in range(3)]

        # Use nearest neighbor interpolation to preserve labels
        scaled_data = zoom(seg_data, zoom_factors, order=0)

        # Ensure the output shape is exactly what we want
        if scaled_data.shape != target_shape:
            # Crop or pad if there are slight differences due to rounding
            scaled_data = crop_or_pad_to_shape(scaled_data, target_shape)

        return scaled_data.astype(seg_data.dtype, copy=False)

    except ImportError:
        logger.warning("scipy not available, saving segmentation at original resolution")
        return seg_data
    except Exception as e:
        logger.exception(f"Error scaling segmentation: {str(e)}")
        return seg_data


def crop_or_pad_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Crop or pad data to match target shape."""
    result = np.zeros(target_shape, dtype=data.dtype)

    # Calculate slices for copying
    slices = []
    for i in range(3):
        if data.shape[i] <= target_shape[i]:
            # Pad case - center the data
            start = (target_shape[i] - data.shape[i]) // 2
            slices.append(slice(start, start + data.shape[i]))
        else:
            # Crop case - take from center
            start = (data.shape[i] - target_shape[i]) // 2
            slices.append(slice(start, start + target_shape[i]))

    # Copy data
    if all(data.shape[i] <= target_shape[i] for i in range(3)):
        # Padding case
        result[tuple(slices)] = data
    else:
        # Cropping case
        result = data[tuple(slices)]

    return result


def validate_segmentation_values(seg_data: np.ndarray, segmentation_type: str, converting: bool = False) -> None:
    """Refuse data that the chosen segmentation type cannot hold (instead of letting copick narrow or reject it later).

    Args:
        seg_data: The label volume.
        segmentation_type: ``"binary"``, ``"multilabel"`` or ``"instance"``.
        converting: The data will be converted (binary conversion or split), so any labels are fine.

    Raises:
        ValueError: For negative or non-integral values, or a binary volume with labels other than 0 and 1.
    """
    data = np.asarray(seg_data)
    if data.size == 0:
        return
    if data.dtype.kind == "f":
        if not np.all(np.isfinite(data)) or not np.all(data == np.round(data)):
            raise ValueError("Segmentation values must be whole numbers.")
    elif data.dtype.kind not in "uib":
        raise ValueError(f"Segmentation dtype {data.dtype} is not an integer type.")
    if data.min() < 0:
        raise ValueError("Segmentation values must not be negative.")
    if segmentation_type == "binary" and not converting and data.max() > 1:
        raise ValueError(
            f"A binary segmentation holds only 0 and 1, but the layer has labels up to {int(data.max())}. "
            "Enable 'Convert to binary', or save it as an instance or multilabel segmentation.",
        )


def split_segmentation_into_instances(seg_data: np.ndarray, session_id: str) -> List[Dict[str, Any]]:
    """Split a multi-class segmentation into binary instances with unique session IDs.

    Args:
        seg_data: Multi-class segmentation array
        session_id: Base session ID to append indices to

    Returns:
        List of dictionaries containing instance data and session IDs
    """
    # Find unique labels (excluding background/0)
    unique_labels = np.unique(seg_data)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    if len(unique_labels) == 0:
        logger.warning("No non-zero labels found in segmentation")
        return []

    instances = []
    for i, label in enumerate(unique_labels):
        # Create binary mask for this label
        binary_mask = (seg_data == label).astype(np.uint8)

        # Generate session ID with suffix
        instance_session_id = f"{session_id}-{i}"

        instances.append(
            {
                "data": binary_mask,
                "session_id": instance_session_id,
                "label": int(label),
            },
        )

        logger.info(f"Created instance {i} for label {label} with session ID '{instance_session_id}'")

    return instances


def convert_segmentation_to_binary(seg_data: np.ndarray) -> np.ndarray:
    """Convert a multi-class segmentation to binary by setting all non-zero labels to 1.

    Args:
        seg_data: Multi-class segmentation array

    Returns:
        Binary segmentation array where all non-zero values are set to 1
    """
    binary_data = (seg_data > 0).astype(np.uint8)

    unique_labels = np.unique(seg_data)
    non_zero_labels = unique_labels[unique_labels > 0]

    logger.info(
        f"Converting segmentation to binary: found {len(non_zero_labels)} non-zero labels {list(non_zero_labels)}",
    )

    return binary_data
