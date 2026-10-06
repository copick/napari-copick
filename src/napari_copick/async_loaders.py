"""
Async data loaders for napari-copick using napari's threading system.
"""

from typing import Any, Dict

import copick
import numpy as np
from copick_shared_ui.core.types import segmentation_type_flags, segmentation_type_of
from copick_shared_ui.util.instances import label_bboxes, label_counts, panoptic_segments
from napari.qt.threading import thread_worker

from napari_copick.storage import open_multiscale_level, spatial_scale


@thread_worker
def load_tomogram_worker(tomogram: copick.models.CopickTomogram, resolution_level: int = 0):
    """Load tomogram data in background thread using napari's threading system."""
    try:
        yield f"Opening zarr group for {tomogram.meta.tomo_type}..."
        level = open_multiscale_level(tomogram, resolution_level)
        resolution_level = level.level

        yield f"Loading resolution level {resolution_level}..."
        voxel_size = spatial_scale(level, tomogram.voxel_spacing.meta.voxel_size)

        # Actually load the data (not lazy!)
        yield "Loading image data into memory..."
        loaded_data = np.array(level.array)

        # Return the final result with pre-loaded data
        return {
            "tomogram": tomogram,
            "data": loaded_data,
            "voxel_size": voxel_size,
            "name": f"[{tomogram.voxel_spacing.run.name}] Tomogram: {tomogram.meta.tomo_type} (Level {resolution_level})",
            "resolution_level": resolution_level,
        }

    except Exception:
        raise


@thread_worker
def load_segmentation_worker(segmentation: copick.models.CopickSegmentation, resolution_level: int = 0):
    """Load segmentation data in background thread using napari's threading system."""
    try:
        seg_type = segmentation_type_of(segmentation)
        yield f"Opening zarr group for {segmentation.meta.name}..."
        # OME metadata selects the level; spatial_scale reads the z/y/x axes, so panoptic (c, z, y, x) works too.
        level = open_multiscale_level(segmentation, resolution_level)
        resolution_level = level.level

        yield f"Loading segmentation data from level {resolution_level}..."
        voxel_size = spatial_scale(level, segmentation.meta.voxel_size)

        # Actually load the data (not lazy!)
        yield "Loading segmentation data into memory..."
        loaded_data = np.array(level.array)

        result = {
            "segmentation": segmentation,
            "segmentation_type": seg_type,
            "data": loaded_data,
            "voxel_size": voxel_size,
            "name": (
                f"[{segmentation.run.name}] Segmentation: {segmentation.meta.name} "
                f"({segmentation.user_id} | {segmentation.session_id}) (Level {resolution_level})"
            ),
            "resolution_level": resolution_level,
        }

        if seg_type == "instance":
            yield "Counting instances..."
            result["counts"] = label_counts(loaded_data)
            result["bboxes"] = label_bboxes(loaded_data)
            result["name"] = result["name"].replace("Segmentation:", "Instances:", 1)
        elif seg_type == "panoptic":
            # (2, Z, Y, X): channel 0 = object label, channel 1 = instance ID within the object
            if loaded_data.ndim != 4 or loaded_data.shape[0] != 2:
                raise ValueError(f"Panoptic segmentation has shape {loaded_data.shape}, expected (2, Z, Y, X)")
            yield "Indexing panoptic segments..."
            objects = {
                o.label: (o.name, tuple(np.asarray(o.color or (128, 128, 128, 255), dtype=float) / 255.0))
                for o in segmentation.run.root.config.pickable_objects
            }
            segments, rows = panoptic_segments(loaded_data[0], loaded_data[1], objects)
            result["data"] = loaded_data[0]
            result["segments"] = segments
            result["rows"] = rows
            result["bboxes"] = label_bboxes(segments)
            result["name"] = result["name"].replace("Segmentation:", "Panoptic:", 1)
        return result

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

        # Include voxel spacings implied by segmentations (no tomogram at that size)
        segmentations = run.segmentations
        existing_vs = {vs.voxel_size for vs in voxel_spacings}
        seg_vs = {s.voxel_size for s in segmentations}
        missing_vs = seg_vs - existing_vs
        if missing_vs:
            clz, meta_clz = run._voxel_spacing_factory()
            for size in sorted(missing_vs):
                vm = meta_clz(voxel_size=size)
                vs = clz(run=run, meta=vm)
                voxel_spacings.append(vs)

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

        # Traced filaments (copick versions with the Filaments entity)
        filaments = []
        if hasattr(run, "filaments"):
            yield f"Loading filaments for {run.meta.name}..."
            filaments = list(run.filaments)

        return {"run": run, "voxel_spacings": voxel_spacings, "picks_data": user_dict, "filaments": filaments}

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
        session_id = save_params["session_id"]
        user_id = save_params["user_id"]
        exist_ok = save_params.get("exist_ok", False)
        split_instances = save_params.get("split_instances", False)
        convert_to_binary = save_params.get("convert_to_binary", False)
        is_multilabel = save_params.get("is_multilabel", False)
        seg_type = save_params.get("segmentation_type", "multilabel" if is_multilabel else "binary")

        # Use segmentation_name for multilabel, object_name for single-label
        segmentation_name = save_params.get("segmentation_name", save_params.get("object_name"))

        yield f"Processing segmentation '{segmentation_name}' for run '{run.name}'..."

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

        from napari_copick.save_utils import validate_segmentation_values

        validate_segmentation_values(seg_data, seg_type, convert_to_binary or split_instances)

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
                    name=segmentation_name,
                    session_id=instance["session_id"],
                    user_id=user_id,
                    exist_ok=exist_ok,
                    **segmentation_type_flags("binary"),
                )

                # Save the binary instance data
                segmentation.from_numpy(instance["data"], levels=1)
                saved_segmentations.append(segmentation)

            yield f"Successfully saved {len(instances)} binary instances for '{segmentation_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved {len(instances)} binary instances for '{segmentation_name}' to run '{run.name}'",
                "segmentations": saved_segmentations,
                "object_name": segmentation_name,
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
                name=segmentation_name,
                session_id=session_id,
                user_id=user_id,
                exist_ok=exist_ok,
                **segmentation_type_flags("binary"),
            )

            yield "Saving binary segmentation to copick using from_numpy method..."

            # Save using copick's from_numpy method which follows copick conventions
            segmentation.from_numpy(binary_data, levels=1)

            yield f"Successfully saved binary segmentation '{segmentation_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved binary segmentation '{segmentation_name}' to run '{run.name}'",
                "segmentation": segmentation,
                "object_name": segmentation_name,
                "run_name": run.name,
                "convert_to_binary": True,
            }
        else:
            # Normal save mode: binary, multilabel or instance
            yield f"Creating {seg_type} segmentation..."

            # Create new segmentation
            segmentation = run.new_segmentation(
                voxel_size=voxel_spacing.voxel_size,
                name=segmentation_name,
                session_id=session_id,
                user_id=user_id,
                exist_ok=exist_ok,
                **segmentation_type_flags(seg_type),
            )

            yield "Saving segmentation to copick using from_numpy method..."

            # copick chooses the smallest unsigned dtype that holds every value (uint16+ for instance IDs)
            segmentation.from_numpy(seg_data, levels=1)

            yield f"Successfully saved {seg_type} segmentation '{segmentation_name}' to run '{run.name}'"

            return {
                "success": True,
                "message": f"Saved {seg_type} segmentation '{segmentation_name}' to run '{run.name}'",
                "segmentation": segmentation,
                "object_name": segmentation_name,
                "run_name": run.name,
                "is_multilabel": seg_type == "multilabel",
                "segmentation_type": seg_type,
            }

    except Exception as e:
        error_msg = f"Error saving segmentation: {str(e)}"
        raise ValueError(error_msg) from e


@thread_worker
def load_filaments_worker(filaments: Any, step: float = None):
    """Read a filament set and open it as an editing session."""
    try:
        yield f"Loading filaments {filaments.pickable_object_name} ({filaments.user_id} | {filaments.session_id})..."
        from copick_shared_ui.util.filament_session import FilamentEditSession

        _ = filaments.filaments  # read the file off the UI thread
        session = FilamentEditSession.from_filaments(filaments, step=step or filaments.voxel_spacing or 10.0)
        return {"filaments": filaments, "session": session}
    except Exception as e:
        raise ValueError(f"Error loading filaments: {e}") from e


@thread_worker
def save_filaments_worker(save_params: Dict[str, Any]):
    """Store a filament editing session (and optionally picks sampled along the filaments)."""
    try:
        session = save_params["session"]
        n = len(session.to_list())
        yield f"Saving {n} filaments for '{session.object_name}'..."
        result = session.save(
            user_id=save_params["user_id"],
            session_id=save_params["session_id"],
            voxel_spacing=save_params.get("voxel_spacing"),
            pick_spacing=save_params.get("pick_spacing"),
            exist_ok=save_params.get("exist_ok", True),
        )
        message = f"Saved {result['n_filaments']} filaments for '{session.object_name}'"
        if result["n_picks"]:
            message += f" and {result['n_picks']} sampled picks"
        return {"success": True, "message": message, "run_name": session.run.name, **result}
    except Exception as e:
        raise ValueError(f"Error saving filaments: {e}") from e
