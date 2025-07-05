"""
Async data loaders for napari-copick using napari's threading system.
"""

import logging
import numpy as np
import zarr
from napari.qt.threading import thread_worker


@thread_worker
def load_tomogram_worker(tomogram, resolution_level=0):
    """Load tomogram data in background thread using napari's threading system."""
    logger = logging.getLogger(f"TomogramLoadWorker_{id(tomogram)}")
    
    try:
        logger.info(f"Starting tomogram loading for {tomogram.meta.tomo_type} at resolution level {resolution_level}")
        
        zarr_path = tomogram.zarr()
        logger.info(f"Zarr path: {zarr_path}")
        
        yield f"Opening zarr group for {tomogram.meta.tomo_type}..."
        zarr_group = zarr.open(zarr_path, "r")
        logger.info(f"Zarr group opened successfully. Keys: {list(zarr_group.keys())}")
        
        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]  # noqa: SIM118
        scale_levels.sort(key=int)
        logger.info(f"Found {len(scale_levels)} scale levels: {scale_levels}")
        
        if not scale_levels:
            error_msg = f"No scale levels found in tomogram: {tomogram.meta.tomo_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Validate resolution level
        if resolution_level >= len(scale_levels):
            logger.warning(f"Requested resolution level {resolution_level} not available, using highest available: {len(scale_levels)-1}")
            resolution_level = len(scale_levels) - 1
            
        # Load only the selected resolution level
        selected_level = scale_levels[resolution_level]
        logger.info(f"Loading resolution level {resolution_level} (scale level {selected_level})")
        
        yield f"Loading resolution level {resolution_level}..."
        array = zarr_group[selected_level]
        logger.info(f"Array shape: {array.shape}, chunks: {array.chunks}")
        
        # Calculate voxel size from metadata, adjusting for resolution level
        base_voxel_size = tomogram.voxel_spacing.meta.voxel_size
        # Each resolution level is typically 2x binned
        scale_factor = 2 ** resolution_level
        voxel_size = [base_voxel_size * scale_factor] * 3
        logger.info(f"Voxel size for level {resolution_level}: {voxel_size}")
        
        # Actually load the data (not lazy!)
        yield f"Loading image data into memory..."
        logger.info("Loading array data into memory...")
        loaded_data = np.array(array)
        logger.info(f"Data loaded successfully. Final shape: {loaded_data.shape}")
            
        logger.info("Returning final result...")
        
        # Return the final result with pre-loaded data
        return {
            'tomogram': tomogram,
            'data': loaded_data,
            'voxel_size': voxel_size,
            'name': f"Tomogram: {tomogram.meta.tomo_type} (Level {resolution_level})",
            'resolution_level': resolution_level
        }
                         
    except Exception as e:
        error_msg = f"Error loading tomogram: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@thread_worker
def load_segmentation_worker(segmentation, resolution_level=0):
    """Load segmentation data in background thread using napari's threading system."""
    logger = logging.getLogger(f"SegmentationLoadWorker_{id(segmentation)}")
    
    try:
        logger.info(f"Starting segmentation loading for {segmentation.meta.name} at resolution level {resolution_level}")
        
        zarr_path = segmentation.zarr()
        logger.info(f"Zarr path: {zarr_path}")
        
        yield f"Opening zarr group for {segmentation.meta.name}..."
        zarr_group = zarr.open(zarr_path, "r+")
        logger.info(f"Zarr group opened successfully. Keys: {list(zarr_group.keys())}")
        
        # Try to find data in zarr group
        if "data" in zarr_group:
            data_key = "data"
        elif "0" in zarr_group:
            # Handle multiscale segmentations
            scale_levels = [key for key in zarr_group.keys() if key.isdigit()]
            scale_levels.sort(key=int)
            
            # Validate resolution level for multiscale
            if resolution_level >= len(scale_levels):
                logger.warning(f"Requested resolution level {resolution_level} not available, using highest available: {len(scale_levels)-1}")
                resolution_level = len(scale_levels) - 1
                
            data_key = scale_levels[resolution_level]
            logger.info(f"Using multiscale level {resolution_level} (key: {data_key})")
        else:
            # Fallback to first available key
            data_key = list(zarr_group.keys())[0]
            logger.info(f"Using fallback data key: {data_key}")
        
        yield f"Loading segmentation data from level {resolution_level}..."
        array = zarr_group[data_key]
        logger.info(f"Array shape: {array.shape}, chunks: {getattr(array, 'chunks', 'N/A')}")
        
        # Calculate voxel size from metadata, adjusting for resolution level if multiscale
        base_voxel_size = segmentation.meta.voxel_size
        if data_key.isdigit():
            # Multiscale segmentation
            scale_factor = 2 ** resolution_level
            voxel_size = [base_voxel_size * scale_factor] * 3
        else:
            # Single scale segmentation
            voxel_size = [base_voxel_size] * 3
            resolution_level = 0  # Reset to 0 for display
            
        logger.info(f"Voxel size: {voxel_size}")
        
        # Actually load the data (not lazy!)
        yield f"Loading segmentation data into memory..."
        logger.info("Loading array data into memory...")
        loaded_data = np.array(array)
        logger.info(f"Data loaded successfully. Final shape: {loaded_data.shape}")
            
        logger.info("Returning final result...")
        
        # Return the final result with pre-loaded data
        return {
            'segmentation': segmentation,
            'data': loaded_data,
            'voxel_size': voxel_size,
            'name': f"Segmentation: {segmentation.meta.name} (Level {resolution_level})",
            'resolution_level': resolution_level
        }
                         
    except Exception as e:
        error_msg = f"Error loading segmentation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@thread_worker
def expand_run_worker(run):
    """Expand a run in the tree by gathering voxel spacings and picks data."""
    logger = logging.getLogger(f"ExpandRunWorker_{id(run)}")
    
    try:
        logger.info(f"Starting run expansion for {run.meta.name}")
        
        yield f"Loading voxel spacings for {run.meta.name}..."
        
        # Get voxel spacings (usually fast)
        voxel_spacings = list(run.voxel_spacings)
        logger.info(f"Found {len(voxel_spacings)} voxel spacings")
        
        yield f"Loading picks for {run.meta.name}..."
        
        # Get picks (can be slow)
        logger.info("Loading picks...")
        picks = run.picks
        logger.info(f"Found {len(picks)} picks")
        
        # Organize picks by user_id and session_id
        yield f"Organizing picks data..."
        user_dict = {}
        for pick in picks:
            if pick.meta.user_id not in user_dict:
                user_dict[pick.meta.user_id] = {}
            if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                user_dict[pick.meta.user_id][pick.meta.session_id] = []
            user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)
        
        logger.info("Run expansion completed successfully")
        
        return {
            'run': run,
            'voxel_spacings': voxel_spacings,
            'picks_data': user_dict
        }
        
    except Exception as e:
        error_msg = f"Error expanding run: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@thread_worker  
def expand_voxel_spacing_worker(voxel_spacing):
    """Expand a voxel spacing in the tree by gathering tomograms and segmentations."""
    logger = logging.getLogger(f"ExpandVoxelSpacingWorker_{id(voxel_spacing)}")
    
    try:
        logger.info(f"Starting voxel spacing expansion for {voxel_spacing.meta.voxel_size}")
        
        yield f"Loading tomograms for voxel size {voxel_spacing.meta.voxel_size}..."
        
        # Get tomograms (usually fast)
        tomograms = list(voxel_spacing.tomograms)
        logger.info(f"Found {len(tomograms)} tomograms")
        
        yield f"Loading segmentations for voxel size {voxel_spacing.meta.voxel_size}..."
        
        # Get segmentations (can be slow)
        logger.info("Loading segmentations...")
        segmentations = voxel_spacing.run.get_segmentations(voxel_size=voxel_spacing.meta.voxel_size)
        logger.info(f"Found {len(segmentations)} segmentations")
        
        logger.info("Voxel spacing expansion completed successfully")
        
        return {
            'voxel_spacing': voxel_spacing,
            'tomograms': tomograms,
            'segmentations': segmentations
        }
        
    except Exception as e:
        error_msg = f"Error expanding voxel spacing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise