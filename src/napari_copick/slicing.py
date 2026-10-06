"""Slice thickness for point layers on napari >= 0.9.

Up to napari 0.8, ``out_of_slice_display`` drew a point in every slice within its radius. napari 0.9 replaced it with
thick slices: a point layer's projection mode decides how points *within the slice thickness* are drawn, and the
default thickness is zero, so filaments and picks showed only where a point happens to lie in the current slice. On
napari >= 0.9 the layers below therefore widen the slice to the point size, and tomograms keep showing their single
central slice (``projection_mode="none"``) instead of the mean over the thicker slab.
"""

from typing import Any

import napari
from packaging.version import Version

THICK_SLICES = Version(napari.__version__).release >= (0, 9)


def show_points_within(viewer: Any, size: float) -> None:
    """Make the slice at least ``size`` thick (world units) on the axes that are not displayed, so points of that size
    show in every slice they touch (napari >= 0.9; earlier versions use ``out_of_slice_display``)."""
    if not THICK_SLICES or not size:
        return
    dims = viewer.dims
    thickness = list(dims.thickness)
    changed = False
    for axis in dims.not_displayed:
        if thickness[axis] < size:
            thickness[axis] = float(size)
            changed = True
    if changed:
        dims.thickness = tuple(thickness)


def single_slice(layer: Any) -> None:
    """Show an image layer's central slice rather than the mean over a thick slice (napari >= 0.9)."""
    if THICK_SLICES:
        layer.projection_mode = "none"


def points_slicing(near: bool) -> dict:
    """Keyword arguments for a points layer drawn near the slice (``near``, scaled by distance) or only in it."""
    if THICK_SLICES:
        return {"projection_mode": "rescale_linear" if near else "none"}
    return {"out_of_slice_display": near}


def vectors_near_slice(layer: Any) -> None:
    """Draw a vectors layer in the slices its arrows pass through, fading with distance."""
    if THICK_SLICES:
        layer.projection_mode = "fade"
    else:
        layer.out_of_slice_display = True
