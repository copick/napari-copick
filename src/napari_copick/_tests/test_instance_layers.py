"""Instance colormaps and save-time validation."""

import numpy as np
import pytest
from copick_shared_ui.util.instances import instance_rgb

from napari_copick.instance_layers import instance_colormap, segment_colormap
from napari_copick.save_utils import scale_segmentation_to_target_shape, validate_segmentation_values


def test_instance_colormap_hides_and_colors():
    cmap = instance_colormap([1, 2, 300], hidden=[2])
    d = cmap.color_dict
    assert np.allclose(d[1][:3], instance_rgb(1))
    assert d[2][3] == 0 and d[0][3] == 0
    assert np.allclose(d[300][:3], instance_rgb(300))


def test_segment_colormap():
    from copick_shared_ui.util.instances import InstanceRow

    rows = [InstanceRow(0, color=(0.5, 0.5, 0.5, 1.0), key=1), InstanceRow(3, color=(1, 0, 0, 1), key=2)]
    d = segment_colormap(rows, hidden={2}).color_dict
    assert np.allclose(d[1], [0.5, 0.5, 0.5, 1.0]) and d[2][3] == 0


def test_validate_segmentation_values():
    validate_segmentation_values(np.array([0, 1, 1], dtype=np.uint8), "binary")
    with pytest.raises(ValueError, match="binary"):
        validate_segmentation_values(np.array([0, 2]), "binary")
    validate_segmentation_values(np.array([0, 2]), "binary", converting=True)
    validate_segmentation_values(np.array([0, 70000], dtype=np.uint32), "instance")
    with pytest.raises(ValueError, match="negative"):
        validate_segmentation_values(np.array([-1, 2]), "instance")
    with pytest.raises(ValueError, match="whole"):
        validate_segmentation_values(np.array([0.5]), "instance")


def test_scaling_keeps_large_ids():
    data = np.zeros((2, 2, 2), dtype=np.uint32)
    data[0, 0, 0] = 70001
    scaled = scale_segmentation_to_target_shape(data, (4, 4, 4))
    assert scaled.dtype == np.uint32 and scaled.max() == 70001
