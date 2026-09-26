import numpy as np
import pytest

from cornersim.annotations import annotations_from_masks


def test_disconnected_instances_get_unique_xy_correct_boxes_and_semantic_labels():
    instance = np.zeros((8, 12, 3), dtype=np.uint8)
    semantic = np.zeros_like(instance)
    instance[1:4, 7:11] = (1, 2, 3)
    semantic[1:4, 7:11] = (0, 0, 142)
    instance[5:7, 1:3] = (4, 5, 6)
    semantic[5:7, 1:3] = (220, 20, 60)
    result = annotations_from_masks(instance, semantic, {"car": (0, 0, 142), "pedestrian": (220, 20, 60)})
    assert result == [(1, 7, 1, 11, 4, "car"), (2, 1, 5, 3, 7, "pedestrian")]


def test_small_and_unknown_components_are_excluded():
    instance = np.zeros((4, 4, 3), dtype=np.uint8)
    semantic = np.zeros_like(instance)
    instance[0, 0] = (1, 1, 1)
    instance[2:4, 2:4] = (2, 2, 2)
    semantic[2:4, 2:4] = (99, 99, 99)
    assert annotations_from_masks(instance, semantic, {"car": (0, 0, 142)}) == []


def test_mask_dimensions_must_match():
    with pytest.raises(ValueError, match="identical"):
        annotations_from_masks(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)), {})
