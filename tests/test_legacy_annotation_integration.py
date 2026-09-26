import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.modules.setdefault("cv2", SimpleNamespace())  # CI has OpenCV but lacks its optional libGL runtime.
sys.path.insert(0, "Python")
import image_tools  # noqa: E402


def test_legacy_projection_delegates_to_safe_geometry():
    intrinsic = image_tools.build_projection_matrix(800, 600, 90)
    point = SimpleNamespace(x=10, y=0, z=0)
    assert image_tools.get_image_point(point, intrinsic, np.eye(4)) == pytest.approx((400, 300))
    with pytest.raises(ValueError, match="behind"):
        image_tools.get_image_point(SimpleNamespace(x=-1, y=0, z=0), intrinsic, np.eye(4))


def test_instance_semantic_annotations_use_xy_order_half_open_bounds_and_unique_ids(tmp_path):
    instance = np.zeros((8, 12, 3), dtype=np.uint8)
    semantic = np.zeros_like(instance)
    instance[1:4, 7:11] = (1, 2, 3)
    semantic[1:4, 7:11] = (0, 0, 142)
    instance[5:7, 1:3] = (4, 5, 6)
    semantic[5:7, 1:3] = (220, 20, 60)
    instance_path, semantic_path = tmp_path / "instance.png", tmp_path / "semantic.png"
    Image.fromarray(instance).save(instance_path)
    Image.fromarray(semantic).save(semantic_path)

    annotations = image_tools.process_instance_semantic_segmentation_(
        instance_path, semantic_path, image_tools.CLASS_MAPPING)

    assert annotations == [(1, 7, 1, 11, 4, "car"), (2, 1, 5, 3, 7, "pedestrian")]


def test_instance_semantic_dimensions_must_match(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (10, 10)).save(first)
    Image.new("RGB", (9, 10)).save(second)
    with pytest.raises(ValueError, match="identical dimensions"):
        image_tools.process_instance_semantic_segmentation_(first, second, image_tools.CLASS_MAPPING)
