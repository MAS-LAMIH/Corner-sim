import numpy as np
import pytest

from cornersim.geometry import ProjectionError, build_projection_matrix, project_bbox, project_point


def test_center_point_projects_to_principal_point():
    matrix = build_projection_matrix(800, 600, 90)
    assert project_point((10, 0, 0), matrix, np.eye(4)) == pytest.approx((400, 300, 10))


def test_points_behind_camera_are_rejected():
    assert project_point((-1, 0, 0), build_projection_matrix(800, 600, 90), np.eye(4)) is None


def test_bbox_is_clipped_to_image_boundaries():
    vertices = [(10, y, z) for y in (-20, 20) for z in (-20, 20) for _ in (0, 1)]
    box = project_bbox(vertices, build_projection_matrix(100, 80, 90), np.eye(4), 100, 80)
    assert box is not None
    assert (box.min_x, box.min_y, box.max_x, box.max_y) == (0, 0, 100, 80)


def test_near_plane_crossing_is_rejected():
    vertices = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    assert project_bbox(vertices, build_projection_matrix(100, 80, 90), np.eye(4), 100, 80) is None


@pytest.mark.parametrize("args", [(0, 10, 90), (10, 10, 0), (10, 10, 180)])
def test_invalid_camera_parameters_fail(args):
    with pytest.raises(ProjectionError):
        build_projection_matrix(*args)
