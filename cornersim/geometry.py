"""Camera projection helpers using CARLA's Unreal coordinate convention."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class ProjectionError(ValueError):
    """Raised when camera parameters or geometry cannot be projected safely."""


@dataclass(frozen=True)
class BoundingBox:
    """A clipped image-space box with half-open maximum coordinates."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def area(self) -> float:
        return max(0.0, self.max_x - self.min_x) * max(0.0, self.max_y - self.min_y)


def build_projection_matrix(width: int, height: int, fov_degrees: float) -> np.ndarray:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ProjectionError("image width and height must be positive integers")
    if not math.isfinite(fov_degrees) or not 0.0 < fov_degrees < 180.0:
        raise ProjectionError("field of view must be finite and between 0 and 180 degrees")
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return np.array(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)))


def project_point(point: Sequence[float], intrinsic: np.ndarray, world_to_camera: np.ndarray,
                  *, near_clip: float = 1e-3) -> tuple[float, float, float] | None:
    """Project a world point, returning ``(x, y, depth)`` or ``None`` behind camera."""
    if len(point) != 3:
        raise ProjectionError("a world point must contain exactly three coordinates")
    intrinsic = np.asarray(intrinsic, dtype=float)
    transform = np.asarray(world_to_camera, dtype=float)
    if intrinsic.shape != (3, 3) or transform.shape != (4, 4):
        raise ProjectionError("intrinsic and world-to-camera matrices must be 3x3 and 4x4")
    homogeneous = np.asarray((*point, 1.0), dtype=float)
    if not np.all(np.isfinite(homogeneous)):
        raise ProjectionError("point coordinates must be finite")
    camera = transform @ homogeneous
    # Unreal (forward, right, up) -> conventional camera (right, down, forward).
    conventional = np.array((camera[1], -camera[2], camera[0]))
    depth = float(conventional[2])
    if depth <= near_clip:
        return None
    pixel = intrinsic @ conventional
    return float(pixel[0] / depth), float(pixel[1] / depth), depth


def project_bbox(vertices: Iterable[Sequence[float]], intrinsic: np.ndarray,
                 world_to_camera: np.ndarray, width: int, height: int,
                 *, near_clip: float = 1e-3, min_area: float = 1.0) -> BoundingBox | None:
    """Project and clip a 3-D box; reject boxes crossing/behind the near plane."""
    points = list(vertices)
    if len(points) != 8:
        raise ProjectionError("a 3-D bounding box must contain exactly eight vertices")
    projected = [project_point(point, intrinsic, world_to_camera, near_clip=near_clip) for point in points]
    # Projecting only front vertices can create a misleading box for near-plane crossings.
    if any(point is None for point in projected):
        return None
    visible = [point for point in projected if point is not None]
    box = BoundingBox(
        max(0.0, min(point[0] for point in visible)),
        max(0.0, min(point[1] for point in visible)),
        min(float(width), max(point[0] for point in visible)),
        min(float(height), max(point[1] for point in visible)),
    )
    return box if box.area >= min_area else None
