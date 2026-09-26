"""Testable core utilities for CornerSim.

The GUI remains in :mod:`Python`, while this package contains simulator-independent
validation and dataset primitives.
"""

from .geometry import BoundingBox, ProjectionError, build_projection_matrix, project_bbox, project_point

__all__ = ["BoundingBox", "ProjectionError", "build_projection_matrix", "project_bbox", "project_point"]
