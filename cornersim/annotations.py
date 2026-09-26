"""Dependency-light mask-to-box annotation generation."""

from collections import deque
from typing import Mapping

import numpy as np


def _components(mask: np.ndarray):
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        points = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for adjacent_y, adjacent_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= adjacent_y < height and 0 <= adjacent_x < width and
                        mask[adjacent_y, adjacent_x] and not visited[adjacent_y, adjacent_x]):
                    visited[adjacent_y, adjacent_x] = True
                    queue.append((adjacent_y, adjacent_x))
        yield np.asarray(points)


def annotations_from_masks(instance_array: np.ndarray, semantic_array: np.ndarray,
                           class_mapping: Mapping[str, tuple[int, int, int]],
                           *, minimum_pixels: int = 4) -> list[tuple[int, int, int, int, int, str]]:
    """Create unique, half-open boxes using the majority semantic class per component."""
    instance_array = np.asarray(instance_array)[..., :3]
    semantic_array = np.asarray(semantic_array)[..., :3]
    if instance_array.shape != semantic_array.shape or instance_array.ndim != 3:
        raise ValueError("instance and semantic images must have identical dimensions (HxWx3)")
    reverse_mapping = {}
    for label, color in class_mapping.items():
        # Preserve the first/canonical label when aliases share a palette color
        # (CARLA's `car` and legacy `vehicle` both use 0, 0, 142).
        reverse_mapping.setdefault(tuple(color), label)
    results = []
    next_id = 1
    for color in np.unique(instance_array.reshape(-1, 3), axis=0):
        if np.array_equal(color, (0, 0, 0)):
            continue
        mask = np.all(instance_array == color, axis=2)
        for coordinates in _components(mask):
            if len(coordinates) < minimum_pixels:
                continue
            ys, xs = coordinates[:, 0], coordinates[:, 1]
            colors, counts = np.unique(semantic_array[ys, xs], axis=0, return_counts=True)
            label = reverse_mapping.get(tuple(colors[np.argmax(counts)].tolist()))
            if label is None or label == "unlabeled":
                continue
            results.append((next_id, int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1), label))
            next_id += 1
    return results
