"""Bounded same-frame sensor aggregation."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable
from queue import Empty
import time


class SynchronizationError(ValueError):
    pass


@dataclass(frozen=True)
class SensorFrame:
    frame: int
    measurements: dict[str, Any]


class FrameSynchronizer:
    def __init__(self, sensor_names: Iterable[str], max_pending_frames: int = 8):
        self.required = frozenset(sensor_names)
        if not self.required or max_pending_frames < 1:
            raise SynchronizationError("sensor names and a positive buffer size are required")
        self.max_pending_frames = max_pending_frames
        self._pending: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.dropped_frames: list[int] = []

    def add(self, frame: int, sensor_name: str, measurement: Any) -> SensorFrame | None:
        if sensor_name not in self.required:
            raise SynchronizationError(f"unexpected sensor {sensor_name!r}")
        bucket = self._pending.setdefault(frame, {})
        if sensor_name in bucket:
            raise SynchronizationError(f"duplicate {sensor_name!r} measurement for frame {frame}")
        bucket[sensor_name] = measurement
        if self.required == bucket.keys():
            del self._pending[frame]
            return SensorFrame(frame, bucket)
        while len(self._pending) > self.max_pending_frames:
            dropped, _ = self._pending.popitem(last=False)
            self.dropped_frames.append(dropped)
        return None


def collect_frame(sensor_queue: Any, synchronizer: FrameSynchronizer, target_frame: int,
                  timeout: float) -> SensorFrame:
    """Collect exactly ``target_frame`` or raise with actionable diagnostics."""
    deadline = time.monotonic() + timeout
    delayed: list[int] = []
    while time.monotonic() < deadline:
        try:
            frame, sensor_name, measurement = sensor_queue.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty as error:
            break
        if frame < target_frame:
            delayed.append(frame)
            continue
        completed = synchronizer.add(frame, sensor_name, measurement)
        if completed is not None and completed.frame == target_frame:
            return completed
    pending = sorted(synchronizer._pending.get(target_frame, {}))
    missing = sorted(synchronizer.required.difference(pending))
    details = f"missing sensors {missing} for frame {target_frame}"
    if delayed:
        details += f"; discarded delayed frames {sorted(set(delayed))}"
    raise TimeoutError(details)
