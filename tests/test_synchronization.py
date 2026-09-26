import pytest
from queue import Queue

from cornersim.synchronization import FrameSynchronizer, SynchronizationError, collect_frame


def test_out_of_order_measurements_return_same_frame_only():
    sync = FrameSynchronizer(["rgb", "semantic"])
    assert sync.add(2, "rgb", "r2") is None
    assert sync.add(1, "semantic", "s1") is None
    result = sync.add(2, "semantic", "s2")
    assert result.frame == 2
    assert result.measurements == {"rgb": "r2", "semantic": "s2"}


def test_buffer_is_bounded_and_records_drops():
    sync = FrameSynchronizer(["a", "b"], max_pending_frames=1)
    sync.add(1, "a", 1)
    sync.add(2, "a", 2)
    assert sync.dropped_frames == [1]


def test_duplicate_measurement_is_error():
    sync = FrameSynchronizer(["rgb", "semantic"])
    sync.add(1, "rgb", object())
    with pytest.raises(SynchronizationError, match="duplicate"):
        sync.add(1, "rgb", object())


def test_collect_frame_discards_delayed_and_never_mixes_frames():
    queue = Queue()
    for item in ((4, "rgb", "old"), (5, "rgb", "r5"), (6, "semantic", "s6"),
                 (5, "semantic", "s5")):
        queue.put(item)
    frame = collect_frame(queue, FrameSynchronizer(["rgb", "semantic"]), 5, 0.1)
    assert frame.frame == 5
    assert frame.measurements == {"rgb": "r5", "semantic": "s5"}


def test_collect_frame_reports_missing_sensor_and_delayed_frames():
    queue = Queue()
    queue.put((3, "rgb", "old"))
    queue.put((4, "rgb", "current"))
    with pytest.raises(TimeoutError, match=r"missing sensors \['semantic'\].*delayed frames \[3\]"):
        collect_frame(queue, FrameSynchronizer(["rgb", "semantic"]), 4, 0.01)
