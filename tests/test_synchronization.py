import pytest

from cornersim.synchronization import FrameSynchronizer, SynchronizationError


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
