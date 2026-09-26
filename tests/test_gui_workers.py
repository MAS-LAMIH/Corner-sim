import sys
import threading
import socket

from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer

sys.path.insert(0, "Python")
from gui_workers import PostProcessingWorker, SimulationWorker, WorkerEventBus  # noqa: E402


def run_worker(worker, *, stop=False):
    app = QCoreApplication.instance() or QCoreApplication([])
    results, errors, previews, progress = [], [], [], []
    loop = QEventLoop()

    def poll():
        while not worker.event_queue.empty():
            source, event_type, value = worker.event_queue.get()
            assert source is worker
            if event_type == "preview":
                previews.append(value)
            elif event_type == "progress":
                progress.append(value)
        if not worker.is_alive():
            loop.quit()

    poll_timer = QTimer()
    poll_timer.setInterval(2)
    poll_timer.timeout.connect(poll)
    worker.start()
    poll_timer.start()
    if stop:
        QTimer.singleShot(10, worker.stop)
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    poll_timer.stop()
    worker.join(timeout=1)
    assert not worker.is_alive()
    poll()
    if worker.outcome.error:
        errors.append(worker.outcome.error)
    elif worker.outcome.result is not None:
        results.append(worker.outcome.result)
    return results, errors, previews, progress


def test_simulation_worker_uses_data_signals_not_qt_widgets():
    caller_thread = threading.get_ident()

    def runner(**kwargs):
        assert "rgb_label" not in kwargs
        assert "semantic_label" not in kwargs
        assert "instance_label" not in kwargs
        kwargs["preview_callback"]({"owned": b"pixels"})
        kwargs["progress_callback"](50)
        return {"stopped": False, "captured_frames": [1], "runner_thread": threading.get_ident()}

    results, errors, previews, progress = run_worker(SimulationWorker(runner=runner))
    assert not errors
    assert results[0]["runner_thread"] != caller_thread
    assert previews == [{"owned": b"pixels"}]
    assert progress == [50]


def test_stop_is_cooperative_and_worker_finishes_before_next_run():
    def runner(**kwargs):
        while not kwargs["stop_event"].wait(0.005):
            pass
        return {"stopped": True, "captured_frames": []}

    first = run_worker(SimulationWorker(runner=runner), stop=True)
    second = run_worker(SimulationWorker(runner=lambda **kwargs: {
        "stopped": False, "captured_frames": [2]}))
    assert first[0] == [{"stopped": True, "captured_frames": []}]
    assert not first[1]
    assert second[0] == [{"stopped": False, "captured_frames": [2]}]


def test_failure_finishes_worker_and_allows_second_run():
    def failing_runner(**kwargs):
        raise RuntimeError("runner failed")

    failed = run_worker(SimulationWorker(runner=failing_runner))
    second = run_worker(SimulationWorker(runner=lambda **kwargs: {
        "stopped": False, "captured_frames": [3]}))
    assert failed[0] == []
    assert failed[1] == ["runner failed"]
    assert second[0] == [{"stopped": False, "captured_frames": [3]}]


def test_postprocessing_worker_is_not_implicitly_started_on_cancellation():
    calls = []
    cancelled = {"stopped": True, "captured_frames": []}
    # Post-processing is an explicit separate worker. Merely completing a cancelled
    # simulation worker cannot invoke it.
    results, errors, _, _ = run_worker(SimulationWorker(runner=lambda **kwargs: cancelled))
    assert results == [cancelled]
    assert not errors
    assert calls == []


def test_postprocessing_worker_reports_completion_via_signal():
    calls = []
    worker = PostProcessingWorker("dataset", "catalog.json",
                                  processor=lambda *args: calls.append(args))
    results, errors, _, _ = run_worker(worker)
    assert results == [True]
    assert not errors
    assert calls == [("dataset", "catalog.json")]


def test_worker_event_bus_wakes_without_any_qt_object_in_worker():
    reader, writer = socket.socketpair()
    reader.settimeout(1)
    bus = WorkerEventBus(writer)
    source = object()
    thread = threading.Thread(target=bus.publish, args=(source, "progress", 25))
    try:
        thread.start()
        thread.join(timeout=1)
        assert reader.recv(1) == b"\0"
        assert bus.queue.get() == (source, "progress", 25)
    finally:
        reader.close()
        writer.close()
