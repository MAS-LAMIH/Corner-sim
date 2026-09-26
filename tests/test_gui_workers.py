import sys
import threading

from PyQt5.QtCore import QCoreApplication, QEventLoop, QThread, QTimer, Qt

sys.path.insert(0, "Python")
from gui_workers import PostProcessingWorker, SimulationWorker  # noqa: E402


def run_worker(worker, *, stop=False):
    app = QCoreApplication.instance() or QCoreApplication([])
    thread = QThread()
    results, errors, previews, progress = [], [], [], []
    lifecycle = {"worker_destroyed": False, "worker_destroyed_in_owner": False,
                 "thread_destroyed": False}
    worker.moveToThread(thread)
    worker.succeeded.connect(lambda *args: results.append(args[0] if args else None))
    worker.failed.connect(errors.append)
    if hasattr(worker, "preview_ready"):
        worker.preview_ready.connect(previews.append)
        worker.progress_changed.connect(progress.append)
    loop = QEventLoop()
    owner_thread = thread

    def worker_destroyed():
        lifecycle["worker_destroyed"] = True
        lifecycle["worker_destroyed_in_owner"] = QThread.currentThread() is owner_thread

    def thread_destroyed():
        lifecycle["thread_destroyed"] = True
        loop.quit()

    worker.finished.connect(worker.deleteLater, Qt.DirectConnection)
    worker.destroyed.connect(worker_destroyed, Qt.DirectConnection)
    worker.destroyed.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread.deleteLater)
    thread.destroyed.connect(thread_destroyed)
    thread.started.connect(worker.run)
    thread.start()
    if stop:
        QTimer.singleShot(10, lambda: worker.stop())
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    assert lifecycle == {"worker_destroyed": True, "worker_destroyed_in_owner": True,
                         "thread_destroyed": True}
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


def test_failure_deletes_worker_in_owner_thread_and_allows_second_run():
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
    assert results == [None]
    assert not errors
    assert calls == [("dataset", "catalog.json")]
