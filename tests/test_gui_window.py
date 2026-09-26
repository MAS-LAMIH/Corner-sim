"""QWidget smoke test; skipped when the host lacks the Qt/OpenGL runtime."""

import importlib
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)
from PyQt5.QtCore import QEventLoop, QTimer
sys.path.insert(0, "Python")
try:
    new_ui = importlib.import_module("new_ui")
except ImportError as error:
    pytest.skip(f"GUI dependencies unavailable: {error}", allow_module_level=True)


def test_main_window_declares_lifecycle_signals_and_instantiates_offscreen():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = new_ui.MainWindow()
    try:
        assert window.simulation_finished is not None
        assert window.simulation_failed is not None
        assert window.postprocessing_failed is not None
        window.simulation_finished.emit({"stopped": False})
        window.simulation_failed.emit("test failure")
        assert window.pending_result == {"stopped": False}
        assert window.pending_error == "test failure"
    finally:
        window.close()
        app.processEvents()


def wait_until(predicate, timeout_ms=3000):
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(5)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start(timeout_ms)
    loop.exec()
    poll.stop()
    assert predicate(), "Qt operation did not finish before timeout"


def prepare_window(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = new_ui.MainWindow()
    monkeypatch.setattr(window, "init_carla_client", lambda: (
        setattr(window, "client", object()), setattr(window, "world", object())))
    monkeypatch.setattr(new_ui.QMessageBox, "critical", lambda *args: None)
    return app, window


def test_actual_main_window_start_path_survives_completion_and_second_run(monkeypatch, tmp_path):
    app, window = prepare_window(monkeypatch)
    processed = []

    def runner(**kwargs):
        kwargs["progress_callback"](100)
        return {"stopped": False, "captured_frames": [1], "output_dir": str(tmp_path)}

    window.simulation_runner = runner
    window.postprocessing_processor = lambda *args: processed.append(args)
    try:
        window.start_scenario()
        wait_until(lambda: window.simulation_worker is None and window.postprocessing_worker is None)
        assert len(processed) == 1
        assert window.start_action.isEnabled()
        window.start_scenario()
        wait_until(lambda: window.simulation_worker is None and window.postprocessing_worker is None)
        assert len(processed) == 2
        assert window.progress_bar.value() == 100
    finally:
        window.close()
        app.processEvents()


def test_actual_main_window_stop_path_survives_and_skips_postprocessing(monkeypatch):
    app, window = prepare_window(monkeypatch)
    processed = []

    def runner(**kwargs):
        kwargs["stop_event"].wait(2)
        return {"stopped": kwargs["stop_event"].is_set(), "captured_frames": [], "output_dir": "unused"}

    window.simulation_runner = runner
    window.postprocessing_processor = lambda *args: processed.append(args)
    try:
        window.start_scenario()
        QTimer.singleShot(20, window.stop_scenario)
        wait_until(lambda: window.simulation_worker is None)
        assert processed == []
        assert window.start_action.isEnabled()
    finally:
        window.close()
        app.processEvents()


def test_actual_main_window_failure_path_survives_and_can_run_again(monkeypatch, tmp_path):
    app, window = prepare_window(monkeypatch)
    try:
        window.simulation_runner = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fake failure"))
        window.start_scenario()
        wait_until(lambda: window.simulation_worker is None)
        assert window.start_action.isEnabled()
        window.simulation_runner = lambda **kwargs: {
            "stopped": False, "captured_frames": [], "output_dir": str(tmp_path)}
        window.postprocessing_processor = lambda *args: None
        window.start_scenario()
        wait_until(lambda: window.simulation_worker is None and window.postprocessing_worker is None)
        assert window.start_action.isEnabled()
    finally:
        window.close()
        app.processEvents()
