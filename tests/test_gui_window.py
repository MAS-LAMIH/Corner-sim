"""QWidget smoke test; skipped when the host lacks the Qt/OpenGL runtime."""

import importlib
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets", exc_type=ImportError)
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
