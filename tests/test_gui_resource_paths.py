import ast
from pathlib import Path


def test_gui_bootstrap_adds_repository_root_before_local_core_imports():
    source = Path("Python/new_ui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {target.id for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    assert {"PYTHON_DIR", "REPO_ROOT"} <= assignments
    assert source.index("sys.path.insert") < source.index("from Synchro3 import")


def test_gui_resources_are_resolved_from_source_tree_not_working_directory():
    source = Path("Python/new_ui.py").read_text(encoding="utf-8")
    assert "CornerCaseEditor(str(PYTHON_DIR / 'CC_terminology.json')" in source
    assert "str(PYTHON_DIR / 'environment_object.json')" in source
    assert 'QIcon("icons/' not in source
    for resource in (Path("Python/CC_terminology.json"), Path("Python/environment_object.json"),
                     Path("icons/play.png"), Path("icons/stop.png")):
        assert resource.is_file()


def test_sensor_and_worker_modules_do_not_construct_or_update_widgets():
    for path in (Path("Python/Synchro3.py"), Path("Python/gui_workers.py")):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("QPixmap", "QImage", "QWidget", "QLabel", ".setPixmap(", ".setValue("):
            assert forbidden not in source


def test_startup_exception_handler_guards_uninitialized_window():
    source = Path("Python/new_ui.py").read_text(encoding="utf-8")
    assert "if main_window is not None and main_window.client is not None:" in source


def test_public_lifecycle_signal_signatures_are_typed():
    source = Path("Python/new_ui.py").read_text(encoding="utf-8")
    assert "simulation_finished = pyqtSignal(dict)" in source
    assert "simulation_failed = pyqtSignal(str)" in source
    assert "postprocessing_failed = pyqtSignal(str)" in source


def _method_source(name):
    source = Path("Python/new_ui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_window.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, method)


def test_stop_only_requests_cancellation_until_worker_cleanup_finishes():
    stop_source = _method_source("stop_scenario")
    finished_source = _method_source("_simulation_thread_destroyed")
    assert "self.simulation_worker.stop()" in stop_source
    assert "self.generate_Scenario_name()" not in stop_source
    assert "self.client = None" not in stop_source
    assert "self.start_action.setEnabled(True)" not in stop_source
    assert "self.generate_Scenario_name()" in finished_source
    assert "self.client = None" in finished_source


def test_cancellation_skips_postprocessing_and_does_not_close_window():
    stop_source = _method_source("stop_scenario")
    finished_source = _method_source("_simulation_thread_destroyed")
    assert "self.close()" not in stop_source
    assert 'result.get("stopped")' in finished_source
    cancelled_branch, completed_branch = finished_source.split("elif result:", 1)
    assert "_start_postprocessing" not in cancelled_branch
    assert "_start_postprocessing" in completed_branch
