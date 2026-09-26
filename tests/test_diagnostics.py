import json
from pathlib import Path
import subprocess
import sys

from diagnostics.common import CheckpointLog, ClientTrace, trace_output_resolution


class FakeSettings:
    synchronous_mode = False
    fixed_delta_seconds = None


class FakeMap:
    name = "TownDiagnostic"


class FakeActor:
    id = 7
    type_id = "sensor.camera.rgb"

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        return "stopped"

    def destroy(self):
        return "destroyed"


class FakeWorld:
    def __init__(self):
        self.actor = FakeActor()
        self.attached_to = None

    def get_map(self):
        return FakeMap()

    def get_settings(self):
        return FakeSettings()

    def apply_settings(self, settings):
        self.settings = settings

    def spawn_actor(self, blueprint, transform, *args, **kwargs):
        self.attached_to = kwargs.get("attach_to")
        return self.actor

    def tick(self):
        return 42


class FakeClient:
    def __init__(self):
        self.world = FakeWorld()

    def set_timeout(self, seconds):
        self.timeout = seconds

    def get_world(self):
        return self.world

    def get_trafficmanager(self, port):
        class Manager:
            def set_synchronous_mode(self, enabled):
                self.enabled = enabled

            def set_random_device_seed(self, seed):
                self.seed = seed

        return Manager()


def test_trace_proxy_is_transparent_and_records_lifecycle(tmp_path):
    path = tmp_path / "trace.jsonl"
    log = CheckpointLog(path)
    native = FakeClient()
    client = ClientTrace(native, log)
    client.set_timeout(3)
    world = client.get_world()
    parent = world.spawn_actor("ego", object())
    child = world.spawn_actor("camera", object(), attach_to=parent)
    assert native.world.attached_to is native.world.actor
    assert child.stop() == "stopped"
    assert child.destroy() == "destroyed"
    assert world.tick() == 42
    manager = client.get_trafficmanager(8050)
    manager.set_random_device_seed(4)
    manager.set_synchronous_mode(True)
    manager.set_synchronous_mode(False)
    events = [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert "carla.world.begin" in events
    assert "actor.spawn.begin" in events
    assert "actor.stop" in events
    assert "actor.destroy" in events
    assert "world.first_tick" in events
    assert "carla.traffic_manager.end" in events
    assert events.count("carla.traffic_manager.synchronous_mode") == 2


def test_non_gui_harness_help_does_not_import_pyqt():
    command = [sys.executable, "-c", """
import runpy, sys
sys.argv = ['run_standalone.py', '--help']
try:
    runpy.run_path('diagnostics/run_standalone.py', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name == 'PyQt5' or name.startswith('PyQt5.') for name in sys.modules)
"""]
    completed = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_output_resolution_tracer_targets_production_line(tmp_path):
    sys.path.insert(0, "Python")
    from Synchro3 import run_carla_simulation

    original = sys.gettrace()
    with trace_output_resolution(run_carla_simulation, CheckpointLog(tmp_path / "trace.jsonl")):
        assert sys.gettrace() is not original
    assert sys.gettrace() is original


def test_windows_matrix_uses_fresh_processes_and_separate_logs():
    source = Path("diagnostics/run_windows_isolation.ps1").read_text(encoding="utf-8")
    assert "Start-Process" in source
    assert "-X\", \"faulthandler" in source
    assert ".stdout.log" in source
    assert ".stderr.log" in source
    assert ".checkpoints.jsonl" in source
    assert 'Run-Probe "A-standalone-live"' in source
    assert 'Run-Probe "B3-gui-fake-repeat"' in source
    assert 'Run-Probe "C1-gui-live-natural"' in source
    assert 'Run-Probe "C0-gui-live-natural-no-post"' in source
    assert 'Run-Probe "E1-post-standalone-main"' in source
    assert 'Run-Probe "E4-post-qt-thread"' in source


def test_live_harness_brackets_real_postprocessing_transition():
    source = Path("diagnostics/run_gui_live.py").read_text(encoding="utf-8")
    for checkpoint in (
        "postprocess.transition.begin",
        "postprocess.import.begin",
        "postprocess.import.end",
        "postprocess.worker.entry",
        "postprocess.worker.return",
        "postprocess.transition.return",
    ):
        assert checkpoint in source
    assert 'choices=("real", "skip")' in source


def test_postprocess_harness_help_is_available_without_loading_native_dependencies():
    completed = subprocess.run(
        [sys.executable, "diagnostics/run_postprocess.py", "--help"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert "standalone-main" in completed.stdout
    assert "qt-thread" in completed.stdout
