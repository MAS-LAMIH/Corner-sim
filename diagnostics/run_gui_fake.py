"""Exercise the real MainWindow start path with no CARLA calls."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
for entry in (str(REPO_ROOT), str(PYTHON_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from diagnostics.common import CheckpointLog, module_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("natural", "stop", "repeat"), default="natural")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--stop-after", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostic-output" / "gui-fake"))
    parser.add_argument("--log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = CheckpointLog(args.log)
    log.emit("process.start", configuration=f"gui-fake-{args.mode}", modules=module_state())
    try:
        log.emit("import.pyqt.begin")
        from PyQt5.QtCore import QCoreApplication, QEvent, QThread, QTimer
        from PyQt5.QtWidgets import QApplication, QMessageBox
        log.emit("import.pyqt.end", modules=module_state())
        log.emit("import.main_window.begin")
        import new_ui
        log.emit("import.main_window.end", modules=module_state())

        app = QApplication.instance() or QApplication(sys.argv[:1])
        log.emit("qt.application.created", qt_thread_id=int(QThread.currentThreadId()))
        window = new_ui.MainWindow()
        window.scenario_folder = str(Path(args.output).resolve())
        Path(window.scenario_folder).mkdir(parents=True, exist_ok=True)
        log.emit("qt.window.created", affinity_is_gui=window.thread() is app.thread())
        window.destroyed.connect(
            lambda: log.emit("qt.window.destroyed", qt_thread_id=int(QThread.currentThreadId())))
        window.worker_event_notifier.destroyed.connect(
            lambda: log.emit("qt.notifier.destroyed", qt_thread_id=int(QThread.currentThreadId())))

        # Avoid modal dialogs and all CARLA access while retaining start_scenario().
        QMessageBox.critical = lambda *unused: log.emit("qt.message.critical")
        QMessageBox.warning = lambda *unused: log.emit("qt.message.warning")
        window.init_carla_client = lambda: (
            setattr(window, "client", object()), setattr(window, "world", object()))

        state = {"started": 0, "completed": 0, "failed": False}

        def fake_runner(**kwargs):
            state["started"] += 1
            run_number = state["started"]
            log.emit("fake_runner.entry", run=run_number, modules=module_state())
            deadline = time.monotonic() + args.duration
            progress = 0
            while time.monotonic() < deadline and not kwargs["stop_event"].wait(0.05):
                progress = min(progress + 5, 99)
                kwargs["progress_callback"](progress)
            stopped = kwargs["stop_event"].is_set()
            result = {"stopped": stopped, "captured_frames": [],
                      "output_dir": str(Path(args.output).resolve())}
            log.emit("fake_runner.return", run=run_number, result=result)
            return result

        def fake_postprocess(*values):
            log.emit("fake_postprocess.entry", arguments=values)
            time.sleep(0.1)
            log.emit("fake_postprocess.return")

        window.simulation_runner = fake_runner
        window.postprocessing_processor = fake_postprocess
        window.simulation_finished.connect(
            lambda result: log.emit("qt.simulation_finished", result=result,
                                    qt_thread_id=int(QThread.currentThreadId())))
        window.simulation_failed.connect(
            lambda message: (state.__setitem__("failed", True),
                             log.emit("qt.simulation_failed", message=message)))
        window.show()

        def start_run():
            log.emit("qt.start_scenario", next_run=state["started"] + 1,
                     qt_thread_id=int(QThread.currentThreadId()))
            window.start_scenario()
            if args.mode == "stop":
                QTimer.singleShot(int(args.stop_after * 1000), stop_run)

        def stop_run():
            log.emit("qt.stop_scenario", qt_thread_id=int(QThread.currentThreadId()))
            window.stop_scenario()

        poll = QTimer(window)
        poll.setInterval(25)

        def inspect_state():
            idle = window.simulation_worker is None and window.postprocessing_worker is None
            if not idle or state["started"] == state["completed"]:
                return
            state["completed"] = state["started"]
            log.emit("qt.run.idle", run=state["completed"], start_enabled=window.start_action.isEnabled())
            target = 2 if args.mode == "repeat" else 1
            if state["completed"] < target:
                QTimer.singleShot(0, start_run)
            else:
                poll.stop()
                window.close()
                app.quit()

        poll.timeout.connect(inspect_state)
        poll.start()

        def watchdog():
            state["failed"] = True
            log.emit("watchdog.timeout", started=state["started"], completed=state["completed"])
            if window.simulation_worker is not None:
                window.stop_scenario()
            window.close()
            app.quit()

        QTimer.singleShot(int(args.timeout * 1000), watchdog)
        QTimer.singleShot(0, start_run)
        exit_code = app.exec()
        log.emit("qt.event_loop.return", code=exit_code, state=state)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        return 1 if state["failed"] or state["completed"] != (2 if args.mode == "repeat" else 1) else exit_code
    except BaseException as error:
        log.emit("process.failure", error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        log.emit("process.exit", modules=module_state())


if __name__ == "__main__":
    raise SystemExit(main())
