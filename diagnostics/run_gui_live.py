"""Run the real MainWindow and live CARLA path with diagnostic checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
for entry in (str(REPO_ROOT), str(PYTHON_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from diagnostics.common import CheckpointLog, ClientTrace, module_state, trace_output_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8050)
    parser.add_argument("--ticks", type=int, default=5)
    parser.add_argument("--capture-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostic-output" / "gui-live"))
    parser.add_argument("--scenario")
    parser.add_argument("--stop-after", type=float,
                        help="request cooperative Stop after this many seconds")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="request cooperative Stop if the experiment exceeds this duration")
    parser.add_argument("--log")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--postprocessing", choices=("real", "skip"), default="real",
                        help="run the real post-processor or a logged no-op")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = CheckpointLog(args.log)
    log.emit("process.start", configuration="gui-live", modules=module_state())
    try:
        log.emit("import.pyqt.begin")
        from PyQt5.QtCore import QCoreApplication, QEvent, QThread, QTimer
        from PyQt5.QtWidgets import QApplication, QMessageBox
        log.emit("import.pyqt.end", modules=module_state())
        log.emit("import.main_window.begin")
        import carla
        import new_ui
        from Synchro3 import run_carla_simulation
        log.emit("import.main_window.end", modules=module_state())

        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = new_ui.MainWindow()
        output = str(Path(args.output).resolve())
        Path(output).mkdir(parents=True, exist_ok=True)
        window.scenario_folder = output
        window.scenario_length = args.ticks
        state = {"failed": False, "timed_out": False, "finished_signal": False}
        log.emit("qt.window.created", affinity_is_gui=window.thread() is app.thread(),
                 qt_thread_id=int(QThread.currentThreadId()), output=output)
        window.destroyed.connect(
            lambda: log.emit("qt.window.destroyed", qt_thread_id=int(QThread.currentThreadId())))
        window.worker_event_notifier.destroyed.connect(
            lambda: log.emit("qt.notifier.destroyed", qt_thread_id=int(QThread.currentThreadId())))

        def logged_critical(*values):
            log.emit("qt.message.critical", values=[str(value) for value in values[1:]])
            return QMessageBox.Ok

        QMessageBox.critical = logged_critical
        original_init_client = window.init_carla_client

        def traced_init_client():
            log.emit("gui.carla_initialization.begin")
            result = original_init_client()
            log.emit("gui.carla_initialization.end", connected=window.world is not None)
            return result

        window.init_carla_client = traced_init_client

        def traced_runner(**kwargs):
            log.emit("runner.entry", modules=module_state(), output_requested=kwargs.get("output_dir"))
            real_client = carla.Client(args.host, args.port)
            client = ClientTrace(real_client, log)
            original_preview = kwargs.get("preview_callback")
            original_progress = kwargs.get("progress_callback")
            first_preview: set[str] = set()

            def preview(frame):
                if frame.sensor_name not in first_preview:
                    first_preview.add(frame.sensor_name)
                    log.emit("preview.first", sensor=frame.sensor_name, frame=frame.frame,
                             modules=module_state())
                if original_preview:
                    original_preview(frame)

            def progress(value):
                log.emit("runner.progress", percent=value)
                if original_progress:
                    original_progress(value)

            kwargs.update(
                client=client,
                capture_interval=args.capture_interval,
                traffic_manager_port=args.traffic_manager_port,
                register=not args.no_write,
                scenario_path=args.scenario or kwargs.get("scenario_path"),
                preview_callback=preview,
                progress_callback=progress,
                seed=args.seed,
            )
            with trace_output_resolution(run_carla_simulation, log):
                result = run_carla_simulation(**kwargs)
            log.emit("runner.return", result=result)
            return result

        window.simulation_runner = traced_runner
        original_start_postprocessing = window._start_postprocessing

        def traced_start_postprocessing(output_dir):
            log.emit("postprocess.transition.begin", output_dir=output_dir,
                     modules=module_state(), qt_thread_id=int(QThread.currentThreadId()))
            if args.postprocessing == "skip":
                def processor(*values):
                    log.emit("postprocess.skip.worker", arguments=values, modules=module_state())
            else:
                log.emit("postprocess.import.begin", modules=module_state())
                from image_tools import post_process
                log.emit("postprocess.import.end", modules=module_state())

                def processor(*values):
                    log.emit("postprocess.worker.entry", arguments=values, modules=module_state())
                    result = post_process(*values)
                    log.emit("postprocess.worker.return", modules=module_state())
                    return result

            window.postprocessing_processor = processor
            original_start_postprocessing(output_dir)
            log.emit("postprocess.transition.return", worker_created=window.postprocessing_worker is not None,
                     modules=module_state(), qt_thread_id=int(QThread.currentThreadId()))

        window._start_postprocessing = traced_start_postprocessing
        window.simulation_finished.connect(
            lambda result: (state.__setitem__("finished_signal", True),
                            log.emit("qt.simulation_finished", result=result,
                                     qt_thread_id=int(QThread.currentThreadId()))))
        window.simulation_failed.connect(
            lambda message: (state.__setitem__("failed", True),
                             log.emit("qt.simulation_failed", message=message,
                                      qt_thread_id=int(QThread.currentThreadId()))))
        window.postprocessing_failed.connect(
            lambda message: log.emit("qt.postprocessing_failed", message=message))
        window.show()

        def request_stop(reason):
            log.emit("qt.stop.request", reason=reason, qt_thread_id=int(QThread.currentThreadId()))
            if window.simulation_worker is not None:
                window.stop_scenario()

        if args.stop_after is not None:
            QTimer.singleShot(int(args.stop_after * 1000), lambda: request_stop("stop-after"))

        poll = QTimer(window)
        poll.setInterval(50)

        def inspect_state():
            if window.simulation_worker is None and window.postprocessing_worker is None:
                log.emit("qt.workers.idle", start_enabled=window.start_action.isEnabled())
                poll.stop()
                window.close()
                app.quit()

        poll.timeout.connect(inspect_state)
        poll.start()

        def timeout():
            state["timed_out"] = True
            request_stop("watchdog")

        QTimer.singleShot(int(args.timeout * 1000), timeout)
        log.emit("qt.start_scenario", qt_thread_id=int(QThread.currentThreadId()))
        QTimer.singleShot(0, window.start_scenario)
        exit_code = app.exec()
        log.emit("qt.event_loop.return", code=exit_code, state=state, modules=module_state())
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        return 1 if state["failed"] or state["timed_out"] else exit_code
    except BaseException as error:
        log.emit("process.failure", error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        log.emit("process.exit", modules=module_state())


if __name__ == "__main__":
    raise SystemExit(main())
