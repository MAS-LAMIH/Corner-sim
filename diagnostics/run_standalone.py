"""Run the production simulation path without importing PyQt or creating a GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading
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
    parser.add_argument("--output", default=str(REPO_ROOT / "diagnostic-output" / "standalone"))
    parser.add_argument("--scenario")
    parser.add_argument("--log")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = CheckpointLog(args.log)
    stop_event = threading.Event()

    def cancel(signum, _frame):
        log.emit("cancel.requested", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGINT, cancel)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, cancel)

    log.emit("process.start", configuration="standalone", modules=module_state())
    try:
        log.emit("import.carla.begin")
        import carla
        log.emit("import.carla.end", modules=module_state())
        log.emit("import.runner.begin")
        from Synchro3 import run_carla_simulation
        log.emit("import.runner.end", modules=module_state())
        if module_state()["pyqt_loaded"]:
            raise RuntimeError("standalone harness unexpectedly imported PyQt")

        client = ClientTrace(carla.Client(args.host, args.port), log)
        first_preview: set[str] = set()

        def preview(frame):
            if frame.sensor_name not in first_preview:
                first_preview.add(frame.sensor_name)
                log.emit("preview.first", sensor=frame.sensor_name, frame=frame.frame,
                         width=frame.width, height=frame.height)

        def progress(value):
            log.emit("runner.progress", percent=value)

        output = str(Path(args.output).resolve())
        log.emit("output.input_resolve", requested=args.output, resolved=output)
        log.emit("runner.entry", ticks=args.ticks, capture_interval=args.capture_interval,
                 scenario=args.scenario, modules=module_state())
        with trace_output_resolution(run_carla_simulation, log):
            result = run_carla_simulation(
                max_tick=args.ticks,
                capture_interval=args.capture_interval,
                output_dir=output,
                seed=args.seed,
                stop_event=stop_event,
                scenario_path=args.scenario,
                preview_callback=preview,
                progress_callback=progress,
                traffic_manager_port=args.traffic_manager_port,
                register=not args.no_write,
                client=client,
            )
        log.emit("runner.return", result=result, modules=module_state())
        return 0
    except KeyboardInterrupt:
        stop_event.set()
        log.emit("process.keyboard_interrupt")
        return 130
    except BaseException as error:
        log.emit("process.failure", error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        log.emit("process.exit", cancelled=stop_event.is_set(), modules=module_state())


if __name__ == "__main__":
    raise SystemExit(main())
