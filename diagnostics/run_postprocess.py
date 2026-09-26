"""Isolate the legacy post-processing path with and without a Qt event loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import threading
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
for entry in (str(REPO_ROOT), str(PYTHON_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from diagnostics.common import CheckpointLog, module_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("standalone-main", "standalone-thread",
                                         "qt-import", "qt-thread"))
    parser.add_argument("--dataset", help="captured dataset directory")
    parser.add_argument("--copy-to", help="copy dataset here before processing; must not exist")
    parser.add_argument("--catalog", default=str(PYTHON_DIR / "environment_object.json"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = CheckpointLog(args.log)
    log.emit("process.start", configuration=f"postprocess-{args.mode}", modules=module_state())
    try:
        dataset = Path(args.dataset).resolve() if args.dataset else None
        if args.copy_to:
            if dataset is None:
                raise ValueError("--copy-to requires --dataset")
            destination = Path(args.copy_to).resolve()
            log.emit("dataset.copy.begin", source=dataset, destination=destination)
            shutil.copytree(dataset, destination)
            dataset = destination
            log.emit("dataset.copy.end", destination=destination)
        if args.mode not in {"qt-import"} and dataset is None:
            raise ValueError(f"{args.mode} requires --dataset")

        app = QTimer = QThread = QCoreApplication = QEvent = None
        if args.mode.startswith("qt-"):
            log.emit("import.pyqt.begin", modules=module_state())
            from PyQt5.QtCore import QCoreApplication, QEvent, QThread, QTimer
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv[:1])
            log.emit("import.pyqt.end", modules=module_state(),
                     qt_thread_id=int(QThread.currentThreadId()))

        log.emit("postprocess.import.begin", modules=module_state())
        from image_tools import post_process
        log.emit("postprocess.import.end", modules=module_state())
        if args.mode == "qt-import":
            QTimer.singleShot(250, app.quit)
            log.emit("qt.event_loop.begin")
            app.exec()
            log.emit("qt.event_loop.return")
            return 0

        outcome = {"error": None, "complete": False}

        def invoke():
            log.emit("postprocess.entry", dataset=dataset, modules=module_state())
            try:
                post_process(str(dataset), str(Path(args.catalog).resolve()))
                outcome["complete"] = True
                log.emit("postprocess.return", modules=module_state())
            except BaseException as error:
                outcome["error"] = repr(error)
                log.emit("postprocess.failure", error=repr(error), traceback=traceback.format_exc())

        if args.mode == "standalone-main":
            invoke()
        elif args.mode == "standalone-thread":
            worker = threading.Thread(target=invoke, name="PostProcessDiagnostic", daemon=True)
            worker.start()
            worker.join(args.timeout)
            if worker.is_alive():
                log.emit("watchdog.timeout", seconds=args.timeout)
                return 2
        else:
            worker = threading.Thread(target=invoke, name="PostProcessDiagnostic", daemon=True)
            timer = QTimer()
            timer.setInterval(25)

            def poll():
                if not worker.is_alive():
                    timer.stop()
                    app.quit()

            timer.timeout.connect(poll)
            timer.start()
            QTimer.singleShot(int(args.timeout * 1000), app.quit)
            worker.start()
            log.emit("qt.event_loop.begin", qt_thread_id=int(QThread.currentThreadId()))
            app.exec()
            log.emit("qt.event_loop.return", worker_alive=worker.is_alive())
            timer.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            if worker.is_alive():
                log.emit("watchdog.timeout", seconds=args.timeout)
                return 2
            worker.join()
        return 1 if outcome["error"] or not outcome["complete"] else 0
    except BaseException as error:
        log.emit("process.failure", error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        log.emit("process.exit", modules=module_state())


if __name__ == "__main__":
    raise SystemExit(main())
