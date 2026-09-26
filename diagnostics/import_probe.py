"""Probe native dependency combinations in one fresh Python process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common import CheckpointLog, module_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", choices=("pyqt", "opencv", "carla", "combined"))
    parser.add_argument("--exercise-opencv-gui", action="store_true",
                        help="create/destroy a native OpenCV window; do not use on headless hosts")
    parser.add_argument("--connect-carla", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = CheckpointLog(args.log)
    log.emit("process.start", configuration=args.configuration, modules=module_state())
    try:
        app = widget = None
        if args.configuration in {"pyqt", "combined"}:
            log.emit("import.pyqt.begin")
            from PyQt5.QtCore import QCoreApplication, QEvent, QThread, QTimer
            from PyQt5.QtWidgets import QApplication, QWidget
            log.emit("import.pyqt.end", modules=module_state())
            app = QApplication.instance() or QApplication(sys.argv[:1])
            widget = QWidget()
            log.emit("pyqt.widget.created", affinity_is_gui=widget.thread() is app.thread(),
                     qt_thread_id=int(QThread.currentThreadId()))
            widget.destroyed.connect(
                lambda: log.emit("pyqt.widget.destroyed", qt_thread_id=int(QThread.currentThreadId())))

        if args.configuration in {"opencv", "combined"}:
            log.emit("import.opencv.begin")
            import cv2
            import numpy as np
            build_gui = next((line.strip() for line in cv2.getBuildInformation().splitlines()
                              if line.strip().startswith("GUI:")), "GUI: unknown")
            log.emit("import.opencv.end", modules=module_state(), build_gui=build_gui)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            log.emit("opencv.compute.exercised")
            if args.exercise_opencv_gui:
                log.emit("opencv.gui.create.begin")
                cv2.namedWindow("CornerSim import probe", cv2.WINDOW_NORMAL)
                cv2.imshow("CornerSim import probe", image)
                cv2.waitKey(50)
                cv2.destroyAllWindows()
                log.emit("opencv.gui.destroy.end")

        if args.configuration in {"carla", "combined"}:
            log.emit("import.carla.begin")
            import carla
            log.emit("import.carla.end", modules=module_state())
            client = carla.Client(args.host, args.port)
            client.set_timeout(3.0)
            log.emit("carla.client.created")
            if args.connect_carla:
                log.emit("carla.world.begin")
                world = client.get_world()
                log.emit("carla.world.end", map_name=world.get_map().name)

        if app is not None:
            widget.show()
            QTimer.singleShot(250, widget.close)
            QTimer.singleShot(300, app.quit)
            log.emit("pyqt.event_loop.begin")
            app.exec()
            log.emit("pyqt.event_loop.end")
            widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
        log.emit("process.success", modules=module_state())
        return 0
    except BaseException as error:
        log.emit("process.failure", error=repr(error), traceback=traceback.format_exc(),
                 modules=module_state())
        return 1
    finally:
        log.emit("process.exit", modules=module_state())


if __name__ == "__main__":
    raise SystemExit(main())
