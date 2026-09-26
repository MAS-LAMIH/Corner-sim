"""Qt workers that communicate with the GUI exclusively through queued signals."""

from __future__ import annotations

import threading

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from Synchro3 import run_carla_simulation


class SimulationWorker(QObject):
    preview_ready = pyqtSignal(object)
    progress_changed = pyqtSignal(int)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, length=200, output_dir="images", seed=0, scenario_path=None, runner=None):
        super().__init__()
        self.scenario_length = length
        self.output_dir = output_dir
        self.seed = seed
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.scenario_path = scenario_path
        self.runner = runner or run_carla_simulation

    @pyqtSlot()
    def run(self):
        try:
            result = self.runner(
                max_tick=self.scenario_length,
                output_dir=self.output_dir,
                seed=self.seed,
                stop_event=self.stop_event,
                scenario_path=self.scenario_path,
                pause_event=self.pause_event,
                preview_callback=self.preview_ready.emit,
                progress_callback=self.progress_changed.emit,
            )
            self.succeeded.emit(result)
        except Exception as error:
            self.failed.emit(str(error))

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()


class PostProcessingWorker(QObject):
    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, batch_folder, catalog_json_filename, processor=None):
        super().__init__()
        self.batch_folder = batch_folder
        self.catalog_json_filename = catalog_json_filename
        self.processor = processor

    @pyqtSlot()
    def run(self):
        try:
            if self.processor is None:
                from image_tools import post_process
                processor = post_process
            else:
                processor = self.processor
            processor(self.batch_folder, self.catalog_json_filename)
            self.succeeded.emit()
        except Exception as error:
            self.failed.emit(str(error))
