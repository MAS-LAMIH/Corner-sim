"""Non-Qt background workers for the CornerSim GUI.

No QObject or Qt-bound callable crosses into a worker or CARLA callback thread. The
GUI consumes plain Python events with a GUI-owned QTimer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import partial
from queue import SimpleQueue
from typing import Any

from Synchro3 import run_carla_simulation


@dataclass
class WorkerOutcome:
    result: Any = None
    error: str | None = None


def _enqueue_event(event_queue, event_type, value):
    event_queue.put((event_type, value))


class SimulationWorker(threading.Thread):
    def __init__(self, length=200, output_dir="images", seed=0, scenario_path=None,
                 runner=None, outcome=None):
        super().__init__(name="CornerSimSimulation", daemon=False)
        self.scenario_length = length
        self.output_dir = output_dir
        self.seed = seed
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.scenario_path = scenario_path
        self.runner = runner or run_carla_simulation
        self.outcome = outcome or WorkerOutcome()
        self.event_queue = SimpleQueue()

    def run(self):
        try:
            self.outcome.result = self.runner(
                max_tick=self.scenario_length,
                output_dir=self.output_dir,
                seed=self.seed,
                stop_event=self.stop_event,
                scenario_path=self.scenario_path,
                pause_event=self.pause_event,
                preview_callback=partial(_enqueue_event, self.event_queue, "preview"),
                progress_callback=partial(_enqueue_event, self.event_queue, "progress"),
            )
        except Exception as error:
            self.outcome.error = str(error)
        finally:
            self.event_queue.put(("finished", None))

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()


class PostProcessingWorker(threading.Thread):
    def __init__(self, batch_folder, catalog_json_filename, processor=None, outcome=None):
        super().__init__(name="CornerSimPostProcessing", daemon=False)
        self.batch_folder = batch_folder
        self.catalog_json_filename = catalog_json_filename
        self.processor = processor
        self.outcome = outcome or WorkerOutcome()
        self.event_queue = SimpleQueue()

    def run(self):
        try:
            if self.processor is None:
                from image_tools import post_process
                processor = post_process
            else:
                processor = self.processor
            processor(self.batch_folder, self.catalog_json_filename)
            self.outcome.result = True
        except Exception as error:
            self.outcome.error = str(error)
        finally:
            self.event_queue.put(("finished", None))
