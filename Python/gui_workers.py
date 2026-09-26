"""Non-Qt background workers for the CornerSim GUI.

Workers publish plain Python values to :class:`WorkerEventBus`.  The optional
``wake_socket`` is an ordinary Python socket; writing to it wakes the GUI's
``QSocketNotifier`` without retaining or invoking a QObject from a worker (or a
CARLA sensor callback) thread.
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


class WorkerEventBus:
    """Thread-safe event queue with an optional non-Qt wake-up socket."""

    def __init__(self, wake_socket=None):
        self.queue = SimpleQueue()
        self.wake_socket = wake_socket

    def publish(self, source, event_type, value):
        self.queue.put((source, event_type, value))
        if self.wake_socket is not None:
            try:
                self.wake_socket.send(b"\0")
            except BlockingIOError:
                # One unread byte is enough to wake the GUI.  The queued event is
                # not lost when the socket buffer is already full.
                pass


def _publish_event(event_bus, source, event_type, value):
    event_bus.publish(source, event_type, value)


class SimulationWorker(threading.Thread):
    def __init__(self, length=200, output_dir="images", seed=0, scenario_path=None,
                 runner=None, outcome=None, event_bus=None):
        super().__init__(name="CornerSimSimulation", daemon=False)
        self.scenario_length = length
        self.output_dir = output_dir
        self.seed = seed
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.scenario_path = scenario_path
        self.runner = runner or run_carla_simulation
        self.outcome = outcome or WorkerOutcome()
        self.event_bus = event_bus or WorkerEventBus()
        self.event_queue = self.event_bus.queue

    def run(self):
        try:
            self.outcome.result = self.runner(
                max_tick=self.scenario_length,
                output_dir=self.output_dir,
                seed=self.seed,
                stop_event=self.stop_event,
                scenario_path=self.scenario_path,
                pause_event=self.pause_event,
                preview_callback=partial(_publish_event, self.event_bus, self, "preview"),
                progress_callback=partial(_publish_event, self.event_bus, self, "progress"),
            )
        except Exception as error:
            self.outcome.error = str(error)
        finally:
            self.event_bus.publish(self, "finished", None)

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()


class PostProcessingWorker(threading.Thread):
    def __init__(self, batch_folder, catalog_json_filename, processor=None, outcome=None,
                 event_bus=None):
        super().__init__(name="CornerSimPostProcessing", daemon=False)
        self.batch_folder = batch_folder
        self.catalog_json_filename = catalog_json_filename
        self.processor = processor
        self.outcome = outcome or WorkerOutcome()
        self.event_bus = event_bus or WorkerEventBus()
        self.event_queue = self.event_bus.queue

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
            self.event_bus.publish(self, "finished", None)
