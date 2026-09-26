"""Structured, dependency-free diagnostics shared by isolation harnesses."""

from __future__ import annotations

import json
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


class CheckpointLog:
    """Write each checkpoint to stdout and an optional JSON-lines log."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path).resolve() if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **details: Any) -> None:
        record = {
            "time_unix": time.time(),
            "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "pid": os.getpid(),
            "python_thread_id": threading.get_ident(),
            "python_thread_name": threading.current_thread().name,
            **details,
        }
        line = json.dumps(record, default=str, sort_keys=True)
        with self._lock:
            print(line, flush=True)
            if self.path:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                    stream.flush()


def module_state() -> dict[str, bool]:
    """Report relevant native modules without importing any of them."""
    return {
        "pyqt_loaded": any(name == "PyQt5" or name.startswith("PyQt5.") for name in sys.modules),
        "cv2_loaded": "cv2" in sys.modules,
        "carla_loaded": "carla" in sys.modules,
    }


@contextmanager
def trace_output_resolution(function: Any, log: CheckpointLog):
    """Trace the exact production source line containing output ``Path.resolve``."""
    source, start_line = inspect.getsourcelines(function)
    target_line = next(
        start_line + offset
        for offset, line in enumerate(source)
        if '"output_dir": str(Path(output_dir).resolve())' in line
    )
    previous = sys.gettrace()
    entered = False

    def tracer(frame, event, value):
        nonlocal entered
        if frame.f_code is function.__code__:
            if event == "line" and frame.f_lineno == target_line and not entered:
                entered = True
                log.emit("runner.output_resolve.begin", source_line=target_line,
                         output_dir=frame.f_locals.get("output_dir"))
            elif event == "return" and entered:
                log.emit("runner.output_resolve.return", source_line=target_line)
            elif event == "exception" and entered:
                error_type, error, _ = value
                log.emit("runner.output_resolve.exception", source_line=target_line,
                         error_type=getattr(error_type, "__name__", str(error_type)), error=repr(error))
        return tracer

    sys.settrace(tracer)
    try:
        yield
    finally:
        sys.settrace(previous)


class ActorTrace:
    def __init__(self, actor: Any, log: CheckpointLog):
        object.__setattr__(self, "_actor", actor)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actor, name)

    def __getattribute__(self, name: str) -> Any:
        # Defining the tracing method must not make non-sensor actors appear to
        # implement CARLA's optional stop() API to CarlaRuntime.
        if name == "stop":
            actor = object.__getattribute__(self, "_actor")
            if not hasattr(actor, "stop"):
                raise AttributeError("wrapped actor has no stop method")
        return object.__getattribute__(self, name)

    def listen(self, callback: Any) -> Any:
        self._log.emit("sensor.callback.register", actor_id=getattr(self._actor, "id", None),
                       actor_type=getattr(self._actor, "type_id", None))

        first = True

        def traced_callback(data: Any) -> None:
            nonlocal first
            if first:
                first = False
                self._log.emit("sensor.first_callback", actor_id=getattr(self._actor, "id", None),
                               frame=getattr(data, "frame", None))
            callback(data)

        return self._actor.listen(traced_callback)

    def stop(self) -> Any:
        self._log.emit("actor.stop", actor_id=getattr(self._actor, "id", None))
        return self._actor.stop()

    def destroy(self) -> Any:
        self._log.emit("actor.destroy", actor_id=getattr(self._actor, "id", None))
        return self._actor.destroy()


class WorldTrace:
    def __init__(self, world: Any, log: CheckpointLog):
        self._world = world
        self._log = log
        self._first_tick = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)

    def get_settings(self) -> Any:
        settings = self._world.get_settings()
        self._log.emit("world.settings.read",
                       synchronous_mode=getattr(settings, "synchronous_mode", None),
                       fixed_delta_seconds=getattr(settings, "fixed_delta_seconds", None))
        return settings

    def apply_settings(self, settings: Any) -> Any:
        self._log.emit("world.settings.apply",
                       synchronous_mode=getattr(settings, "synchronous_mode", None),
                       fixed_delta_seconds=getattr(settings, "fixed_delta_seconds", None))
        return self._world.apply_settings(settings)

    def spawn_actor(self, blueprint: Any, transform: Any, *args: Any, **kwargs: Any) -> ActorTrace:
        attach_to = kwargs.get("attach_to")
        if isinstance(attach_to, ActorTrace):
            kwargs["attach_to"] = attach_to._actor
        self._log.emit("actor.spawn.begin", blueprint=getattr(blueprint, "id", str(blueprint)),
                       attached=attach_to is not None)
        actor = self._world.spawn_actor(blueprint, transform, *args, **kwargs)
        self._log.emit("actor.spawn.end", actor_id=getattr(actor, "id", None),
                       actor_type=getattr(actor, "type_id", None))
        return ActorTrace(actor, self._log)

    def tick(self, *args: Any, **kwargs: Any) -> Any:
        frame = self._world.tick(*args, **kwargs)
        if self._first_tick:
            self._first_tick = False
            self._log.emit("world.first_tick", frame=frame)
        return frame


class TrafficManagerTrace:
    def __init__(self, manager: Any, log: CheckpointLog, port: int):
        self._manager = manager
        self._log = log
        self._port = port

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def set_synchronous_mode(self, enabled: bool) -> Any:
        self._log.emit("carla.traffic_manager.synchronous_mode", port=self._port,
                       enabled=bool(enabled))
        return self._manager.set_synchronous_mode(enabled)

    def set_random_device_seed(self, seed: int) -> Any:
        self._log.emit("carla.traffic_manager.seed", port=self._port, seed=seed)
        return self._manager.set_random_device_seed(seed)


class ClientTrace:
    """Transparent CARLA client proxy logging lifecycle boundaries."""

    def __init__(self, client: Any, log: CheckpointLog):
        self._client = client
        self._log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def set_timeout(self, seconds: float) -> Any:
        self._log.emit("carla.timeout.set", seconds=seconds)
        return self._client.set_timeout(seconds)

    def get_world(self) -> WorldTrace:
        self._log.emit("carla.world.begin")
        world = self._client.get_world()
        self._log.emit("carla.world.end", map_name=getattr(world.get_map(), "name", None))
        return WorldTrace(world, self._log)

    def get_trafficmanager(self, port: int) -> Any:
        self._log.emit("carla.traffic_manager.begin", port=port)
        manager = self._client.get_trafficmanager(port)
        self._log.emit("carla.traffic_manager.end", port=port,
                       manager_type=type(manager).__name__)
        return TrafficManagerTrace(manager, self._log, port)
