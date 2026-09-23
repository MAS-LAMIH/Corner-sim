"""Exception-safe lifecycle management for a CARLA experiment."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class CarlaRuntime(AbstractContextManager["CarlaRuntime"]):
    """Own world settings and actors created during one simulation run.

    Sensors are stopped before every actor is destroyed, actors are destroyed in
    reverse creation order, and the exact settings object read on entry is restored.
    The class intentionally uses duck typing so lifecycle behavior is unit-testable
    without importing CARLA.
    """

    def __init__(self, world: Any, *, fps: float, traffic_manager: Any = None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.world = world
        self.fps = fps
        self.traffic_manager = traffic_manager
        self.original_settings: Any = None
        self.original_tm_sync: bool | None = None
        self.actors: list[Any] = []

    def __enter__(self) -> "CarlaRuntime":
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        self.world.apply_settings(settings)
        if self.traffic_manager is not None:
            getter = getattr(self.traffic_manager, "get_synchronous_mode", None)
            self.original_tm_sync = getter() if getter else False
            self.traffic_manager.set_synchronous_mode(True)
        return self

    def own(self, actor: Any) -> Any:
        self.actors.append(actor)
        return actor

    def close(self) -> list[Exception]:
        errors: list[Exception] = []
        for actor in reversed(self.actors):
            try:
                if getattr(actor, "is_alive", True) and hasattr(actor, "stop"):
                    actor.stop()
            except Exception as error:  # cleanup must continue
                errors.append(error)
        for actor in reversed(self.actors):
            try:
                if getattr(actor, "is_alive", True):
                    actor.destroy()
            except Exception as error:  # cleanup must continue
                errors.append(error)
        self.actors.clear()
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(bool(self.original_tm_sync))
            except Exception as error:
                errors.append(error)
        if self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception as error:
                errors.append(error)
        return errors

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        cleanup_errors = self.close()
        if exc is None and cleanup_errors:
            raise RuntimeError(f"CARLA cleanup failed: {cleanup_errors!r}") from cleanup_errors[0]
        return False
