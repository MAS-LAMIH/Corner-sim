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
        self.original_synchronous_mode: bool | None = None
        self.original_fixed_delta_seconds: float | None = None
        self.original_tm_sync: bool | None = None
        self.actors: list[Any] = []

    def __enter__(self) -> "CarlaRuntime":
        settings = self.world.get_settings()
        # Store immutable values before modifying the settings object. Some CARLA
        # bindings/fakes may return aliased proxy objects from get_settings().
        self.original_synchronous_mode = bool(settings.synchronous_mode)
        self.original_fixed_delta_seconds = settings.fixed_delta_seconds
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        self.world.apply_settings(settings)
        if self.traffic_manager is not None:
            getter = getattr(self.traffic_manager, "get_synchronous_mode", None)
            if not callable(getter):
                # Restore the world before refusing to enter a lifecycle we cannot
                # safely reverse.
                settings = self.world.get_settings()
                settings.synchronous_mode = self.original_synchronous_mode
                settings.fixed_delta_seconds = self.original_fixed_delta_seconds
                self.world.apply_settings(settings)
                raise RuntimeError("Traffic Manager cannot report its synchronous mode; refusing unsafe mutation")
            self.original_tm_sync = bool(getter())
            try:
                self.traffic_manager.set_synchronous_mode(True)
            except Exception:
                settings = self.world.get_settings()
                settings.synchronous_mode = self.original_synchronous_mode
                settings.fixed_delta_seconds = self.original_fixed_delta_seconds
                self.world.apply_settings(settings)
                raise
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
        if self.original_synchronous_mode is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = self.original_synchronous_mode
                settings.fixed_delta_seconds = self.original_fixed_delta_seconds
                self.world.apply_settings(settings)
            except Exception as error:
                errors.append(error)
        return errors

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        cleanup_errors = self.close()
        if cleanup_errors:
            message = f"CARLA cleanup failed: {cleanup_errors!r}"
            if exc is not None:
                message += f" while handling {exc!r}"
            raise RuntimeError(message) from (exc or cleanup_errors[0])
        return False
