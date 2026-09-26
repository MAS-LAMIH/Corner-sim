from types import SimpleNamespace

import pytest

from cornersim.runtime import CarlaRuntime


class FakeWorld:
    def __init__(self):
        self.original = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None, marker="original")
        self.applied = []

    def get_settings(self):
        current = self.original if not self.applied else self.applied[-1]
        return SimpleNamespace(**vars(current))

    def apply_settings(self, settings):
        self.applied.append(settings)


class AliasedSettingsWorld(FakeWorld):
    """Models bindings that return the same mutable settings proxy every time."""

    def __init__(self):
        super().__init__()
        self.proxy = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)

    def get_settings(self):
        return self.proxy

    def apply_settings(self, settings):
        self.applied.append((settings.synchronous_mode, settings.fixed_delta_seconds))


class FakeActor:
    def __init__(self, name, events, sensor=False, fail_destroy=False):
        self.name, self.events, self.sensor, self.fail_destroy = name, events, sensor, fail_destroy
        self.is_alive = True

    def stop(self):
        if self.sensor:
            self.events.append(f"stop:{self.name}")

    def destroy(self):
        self.events.append(f"destroy:{self.name}")
        self.is_alive = False
        if self.fail_destroy:
            raise RuntimeError("destroy failed")


class FakeTrafficManager:
    def __init__(self):
        self.sync = False
        self.history = []

    def get_synchronous_mode(self):
        return self.sync

    def set_synchronous_mode(self, value):
        self.sync = value
        self.history.append(value)


def test_runtime_restores_settings_and_destroys_in_reverse_after_exception():
    world, traffic, events = FakeWorld(), FakeTrafficManager(), []
    with pytest.raises(RuntimeError, match="experiment failed"):
        with CarlaRuntime(world, fps=20, traffic_manager=traffic) as runtime:
            runtime.own(FakeActor("vehicle", events))
            runtime.own(FakeActor("camera", events, sensor=True))
            assert world.applied[-1].synchronous_mode
            assert world.applied[-1].fixed_delta_seconds == 0.05
            raise RuntimeError("experiment failed")
    assert events == ["stop:camera", "destroy:camera", "destroy:vehicle"]
    assert world.applied[-1].marker == "original"
    assert traffic.history == [True, False]


def test_cleanup_continues_when_one_actor_fails():
    world, events = FakeWorld(), []
    with pytest.raises(RuntimeError, match="cleanup failed"):
        with CarlaRuntime(world, fps=10) as runtime:
            runtime.own(FakeActor("vehicle", events))
            runtime.own(FakeActor("camera", events, sensor=True, fail_destroy=True))
    assert events == ["stop:camera", "destroy:camera", "destroy:vehicle"]
    assert world.applied[-1].marker == "original"


def test_aliased_settings_values_are_restored_after_cancellation():
    world, traffic = AliasedSettingsWorld(), FakeTrafficManager()
    with CarlaRuntime(world, fps=10, traffic_manager=traffic):
        assert world.proxy.synchronous_mode is True
        assert world.proxy.fixed_delta_seconds == 0.1
    assert world.proxy.synchronous_mode is False
    assert world.proxy.fixed_delta_seconds is None
    assert world.applied[-1] == (False, None)
    assert traffic.sync is False


def test_cleanup_failure_is_reported_even_during_body_exception():
    world, events = FakeWorld(), []
    with pytest.raises(RuntimeError, match="cleanup failed.*body failed"):
        with CarlaRuntime(world, fps=10) as runtime:
            runtime.own(FakeActor("camera", events, sensor=True, fail_destroy=True))
            raise ValueError("body failed")


def test_runtime_refuses_unrestorable_traffic_manager_and_restores_world():
    world = AliasedSettingsWorld()
    traffic = SimpleNamespace(set_synchronous_mode=lambda value: None)
    with pytest.raises(RuntimeError, match="refusing unsafe mutation"):
        with CarlaRuntime(world, fps=10, traffic_manager=traffic):
            pass
    assert world.proxy.synchronous_mode is False
    assert world.proxy.fixed_delta_seconds is None
