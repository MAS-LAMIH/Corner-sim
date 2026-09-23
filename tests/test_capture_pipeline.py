import json
import sys
from types import SimpleNamespace

sys.path.insert(0, "Python")
import Synchro3  # noqa: E402


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, other):
        value = self - other
        return (value.x ** 2 + value.y ** 2 + value.z ** 2) ** 0.5


class Rotation:
    def __init__(self, pitch=0, yaw=0, roll=0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location, self.rotation = location or Vector(), rotation or Rotation()

    def get_forward_vector(self):
        return SimpleNamespace(dot=lambda other: other.x)


class FakeImage:
    width, height = 4, 3

    def __init__(self, frame):
        self.frame = frame
        self.converted = False

    def convert(self, converter):
        self.converted = True

    def save_to_disk(self, path):
        from PIL import Image
        Image.new("RGB", (self.width, self.height), "black").save(path)


class FakeActor:
    def __init__(self, type_id="irrelevant"):
        self.type_id, self.semantic_tags, self.id = type_id, [], id(self)
        self.bounding_box = SimpleNamespace(location=Vector(), extent=Vector(), get_world_vertices=lambda transform: [])
        self.transform = Transform()
        self.callback = None
        self.is_alive = True
        self.stopped = False

    def listen(self, callback): self.callback = callback
    def stop(self): self.stopped = True
    def destroy(self): self.is_alive = False
    def set_autopilot(self, value): pass
    def set_transform(self, value): self.transform = value
    def get_transform(self): return self.transform
    def apply_control(self, value): pass


class FakeBlueprint:
    def __init__(self, type_id): self.type_id = type_id
    def set_attribute(self, key, value): pass


class FakeBlueprints:
    def find(self, type_id): return FakeBlueprint(type_id)
    def filter(self, pattern):
        return [FakeBlueprint("static.prop.test" if pattern.startswith("static") else "vehicle.audi.test")]


class FakeWorld:
    def __init__(self):
        self.frame, self.actors, self.sensors = 0, [], []
        self.original = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None, marker="original")
        self.applied = []
        self.spectator = FakeActor()

    def get_settings(self):
        source = self.applied[-1] if self.applied else self.original
        return SimpleNamespace(**vars(source))
    def apply_settings(self, settings): self.applied.append(settings)
    def get_blueprint_library(self): return FakeBlueprints()
    def get_spectator(self): return self.spectator
    def get_environment_objects(self): return []
    def spawn_actor(self, blueprint, transform, attach_to=None):
        actor = FakeActor(blueprint.type_id)
        actor.transform = transform
        self.actors.append(actor)
        if blueprint.type_id.startswith("sensor"):
            self.sensors.append(actor)
        return actor
    def tick(self):
        self.frame += 1
        for sensor in self.sensors:
            sensor.callback(FakeImage(self.frame))
        return self.frame


class FakeTrafficManager:
    def __init__(self): self.sync = False
    def get_synchronous_mode(self): return self.sync
    def set_synchronous_mode(self, value): self.sync = value
    def set_random_device_seed(self, seed): self.seed = seed


class FakeClient:
    def __init__(self): self.world, self.traffic = FakeWorld(), FakeTrafficManager()
    def set_timeout(self, timeout): self.timeout = timeout
    def get_world(self): return self.world
    def get_trafficmanager(self): return self.traffic


def fake_carla():
    return SimpleNamespace(
        Location=Vector, Rotation=Rotation, Transform=Transform,
        VehicleControl=lambda **kwargs: kwargs, WalkerControl=lambda **kwargs: kwargs,
        ColorConverter=SimpleNamespace(CityScapesPalette="palette"),
    )


def test_actual_capture_pipeline_writes_only_complete_same_frame_samples_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(Synchro3, "carla", fake_carla())
    client = FakeClient()
    result = Synchro3.run_carla_simulation(
        max_tick=2, capture_interval=1, output_dir=tmp_path, client=client, seed=4)
    assert result["captured_frames"] == [1, 2]
    for frame in result["captured_frames"]:
        metadata = json.loads((tmp_path / "metadata" / f"image_{frame}.json").read_text())
        objects = json.loads((tmp_path / "simulation_objects" / f"image_{frame}.json").read_text())
        assert metadata == {"complete": True, "frame": frame,
                            "streams": ["rgb", "semantic_segmentation", "instance_segmentation"]}
        assert objects["frame"] == frame
    assert client.world.applied[-1].marker == "original"
    assert client.traffic.sync is False
    assert all(not actor.is_alive for actor in client.world.actors)
    assert all(sensor.stopped for sensor in client.world.sensors)
