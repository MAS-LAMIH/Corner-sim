"""Primary CARLA capture pipeline used by the CornerSim GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
import random
import math
import threading
import time
from typing import Any
from dataclasses import dataclass

import carla
import numpy as np

from cornersim.dataset import SampleWriter
from cornersim.runtime import CarlaRuntime
from cornersim.scenario import load_scenario, seed_everything
from cornersim.synchronization import FrameSynchronizer, collect_frame

SENSOR_NAMES = ("rgb", "semantic_segmentation", "instance_segmentation")
STATIC_KEYWORDS = (
    "barrel", "bin", "clothcontainer", "container", "glasscontainer", "box", "trashbag", "colacan",
    "garbage", "platformgarbage", "trashcan", "bench", "gardenlamp", "pergola", "plasticchair",
    "plastictable", "slide", "swing", "table", "trampoline", "barbeque", "clothesline", "doghouse",
    "gnome", "wateringcan", "haybale", "plantpot", "plasticbag", "shoppingbag", "shoppingcart",
    "shoppingtrolley", "briefcase", "guitarcase", "travelcase", "helmet", "mobile", "purse",
    "barrier", "cone", "ironplank", "warning", "brokentile", "dirtdebris", "foodcart", "kiosk_01",
    "fountain", "maptable", "advertisement", "streetsign", "busstop", "atm", "mailbox",
    "streetfountain", "vendingmachine", "calibrator",
)


class ActorInfo:
    def __init__(self, actor: Any):
        self.transform = actor.get_transform()
        self.bounding_box = actor.bounding_box
        self.id = actor.id
        self.name = actor.type_id + " " + " ".join(str(tag) for tag in actor.semantic_tags)
        self.type = actor.type_id


@dataclass(frozen=True)
class PreviewFrame:
    """Owned BGRA sensor bytes safe to transfer across Qt threads."""

    sensor_name: str
    frame: int
    width: int
    height: int
    raw_data: bytes


def sensor_callback(sensor_data: Any, sensor_queue: Queue, sensor_name: str,
                    preview_callback: Any = None) -> None:
    """Queue every measurement; sampling decisions are made after same-frame aggregation."""
    if sensor_name == "semantic_segmentation":
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)
    if preview_callback is not None:
        preview_callback(PreviewFrame(sensor_name, sensor_data.frame, sensor_data.width,
                                      sensor_data.height, bytes(sensor_data.raw_data)))
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


def _relative_transform(reference: Any, location: list[float], orientation: list[float]) -> Any:
    yaw = math.radians(reference.rotation.yaw)
    x = -location[0] * math.cos(yaw) + location[1] * math.sin(yaw)
    y = location[0] * math.sin(yaw) + location[1] * math.cos(yaw)
    return carla.Transform(
        carla.Location(x=reference.location.x + x, y=reference.location.y + y, z=location[2] + 1),
        carla.Rotation(pitch=0, yaw=reference.rotation.yaw + orientation[1], roll=0),
    )


def _apply_scenario(path: str | os.PathLike[str], world: Any, blueprints: Any,
                    runtime: CarlaRuntime, ego: Any) -> list[Any]:
    scenario = load_scenario(path)
    spectator_action = scenario.actions[0]
    if spectator_action["type"] != "spectator":
        raise ValueError("the first scenario action must configure the spectator")
    reference = carla.Transform(carla.Location(*spectator_action["location"]),
                                carla.Rotation(*spectator_action["orientation"]))
    world.get_spectator().set_transform(reference)
    ego.set_autopilot(False)
    ego.set_transform(reference)
    actors: dict[str, Any] = {}
    for action in scenario.actions[1:]:
        kind = action["type"]
        if kind == "spawn_vehicle":
            candidates = list(blueprints.filter("vehicle.audi*")) or list(blueprints.filter("vehicle.*"))
            if not candidates:
                raise RuntimeError("scenario requires a vehicle blueprint")
            actor = runtime.own(world.spawn_actor(random.choice(candidates), _relative_transform(
                reference, action["location"], action["orientation"])))
            control = carla.VehicleControl(throttle=float(action.get("speed", 0)))
            actor.apply_control(control)
            actors[action["vehicle_id"]] = actor
        elif kind == "spawn_pedestrian":
            candidates = list(blueprints.filter("walker.pedestrian.*"))
            if not candidates:
                raise RuntimeError("scenario requires a pedestrian blueprint")
            actor = runtime.own(world.spawn_actor(random.choice(candidates), _relative_transform(
                reference, action["location"], action["orientation"])))
            actor.apply_control(carla.WalkerControl(speed=float(action.get("speed", 0))))
            actors[action["pedestrian_id"]] = actor
        elif kind == "change_vehicle_direction":
            actors[action["vehicle_id"]].apply_control(carla.VehicleControl(
                steer=float(action["direction"][0]), throttle=float(action["direction"][1])))
        elif kind == "pedestrian_jump":
            actor = actors[action["pedestrian_id"]]
            actor.set_transform(_relative_transform(reference, action["location"], action["orientation"]))
            actor.apply_control(carla.WalkerControl(speed=float(action.get("speed", 0))))
    return list(actors.values())


def _object_snapshot(world: Any, camera: Any, spawned_objects: list[Any], sensor_params: dict[str, Any],
                     frame: int) -> dict[str, Any]:
    transform = camera.get_transform()
    camera_location = transform.location
    forward = transform.get_forward_vector()
    objects = [ActorInfo(actor) for actor in spawned_objects]
    objects.extend(world.get_environment_objects())
    filtered = []
    for item in objects:
        name = item.name.lower()
        location = item.transform.location
        direction = location - camera_location
        distance = location.distance(camera_location)
        if distance >= 100 or forward.dot(direction) <= 0 or not any(key in name for key in STATIC_KEYWORDS):
            continue
        vertices = item.bounding_box.get_world_vertices(carla.Transform())
        filtered.append({
            "id": item.id, "name": item.name, "type": item.type, "distance": distance,
            "bounding_box": {
                "location": vars_xyz(item.bounding_box.location), "extent": vars_xyz(item.bounding_box.extent),
                "vertices": [vars_xyz(vertex) for vertex in vertices],
            },
            "transform": {"location": vars_xyz(location), "rotation": vars_rotation(item.transform.rotation)},
        })
    return {
        "frame": frame,
        "camera_transform": {"location": vars_xyz(transform.location), "rotation": vars_rotation(transform.rotation)},
        "camera_parameters": sensor_params["rgb"]["attributes"],
        "filtered_objects": filtered,
    }


def vars_xyz(value: Any) -> dict[str, float]:
    return {axis: float(getattr(value, axis)) for axis in ("x", "y", "z")}


def vars_rotation(value: Any) -> dict[str, float]:
    return {axis: float(getattr(value, axis)) for axis in ("pitch", "yaw", "roll")}


def run_carla_simulation(rgb_label: Any = None, semantic_label: Any = None, instance_label: Any = None,
                         register: bool = True, progress_bar: Any = None, image_width: int = 600,
                         image_height: int = 400, max_tick: int = 200, output_dir: str | os.PathLike[str] = "images",
                         seed: int = 0, stop_event: threading.Event | None = None,
                         capture_interval: int = 20, client: Any = None,
                         scenario_path: str | os.PathLike[str] | None = None,
                         pause_event: threading.Event | None = None,
                         preview_callback: Any = None, progress_callback: Any = None) -> dict[str, Any]:
    """Run one deterministic, synchronized experiment and always restore CARLA state."""
    # Widget arguments are retained for API compatibility but deliberately ignored:
    # CARLA callbacks must never touch Qt objects.
    del image_width, image_height, rgb_label, semantic_label, instance_label, progress_bar
    if max_tick < 1 or capture_interval < 1:
        raise ValueError("max_tick and capture_interval must be positive")
    seed_everything(seed)
    stop_event = stop_event or threading.Event()
    pause_event = pause_event or threading.Event()
    client = client or carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager()
    if hasattr(traffic_manager, "set_random_device_seed"):
        traffic_manager.set_random_device_seed(seed)
    blueprints = world.get_blueprint_library()
    sensor_params = {
        name: {"type": sensor_type, "attributes": {"image_size_x": 1820, "image_size_y": 1240,
                                                     "fov": 90, "sensor_tick": 0.0}}
        for name, sensor_type in (("rgb", "sensor.camera.rgb"),
                                  ("semantic_segmentation", "sensor.camera.semantic_segmentation"),
                                  ("instance_segmentation", "sensor.camera.instance_segmentation"))
    }
    queue: Queue = Queue()
    synchronizer = FrameSynchronizer(SENSOR_NAMES, max_pending_frames=8)
    writer = SampleWriter(output_dir) if register else None
    captured: list[int] = []

    with CarlaRuntime(world, fps=10.0, traffic_manager=traffic_manager) as runtime:
        ego_bp = blueprints.find("vehicle.mercedes.sprinter")
        ego_transform = carla.Transform(carla.Location(x=-110.291763, y=97.093193, z=3.002939),
                                        carla.Rotation(pitch=-8.006648, yaw=66.636604, roll=0.000058))
        ego = runtime.own(world.spawn_actor(ego_bp, ego_transform))
        ego.set_autopilot(True)
        world.get_spectator().set_transform(ego_transform)
        sensors = []
        for name, params in sensor_params.items():
            blueprint = blueprints.find(params["type"])
            for key, value in params["attributes"].items():
                blueprint.set_attribute(key, str(value))
            sensor = runtime.own(world.spawn_actor(blueprint, carla.Transform(carla.Location(x=2.5, z=2.2)),
                                                   attach_to=ego))
            sensor.listen(lambda image, sensor_name=name: sensor_callback(
                image, queue, sensor_name, preview_callback))
            sensors.append(sensor)
        camera = sensors[0]
        scenario_actors = _apply_scenario(scenario_path, world, blueprints, runtime, ego) if scenario_path else []
        props = list(blueprints.filter("static.prop.*"))
        if not props:
            raise RuntimeError("CARLA map exposes no static.prop blueprints")
        spawned_objects = list(scenario_actors)
        for _ in range(100):
            actor = runtime.own(world.spawn_actor(random.choice(props), carla.Transform(carla.Location(
                x=random.uniform(-200, 200), y=random.uniform(-200, 200), z=0.5))))
            spawned_objects.append(actor)

        for tick_index in range(max_tick):
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break
            frame = world.tick()
            synchronized = collect_frame(queue, synchronizer, frame, timeout=12.0)
            if tick_index % capture_interval == 0:
                snapshot = _object_snapshot(world, camera, spawned_objects, sensor_params, frame)
                if writer is not None:
                    writer.write(frame, synchronized.measurements, snapshot)
                captured.append(frame)
            if progress_callback is not None:
                progress_callback(int((tick_index + 1) / max_tick * 100))
    return {"captured_frames": captured, "dropped_frames": synchronizer.dropped_frames,
            "stopped": stop_event.is_set(), "output_dir": str(Path(output_dir).resolve())}


def main() -> None:
    run_carla_simulation()


if __name__ == "__main__":
    main()
