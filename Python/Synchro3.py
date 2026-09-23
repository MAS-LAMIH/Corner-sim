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


def update_progress_bar(current_tick: int, progress_bar: Any, max_ticks: int) -> None:
    if progress_bar is not None and current_tick < max_ticks:
        progress_bar.setValue(int(current_tick / max_ticks * 100))


def update_camera_view(image: Any, view: Any, desired_width: int = 600, desired_height: int = 400,
                       city_scape_convert: bool = False) -> None:
    if view is None:
        return
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPixmap
    if city_scape_convert:
        image.convert(carla.ColorConverter.CityScapesPalette)
    pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[..., :3]
    q_image = QImage(bytes(pixels.data), image.width, image.height, QImage.Format_RGB888).rgbSwapped()
    view.setPixmap(QPixmap.fromImage(q_image.scaled(desired_width, desired_height, Qt.KeepAspectRatio)))


def sensor_callback(sensor_data: Any, sensor_queue: Queue, sensor_name: str, rgb_label: Any = None,
                    semantic_label: Any = None, instance_label: Any = None) -> None:
    """Queue every measurement; sampling decisions are made after same-frame aggregation."""
    targets = {"rgb": rgb_label, "semantic_segmentation": semantic_label, "instance_segmentation": instance_label}
    if sensor_name == "semantic_segmentation":
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)
    update_camera_view(sensor_data, targets[sensor_name], 600, 300)
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
                         pause_event: threading.Event | None = None) -> dict[str, Any]:
    """Run one deterministic, synchronized experiment and always restore CARLA state."""
    del image_width, image_height  # kept for public API compatibility
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
                image, queue, sensor_name, rgb_label, semantic_label, instance_label))
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
            update_progress_bar(tick_index + 1, progress_bar, max_tick)
    return {"captured_frames": captured, "dropped_frames": synchronizer.dropped_frames,
            "stopped": stop_event.is_set(), "output_dir": str(Path(output_dir).resolve())}


def main() -> None:
    run_carla_simulation()


if __name__ == "__main__":
    main()
