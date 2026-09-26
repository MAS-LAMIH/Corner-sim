import numpy as np
from queue import Queue
from collections import namedtuple
import carla
import configparser
import platform
import subprocess
import os
import psutil
import random
import math
import time
import importlib
import sys
#from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt  #QTimer, QThread, pyqtSignal


class _LazyOpenCV:
    """Avoid initializing OpenCV's optional Qt backend during GUI module import."""

    _module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module("cv2")
        return getattr(self._module, name)


cv2 = _LazyOpenCV()

messages = []
CLASS_MAPPING = {
    'unlabeled': (0, 0, 0),
    'building': (70, 70, 70),
    'fence': (100, 40, 40),
    'other': (55, 90, 80),
    'pedestrian': (220, 20, 60),
    'cyclist': (255, 0, 0),
    'pole': (153, 153, 153),
    'roadLines': (157, 234, 50),
    'roads': (128, 64, 128),
    'sidewalks': (244, 35, 232),
    'vegetation': (107, 142, 35),
    'bicycle': (119, 11, 32),
    'bus': (0, 60, 100),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'motorcycle': (0, 0, 230),
    'vehicle': (0, 0, 142),
    'walls': (102, 102, 156),
    'traffic_sign': (220, 220, 0),
    'sky': (70, 130, 180),
    'ground': (81, 0, 81),
    'bridge': (150, 100, 100),
    'rail_track': (230, 150, 140),
    'guard_rail': (180, 165, 180),
    'traffic_light': (250, 170, 30),
    'static': (110, 190, 160),
    'dynamic': (170, 120, 50),
    'water': (45, 60, 150),
    'terrain': (145, 170, 100)
}
multiple_bbox_tags = ['sidewalks', 'vegetation', 'traffic_sign', 'sky', 'traffic_light', 'static', 'dynamic']


def build_projection_matrix(w, h, fov):
    from cornersim.geometry import build_projection_matrix as maintained_projection_matrix
    return maintained_projection_matrix(int(w), int(h), float(fov))


def convert_coordinates(reference_transform, location, orientation):
    reference_location = reference_transform.location
    reference_rotation = reference_transform.rotation

    # Convert local X and Y coordinates to global coordinates
    x = -location[0] * math.cos(math.radians(reference_rotation.yaw)) + location[1] * math.sin(
        math.radians(reference_rotation.yaw))
    y = location[0] * math.sin(math.radians(reference_rotation.yaw)) + location[1] * math.cos(
        math.radians(reference_rotation.yaw))

    # Calculate global location
    global_location = carla.Location(reference_location.x + x, reference_location.y + y, location[2] + 1)

    # Calculate global rotation
    global_pitch = 0
    global_yaw = reference_rotation.yaw + orientation[1]
    global_roll = 0
    global_rotation = carla.Rotation(global_pitch, global_yaw, global_roll)

    return global_location, global_rotation


def spawn_vehicle(world, spectator_transform, location, orientation, speed):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.audi*'))

    spawn_location, spawn_rotation = convert_coordinates(spectator_transform, location, orientation)

    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    print("vehicle is spawned at")
    print(spawn_transform)
    vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    control = carla.VehicleControl()
    control.throttle = speed
    vehicle.apply_control(control)

    return vehicle


def spawn_pedestrian(world, spectator_transform, location, orientation, speed):
    blueprint_library = world.get_blueprint_library()
    pedestrian_bp = random.choice(
        blueprint_library.filter('walker.pedestrian.0009'))  # blueprint_library.find('walker.pedestrian.*')

    spawn_location, spawn_rotation = convert_coordinates(spectator_transform, location, orientation)

    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    print("pedistrian is spawned at")
    print(location)
    print(orientation)
    print(spawn_transform)
    pedestrian = world.spawn_actor(pedestrian_bp, spawn_transform)
    print(pedestrian.get_transform())
    control = carla.WalkerControl()
    control.speed = speed
    pedestrian.apply_control(control)
    pedestrian.set_transform(spawn_transform)

    return pedestrian


def change_vehicle_direction(vehicle, direction):
    control = carla.VehicleControl()
    control.steer = direction[0]
    control.throttle = direction[1]
    vehicle.apply_control(control)


def pedestrian_jump(pedestrian, spectator_transform, location, orientation, speed):
    new_location, new_rotation = convert_coordinates(spectator_transform, location, orientation)

    new_transform = carla.Transform(new_location, new_rotation)
    pedestrian.set_transform(new_transform)

    control = carla.WalkerControl()
    control.speed = speed
    pedestrian.apply_control(control)
    print("pedistrian jump to")
    print(pedestrian.get_transform())


def load_scenario_from_yaml(file_path):
    # Preserve the legacy list return type while enforcing the maintained schema.
    from cornersim.scenario import load_scenario
    return [dict(action) for action in load_scenario(file_path).actions]


def move_spectator(world, location, orientation):
    spectator = world.get_spectator()
    new_transform = carla.Transform(carla.Location(*location), carla.Rotation(*orientation))
    spectator.set_transform(new_transform)
    world.tick()
    print("scenario is at")
    print(spectator.get_transform())


def update_seg_camera_view(image, view, desired_width, desired_height):
    image.convert(carla.ColorConverter.CityScapesPalette)
    np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    np_img = np_img.reshape((image.height, image.width, 4))
    np_img = np_img[..., :3]  # Remove the alpha channel
    # q_image = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
    # q_image = q_image.rgbSwapped()
    # self.semantic_image= image
    # Use the palette to create a QImage
    q_image = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
    # q_image = q_image.rgbSwapped()
    # self.semsegConversion(q_image)
    # q_image.setColorTable(palette_list)
    scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_q_image)
    # pixmap = QtGui.QPixmap.fromImage(q_image)
    view.setPixmap(pixmap)
    return image
    # self.camera_tick +=1
    # self.synchroTick()


def update_rgb_camera_view(image, view, desired_width, desired_height):
    # Convert the Image object to a QImage object
    # new_raw_data = image.raw_data#bytearray(image.raw_data)
    # new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    q_image = QtGui.QImage(array.data, image.width, image.height, QtGui.QImage.Format_RGB32)
    # pixmap = QtGui.QPixmap.fromImage(q_image)
    scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_q_image)
    view.setPixmap(pixmap)
    return get_RGB_DATA(image)


def get_RGB_DATA(image):
    new_raw_data = copyImageData(image)
    # copy.deepcopy(image.raw_data)#bytearray(image.raw_data)
    new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height, fov=image.fov)
    return new_image


def copyImageData(source_image):
    if source_image is None:
        return None

    source_array = np.array(source_image.raw_data)
    # source_array = source_array.reshape((source_image.height, source_image.width, 4))

    # Create a new RGB image
    new_image = np.zeros(source_array.shape,
                         dtype=np.uint8)  # np.zeros((source_image.height, source_image.width, 3), dtype=np.uint8)

    # Copy the pixel values from source_array to new_image
    new_image[:] = source_array[:]
    # new_image[:] = source_array[:, :, :3]
    return new_image


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
synchro_queue = Queue()

ImageObj = namedtuple('ImageObj', ['raw_data', 'width', 'height', 'fov'])

selected_labels=['Bicycle','Bus','Car','Motorcycle','Rider','Pedestrians','traffic_light','traffic_sign','Truck']
bb_labels= {
    #'Any' : carla.CityObjectLabel.Any,
    'Bicycle' : carla.CityObjectLabel.Bicycle,
    #'Bridge' : carla.CityObjectLabel.Bridge,
    'Buildings' : carla.CityObjectLabel.Buildings,
    'Bus' : carla.CityObjectLabel.Bus,
    'Car' : carla.CityObjectLabel.Car,
    'Dynamic' : carla.CityObjectLabel.Dynamic,
    'Fences' : carla.CityObjectLabel.Fences,
    'Ground' : carla.CityObjectLabel.Ground,
    'GuardRail' : carla.CityObjectLabel.GuardRail,
    'Motorcycle' : carla.CityObjectLabel.Motorcycle,
    #'NONE' : carla.CityObjectLabel.NONE,
    'Other' : carla.CityObjectLabel.Other,
    'Pedestrians' : carla.CityObjectLabel.Pedestrians,
    #'Poles' : carla.CityObjectLabel.Poles,
    'RailTrack' : carla.CityObjectLabel.RailTrack,
    'Rider' : carla.CityObjectLabel.Rider,
    'RoadLines' : carla.CityObjectLabel.RoadLines,
    'Roads' : carla.CityObjectLabel.Roads,
    'Sidewalks' : carla.CityObjectLabel.Sidewalks,
    'Sky' : carla.CityObjectLabel.Sky,
    'Static' : carla.CityObjectLabel.Static,
    'Terrain' : carla.CityObjectLabel.Terrain,
    'traffic_light' : carla.CityObjectLabel.TrafficLight,
    'traffic_sign' : carla.CityObjectLabel.TrafficSigns,
    #'Train' : carla.CityObjectLabel.Train,
    'Truck' : carla.CityObjectLabel.Truck,
    'Vegetation' : carla.CityObjectLabel.Vegetation,
    'Walls' : carla.CityObjectLabel.Walls,
    'Water' : carla.CityObjectLabel.Water

}


def max_projected_length(length, distance, K):
    f = K[0, 0]
    l_max = (f * length) / distance
    return l_max


def is_visible(bb, camera_transform, k):
    forward_vec = camera_transform.get_forward_vector()
    bb_direction = bb.location - camera_transform.location
    dist = bb.location.distance(camera_transform.location)
    bb_width = bb.extent.z
    if max_projected_length(bb_width, dist, k) < 5:
        # print(bb)
        # print("too small")
        return False
    dot_product = (forward_vec.x) * (bb_direction.x) + forward_vec.y * (bb_direction.y) + forward_vec.z * (
        bb_direction.z)
    # if dot_product<=0:
    # print(bb)
    # print("behind camera")
    return dot_product > 0


def expand_bb(bounding_boxes):
    for label in ["Bicycle", "Motorcycle"]:
        if label in bounding_boxes:
            for bb in bounding_boxes[label]:
                # Expand the bounding box and append it to the modified_bbs list
                # Add a new entry for "Rider" if it doesn't exist
                if "Rider" not in bounding_boxes:
                    bounding_boxes["Rider"] = []
                # Add a new bounding box for "Rider" with the same location and dimensions
                rider_bb = carla.BoundingBox(carla.Location(bb.location), carla.Vector3D(bb.extent))
                bounding_boxes["Rider"].append(rider_bb)


def process_colors(rgb_image_draw, blended_image, sem_array, multiple_bbox_tags_colors, min_width, min_height,
                   CLASS_MAPPING):
    bounding_boxes = []

    for label, color in CLASS_MAPPING.items():
        mask = np.all(blended_image == color, axis=-1)
        if color in multiple_bbox_tags_colors:
            contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            isolated = False  # is_area_isolated(mask, min_width, min_height)
            if isolated:
                contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                continue

        for contour in contours:
            pixel_coordinates = contour.squeeze(axis=1).tolist()
            if len(pixel_coordinates) > 0:
                if is_big_enough(pixel_coordinates, min_width, min_height):
                    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(
                        coord[0] for coord in pixel_coordinates)
                    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(
                        coord[1] for coord in pixel_coordinates)

                    if not is_object_fragmented(blended_image, color, min_x, min_y, max_x, max_y):
                        cv2.rectangle(rgb_image_draw, (min_x, min_y), (max_x, max_y), color, 2)
                        bounding_boxes.append(((min_x, min_y), (max_x, max_y), label))

    return bounding_boxes


def is_object_fragmented(instance_image, class_color, min_x, min_y, max_x, max_y):
    # Crop the instance image within the bounding box
    cropped_image = instance_image[min_y:max_y, min_x:max_x]

    nb_total = (max_y - min_y) * (max_x - min_x)

    # Compute the mask of class color pixels within the cropped image
    mask = np.all(cropped_image == class_color, axis=-1)

    # Find the contours of the class color pixels
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there are multiple contours (areas) for the object
    if len(contours) > 5:

        # Count the number of class color pixels
        class_color_pixels = np.count_nonzero(mask)

        # Count the total number of pixels within the bounding box
        total_pixels = cropped_image.shape[0] * cropped_image.shape[1]

        # Compute the ratio of class color pixels to total pixels
        ratio = class_color_pixels / total_pixels

        # Check if the object is fragmented based on the ratio threshold (70%)
        if ratio < 0.5:
            return True

    # by default, return False
    return False


def is_big_enough(pixel_coordinates, t_width, t_height):
    # return True
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    if ((max_x - min_x + 1) < (t_width)) or ((max_y - min_y + 1) < (t_height)):
        return False

    # Create sets of unique x and y values
    unique_x = set(coord[0] for coord in pixel_coordinates)
    unique_y = set(coord[1] for coord in pixel_coordinates)

    # Iterate over unique x and y values to check if the rectangle can fit
    # Iterate over unique x and y values to check if the rectangle can fit
    for x in unique_x:
        # Skip if the rectangle exceeds the contour's maximum x coordinate
        if x + t_width - 1 > max_x:
            continue
        for y in unique_y:
            # Skip if the rectangle exceeds the contour's maximum y coordinate
            if y + t_height - 1 > max_y:
                continue
            # Check if any pixel coordinates are within the current rectangle
            if any((x <= coord[0] <= x + t_width - 1) and (y <= coord[1] <= y + t_height - 1) for coord in
                   pixel_coordinates):
                return True
    return False


def is_area_isolated(pixel_coordinates, t_width, t_height):
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    if (w < t_width) or (h < t_height):
        return False
    width = (max_x - min_x) / 2
    height = (max_y - min_y) / 2

    # Create a new list of pixel coordinates within the range of min_x, max_x, min_y, max_y
    pixel_coordinates_2 = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1) if
                           (x, y) not in pixel_coordinates]

    # Check if the new area is big enough using the is_big_enough function
    return is_big_enough(pixel_coordinates_2, width, height)


def sensor_callback(world, actor, sensor_data, synchro_queue, sensor_name, K=None):
    if sensor_name == 'rgb_camera':
        transform = actor.get_transform()
        # Calculate distances to camera for each bounding box
        bounding_boxes_with_distances = {
            label: [(bb, bb.location.distance(transform.location)) for bb in world.get_level_bbs(bb_labels[label]) if
                    is_visible(bb, transform, K)]
            for label in selected_labels
        }
        # Sort bounding boxes by distance from camera in place
        sorted_bounding_boxes = {
            label: [bb for bb, _ in sorted(bounding_boxes_with_distances[label], key=lambda x: x[1])]
            for label in selected_labels
        }
        bounding_boxes_3d = sorted_bounding_boxes
        expand_bb(bounding_boxes_3d)
        camera_transform = carla.Transform(transform.location, transform.rotation)
        synchro_queue.put((sensor_data.frame, "bounding_boxes", bounding_boxes_3d))
        synchro_queue.put((sensor_data.frame, "camera_transform", camera_transform))
        # print(sensor_data.frame)

    synchro_queue.put((sensor_data.frame, sensor_name, sensor_data))



config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
carla_path = os.environ.get('CARLA_ROOT', config.get('Carla', 'path', fallback='')).strip()


def check_carla_server():
    print("Attempting to connect to Carla server .. ")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        client.get_world()
        return client
    except Exception as e:
        # print(f"Failed to connect to Carla server: {e}")
        return None


def check_connection_status(client):
    try:
        # Check the connection status by performing a simple operation,
        # such as retrieving the world or a specific actor
        world = client.get_world()


    except Exception as e:
        print("Connection lost:", e)
        # Close the connection or kill the client here
        client.disconnect()
        client = None


def launch_carla_server():
    # launch the Carla server
    if not carla_path:
        raise RuntimeError("CARLA_ROOT or Python/config.ini [Carla] path must be configured to launch CARLA")
    os_name = platform.system()

    print(f"Opening CARLA from path: {carla_path}")
    if os_name == 'Windows':
        path_to_run = os.path.join(carla_path, 'CarlaUE4.exe')
        subprocess.Popen(path_to_run, cwd=carla_path)
    elif os_name == 'Linux':
        path_to_run = os.path.join(carla_path, 'CarlaUE4.sh')
        subprocess.Popen([path_to_run, '-opengl'], cwd=carla_path)
    else:
        print('Unsupported operating system')


def close_carla_server():
    # get the process list
    processes = psutil.process_iter()

    # find all processes associated with CARLA
    carla_processes = []
    for process in processes:
        if process.name().startswith('Carla'):
            carla_processes.append(process)

    # terminate all CARLA processes
    for process in carla_processes:
        process.terminate()

    # wait for all CARLA processes to terminate
    psutil.wait_procs(carla_processes)


def run_scenario(world, scenario_data, spawned_actors, ego):
    # Get the spectator position and orientation from the scenario data
    spectator_position = scenario_data[0]['location']
    spectator_orientation = scenario_data[0]['orientation']

    # Move spectator view towards the specified position
    move_spectator(world, spectator_position, spectator_orientation)

    # Get the spectator transform for relative spawning
    time.sleep(1)
    spectator = world.get_spectator()
    spectator_transform = spectator.get_transform()

    # List to keep track of spawned actors
    # spawned_actors = []
    ego.set_autopilot(False)

    # Move the ego vehicle and cameras to the spectator location and orientation
    ego.set_transform(spectator_transform)
    for event in scenario_data[1:]:
        time.sleep(1)
        event_type = event.get('type')
        if event_type == 'spawn_vehicle':
            vehicle_id = event['vehicle_id']
            vehicle = spawn_vehicle(
                world,
                spectator_transform,
                event['location'],
                event['orientation'],
                event.get('speed', 0)
            )
            spawned_actors.append((vehicle_id, vehicle))  # Add spawned vehicle to the list
        elif event_type == 'spawn_pedestrian':
            pedestrian_id = event['pedestrian_id']
            pedestrian = spawn_pedestrian(
                world,
                spectator_transform,
                event['location'],
                event['orientation'],
                event.get('speed', 0)
            )
            spawned_actors.append((pedestrian_id, pedestrian))  # Add spawned pedestrian to the list
        elif event_type == 'change_vehicle_direction':
            vehicle_id = event['vehicle_id']
            for actor_id, actor in spawned_actors:
                if actor_id == vehicle_id:
                    change_vehicle_direction(actor, event['direction'])
                    break
        elif event_type == 'pedestrian_jump':
            pedestrian_id = event['pedestrian_id']
            for actor_id, actor in spawned_actors:
                if actor_id == pedestrian_id:
                    pedestrian_jump(
                        actor,
                        spectator_transform,
                        event['location'],
                        event['orientation'],
                        event.get('speed', 0)
                    )
                    break

    # Wait for a while to observe the scenario

    # Destroy the spawned actors


def update_bounding_box_view_simple(view, camera, img, semantic_image, bounding_box_set, transform, image_ref, counter,
                                    desired_width, desired_height, bb_id=0):
    json_array = []
    for bounding_box in bounding_box_set:
        bb, label = bounding_box[0], bounding_box[1]
        # for bb,label in bounding_box_set.items():
        object_color = CLASS_MAPPING[label]
        x_min_new, y_min_new, x_max_new, y_max_new = bb
        if not (x_min_new == -1 and y_min_new == -1 and x_max_new == -1 and y_max_new == -1):
            # Draw bounding box using OpenCV's rectangle function
            cv2.rectangle(img, (x_min_new, y_min_new), (x_max_new, y_max_new), object_color, 2)

            # label = 'vehicle'  # replace with the appropriate label for each object type
            cv2.putText(img, label, (x_min_new, y_min_new - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
            json_entry = {"id": bb_id, "label": label, "x_min": int(x_min_new), "y_min": int(y_min_new),
                          "x_max": int(x_max_new), "y_max": int(y_max_new)}
            json_array.append(json_entry)
            bb_id += 1

    # Convert the image back to a QImage object and display it
    qimage = QtGui.QImage(img.data, image_ref.width, image_ref.height, QtGui.QImage.Format_RGB32)
    scaled_qimage = qimage.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
    # pixmap = QtGui.QPixmap.fromImage(qimage)

    view.setPixmap(pixmap)
    counter += 1
    return json_array


def compute_bb_distance(camera_location, corners):
    # corners = bb.get_world_vertices(carla.Transform())

    # Compute the center of the bounding box in world coordinates
    bounding_box_center = carla.Location()
    for corner in corners:
        bounding_box_center.x += corner.x
        bounding_box_center.y += corner.y
        bounding_box_center.z += corner.z
    bounding_box_center.x /= len(corners)
    bounding_box_center.y /= len(corners)
    bounding_box_center.z /= len(corners)
    distance = camera_location.distance(bounding_box_center)
    return distance


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    from cornersim.geometry import project_point
    projected = project_point((loc.x, loc.y, loc.z), K, w2c)
    if projected is None:
        raise ValueError("point is behind the camera or intersects the near plane")
    return np.asarray(projected[:2])


def update_bounding_box_view_3D(view, camera, image, semantic_image, bounding_box_set, transform, image_ref, counter,
                                desired_width, desired_height, self):
    # Convert the Image object to a QImage object
    # Define percentage to reduce bounding box size by
    reduction_percentage = 0.1
    if image is None:
        return []
    json_array = []
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    img = np.frombuffer(copyImageData(image), dtype=np.dtype(
        "uint8"))  # np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
    rgb = np.frombuffer(copyImageData(image_ref), dtype=np.dtype(
        "uint8"))  # np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (image.height, image.width, 4))
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = np.reshape(img, (image.height, image.width, 4))
    # carla.CityObjectLabel.Car
    bb_id = 0
    processed_boxes = {label: [] for label in bounding_box_set.keys()}
    for label, bb_set in bounding_box_set.items():
        for bb in bb_set:
            try:
                corners = bb.get_world_vertices(carla.Transform())
                distance = compute_bb_distance(camera.get_transform().location, corners)
                from cornersim.geometry import project_bbox
                projected_box = project_bbox(
                    [(corner.x, corner.y, corner.z) for corner in corners], self.K, world_2_camera,
                    image.width, image.height, min_area=4.0,
                )
                if projected_box is None:
                    continue
                corners = [get_image_point(corner, self.K, world_2_camera) for corner in corners]
                object_color = CLASS_MAPPING[label]
                object_color = (object_color[2], object_color[1], object_color[0])
                # Check if the bounding box is visible to the camera
                # line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
                # forward_vec = self.vehicle.get_transform().get_forward_vector()
                # bb_direction = bb.location - camera.get_transform().location
                # dot_product = forward_vec.x * bb_direction.x + forward_vec.y * bb_direction.y + forward_vec.z * bb_direction.z
                # if dot_product > 0:
                # if np.dot(forward_vec, bb_direction) > 0:
                # relative_location = bb.location - camera.get_location()
                # ang=camera.get_forward_vector().get_angle(relative_location)
                # if ang < 80 and ang>=0:
                # Define percentage to reduce bounding box size by
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                #    center_point = carla.Location()
                #    for v in verts:
                #        center_point += v
                #    center_point /= len(verts)

                # Calculate new vertices by reducing distance from center point by reduction percentage
                #    new_verts = []
                #    for v in verts:
                #        direction = v - center_point
                #        new_direction = direction - direction * reduction_percentage
                #        new_v = center_point + new_direction
                #        new_verts.append(new_v)

                # Get image coordinates of new vertices
                #    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in new_verts]

                # Use NumPy to calculate min/max corners
                corners = np.array(corners, dtype=int)
                x_min, y_min = int(projected_box.min_x), int(projected_box.min_y)
                x_max, y_max = int(projected_box.max_x), int(projected_box.max_y)
                included = False
                # Extract the region of interest from the semantic image using the bounding box coordinates
                # Assume that 'semantic_image' is a CARLA Image object
                semantic_data = np.frombuffer(semantic_image.raw_data, dtype=np.dtype("uint8"))
                semantic_data = np.reshape(semantic_data, (semantic_image.height, semantic_image.width, 4))
                semantic_data = semantic_data[:, :, :3]
                roi = semantic_data[y_min:y_max, x_min:x_max]

                # Count the number of pixels within the bounding box coordinates that have the correct semantic color
                count = np.sum((roi == object_color).all(axis=2))

                # Compute the total number of pixels within the bounding box coordinates
                total = roi.shape[0] * roi.shape[1]

                # If the ratio of the number of pixels with the correct semantic color to the total number of pixels is greater than or equal to 0.5, process the bounding box

                if total > 0 and count * 2 >= total:

                    for processed_bb in processed_boxes[label]:
                        if x_min >= processed_bb[0] and x_max <= processed_bb[2] and y_min >= processed_bb[
                            1] and y_max <= processed_bb[3] and processed_bb[4] < distance:
                            included = True
                            break
                    if not included:
                        # Process the bounding box if it's not included in any previously processed box
                        processed_boxes[label].append((x_min, y_min, x_max, y_max, distance))
                        edge_array = []
                        # Draw edges of the bounding box into the camera output
                        for edge in edges:
                            try:
                                p1 = get_image_point(verts[edge[0]], self.K, world_2_camera)
                                p2 = get_image_point(verts[edge[1]], self.K, world_2_camera)
                                # Draw the edges into the camera output
                                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), object_color, 1)
                                edge_array.append((p1.tolist(), p2.tolist()))
                            except Exception as e:
                                print(e)
                                continue
                            # label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
                        json_entry = {"id": bb_id, "label": label, "edges": edge_array}
                        json_array.append(json_entry)
                        bb_id += 1
            except Exception as e:
                continue

    # record_tick(json_array,rgb,semantic_image,img)
    # Convert the image back to a QImage object and display it
    qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
    # pixmap = QtGui.QPixmap.fromImage(qimage)

    # Convert the QImage object to a QPixmap object and display it
    # pixmap = QtGui.QPixmap.fromImage(qimage)
    scaled_qimage = qimage.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
    view.setPixmap(pixmap)
    counter += 1
    return json_array


def record_tick(registry, json_array, rgb, semantic_image, img_bb, is_running, is_recording):
    if is_running and is_recording:
        # Create a dictionary with the data to be stored
        # rgb_data=self.get_RGB_DATA(rgb)
        data_dict = {
            "json_array": json_array,
            "rgb": rgb,
            "semantic_image": semantic_image,
            "img_bb": img_bb,
        }
        # Append the data dictionary to the list
        registry.append(data_dict)


class ConsoleLogger:
    def __init__(self, window=None):
        self.terminal = sys.stdout

    def flush(self):
        self.terminal.flush()

    def write(self, message, window=None):
        # Handle the "INFO: ..." messages as desired
        if "connection failed" in message:
            # Do something with the message, such as logging to a file
            # or displaying it in a separate widget
            print("Found a message containing 'specific_word':", message)
            if window is not None:
                try:
                    client = window.client
                    check_connection_status(client)
                except Exception as e:
                    raise
                else:
                    pass
                finally:
                    pass

        if message.startswith("INFO: "):
            # Do something with the message, such as logging to a file
            # or displaying it in a separate widget
            # pass
            messages.append(message)

        # Forward the message to the original stdout stream
        else:
            self.terminal.write(message)
