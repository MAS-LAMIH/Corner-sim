import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import carla
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing
import os

def save_result_to_json(result, output_file):
    result_json = [{
        'id': int(id_),
        'min_x': int(min_x),
        'min_y': int(min_y),
        'max_x': int(max_x),
        'max_y': int(max_y),
        'label': label
    } for id_, min_x, min_y, max_x, max_y, label in result]

    with open(output_file, 'w') as json_file:
        json.dump(result_json, json_file, indent=4)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def process_instance_segmentation(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    background_color = [0, 0, 0, 255]
    unique_colors, unique_counts = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0, return_counts=True)

    bounding_boxes = []

    with ProcessPoolExecutor() as executor:
        for color, count in zip(unique_colors, unique_counts):
            if np.array_equal(color, background_color):
                continue

            color_mask = np.all(image_array == color, axis=2)
            nonzero_indices = np.transpose(np.nonzero(color_mask))

            if len(nonzero_indices) == 0:
                continue

            min_x, min_y = np.min(nonzero_indices, axis=0)
            max_x, max_y = np.max(nonzero_indices, axis=0)

            bounding_boxes.append((count, min_x, min_y, max_x, max_y))

    return bounding_boxes

def process_instance_semantic_segmentation_(instance_image_path, semantic_image_path, class_mapping):
    instance_image = Image.open(instance_image_path)
    semantic_image = Image.open(semantic_image_path)

    instance_array = np.array(instance_image)
    semantic_array = np.array(semantic_image)[:, :, :-1]  # Remove alpha channel

    unique_colors = np.unique(instance_array.reshape(-1, instance_array.shape[2]), axis=0)
    result = []
    id_counter =0

    def process_color(color, id_counter):
        bounding_boxes =[]
        specific= []
        if np.array_equal(color, [0, 0, 0]):
            return []

        color_mask = np.all(instance_array == color, axis=2)

        labeled_mask = cv2.connectedComponents((color_mask * 255).astype(np.uint8))[1]
        num_labels = labeled_mask.max()

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0  # Initialize bounding box coordinates
        region_mask = None
        for label_id in range(1, num_labels + 1):
            region_mask = (labeled_mask == label_id)

            if region_mask.sum() < 4:  # Adjust the threshold as needed
                continue

            region_indices = np.argwhere(region_mask)
            min_y, min_x = min(min_y, region_indices[:, 0].min()), min(min_x, region_indices[:, 1].min())
            max_y, max_x = max(max_y, region_indices[:, 0].max()), max(max_x, region_indices[:, 1].max())

        if min_x == float('inf') or min_y == float('inf') or max_x == 0 or max_y == 0:
            return []

        # Get all the pixel indices of the object mask
        object_pixel_indices = np.transpose(np.nonzero(color_mask))

        # Extract the colors of the object pixels from the semantic image
        object_semantic_colors = semantic_array[object_pixel_indices[:, 0], object_pixel_indices[:, 1]]

        # Find the most common color within the object's semantic colors
        unique_semantic_colors, counts = np.unique(object_semantic_colors, axis=0, return_counts=True)
        object_rgb_color = unique_semantic_colors[np.argmax(counts)]

        label = next((key for key, value in class_mapping.items() if value == tuple(object_rgb_color.tolist())), "unknown")
        if label != "unknown":
            return [(id_counter, min_x, min_y, max_x, max_y, label)]
        if (not (region_mask is None)) and ( label.lower()  in specific):
            contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < 4:  # Adjust the threshold as needed
                    continue
                bounding_boxes.append((id_counter, x, y, x + w, y + h, label))
                id_counter+=1

        return bounding_boxes

    with ThreadPoolExecutor() as executor:
        id_c = id_counter + 1  # Initialize id_counter
        results = executor.map(process_color, unique_colors, [id_c] * len(unique_colors))
        result = [item for sublist in results for item in sublist]
        id_counter = id_c
    return result


def process_instance_semantic_segmentation(instance_image_path, semantic_image_path, class_mapping):
    instance_image = Image.open(instance_image_path)
    semantic_image = Image.open(semantic_image_path)

    instance_array = np.array(instance_image)
    semantic_array = np.array(semantic_image)[:, :, :-1]  # Remove alpha channel

    unique_colors = np.unique(instance_array.reshape(-1, instance_array.shape[2]), axis=0)
    result = []
    id_counter = 0

    def process_color(color, id_counter):
        bounding_boxes = []
        specific = []
        if np.array_equal(color, [0, 0, 0]):
            return []

        color_mask = np.logical_and.reduce(instance_array == color, axis=2)

        labeled_mask = cv2.connectedComponents((color_mask * 255).astype(np.uint8))[1]
        num_labels = labeled_mask.max()

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0  # Initialize bounding box coordinates

        for label_id in range(1, num_labels + 1):
            region_mask = (labeled_mask == label_id)

            if region_mask.sum() < 4:  # Adjust the threshold as needed
                continue

            region_indices = np.argwhere(region_mask)
            min_x = min(min_x, region_indices[:, 1].min())
            min_y = min(min_y, region_indices[:, 0].min())
            max_x = max(max_x, region_indices[:, 1].max())
            max_y = max(max_y, region_indices[:, 0].max())

            # Extract subregions and compute bounding boxes
            subregion_bounding_boxes = []
            subregion_colors = np.unique(semantic_array[region_mask], axis=0)
            for subregion_color in subregion_colors:
                label = next((key for key, value in class_mapping.items() if value == tuple(subregion_color.tolist())), "unknown")
                subregion_mask = np.all(semantic_array == subregion_color, axis=2)
                subregion_contours, _ = cv2.findContours(
                    (subregion_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for subregion_contour in subregion_contours:
                    x, y, w, h = cv2.boundingRect(subregion_contour)
                    if w * h < 4:  # Adjust the threshold as needed
                        continue
                    subregion_bounding_boxes.append((id_counter, x, y, x + w, y + h, label))
                    id_counter += 1

            bounding_boxes.extend(subregion_bounding_boxes)

        if min_x == float('inf') or min_y == float('inf') or max_x == 0 or max_y == 0:
            return []

        return bounding_boxes


    with ThreadPoolExecutor() as executor:
        id_c = id_counter + 1  # Initialize id_counter
        results = executor.map(process_color, unique_colors, [id_c] * len(unique_colors))
        result = [item for sublist in results for item in sublist]
        id_counter = id_c
    return result


#CLASS_MAPPING = {    'car': (0, 0, 142), 'sky': (70, 130, 180)}
CLASS_MAPPING = {    'unlabeled': (0, 0, 0),
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
def draw_bounding_boxes(image_path, bounding_boxes):
    base_image = Image.open(image_path)
    draw = ImageDraw.Draw(base_image)

    for id_, min_x, min_y, max_x, max_y, label in bounding_boxes:
        color = CLASS_MAPPING[label]
        label_text = f"{label} {id_}"

        # Draw the bounding box
        draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=2)

        # Draw the label background
        font = ImageFont.load_default()  # You can choose a different font if needed
        text_width, text_height = draw.textsize(label_text, font=font)
        draw.rectangle(
            [min_x, min_y, min_x + text_width, min_y + text_height],
            fill=color
        )

        # Draw the label text
        draw.text((min_x, min_y), label_text, fill=(255, 255, 255), font=font)

    return base_image

def draw_bounding_boxes2(image_path, bounding_boxes):
    base_image = Image.open(image_path)
    draw = ImageDraw.Draw(base_image)

    for bb in bounding_boxes:
        id_ = bb['id']
        name= bb['name']
        min_x = bb['min_x']
        min_y = bb['min_y']
        max_x = bb['max_x']
        max_y = bb['max_y']
        base_label = bb['base_label']
        other_labels = bb['other_labels']
        if True: #base_label == "static" or base_label == "dynamic":            
            # Get color based on base_label
            color = CLASS_MAPPING[base_label]
            #label_text = f"{base_label} {id_} : {', '.join(other_labels)}"
            label_text = f"{base_label} :{name}"

            # Draw the bounding box
            draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=2)

            # Draw the label background
            font = ImageFont.load_default()  # You can choose a different font if needed
            text_width, text_height = draw.textsize(label_text, font=font)
            draw.rectangle(
                [min_x, min_y, min_x + text_width, min_y + text_height],
                fill=color
            )

            # Draw the label text
            draw.text((min_x, min_y), label_text, fill=(255, 255, 255), font=font)

    return base_image

def match(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2, threshold=0.5):
    r1_area = max(0, max_x1 - min_x1) * max(0, max_y1 - min_y1)
    r2_area = max(0, max_x2 - min_x2) * max(0, max_y2 - min_y2)
    if r1_area == 0 or r2_area == 0:
        return False
    intersection_area = calculate_intersection_area(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2)
    #(intersection_max_x - intersection_min_x) * (intersection_max_y - intersection_min_y)
    
    return (intersection_area > threshold * r1_area and intersection_area > threshold * r2_area)
def calculate_intersection_area(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2):
    #r1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    #r2_area = (max_x2 - min_x2) * (max_y2 - min_y2)
    
    #max_area = max(r1_area, r2_area)
    #min_area = min(r1_area, r2_area)
    #if max_area > 3 * min_area:
    #    return False  # Huge difference in areas
    
    intersection_min_x = max(min_x1, min_x2)
    intersection_min_y = max(min_y1, min_y2)
    intersection_max_x = min(max_x1, max_x2)
    intersection_max_y = min(max_y1, max_y2)
    if intersection_max_x <= intersection_min_x or intersection_max_y <= intersection_min_y:
        return 0  # No intersection
    #print(f"min_xi={intersection_min_x}")
    #print(f"min_yi={intersection_min_y}")
    #print(f"max_xi={intersection_max_x}")
    #print(f"max_yi={intersection_max_y}")
    
    intersection_area = (intersection_max_x - intersection_min_x) * (intersection_max_y - intersection_min_y)
    
    return intersection_area
    #(intersection_area > threshold * r1_area and intersection_area > threshold * r2_area)


def extract_keyword(carla_object_label):
    parts = carla_object_label.split('.')
    last_part = parts[-1]
    keyword = ''.join([c for c in last_part if not c.isdigit()])
    return keyword

def preprocess_catalog_tree(catalog):
    keywords_dict = {}

    def traverse(node, labels):
        if isinstance(node, dict):
            for key, value in node.items():
                keyw = ''.join([c for c in key.replace('alternate', '') if not c.isdigit()])
                new_labels = labels + [keyw.strip()]
                traverse(value, new_labels)
        else:
            if labels:
                keyword = extract_keyword(node)
                if keyword is not None:
                    keywords_dict[keyword] = labels

    traverse(catalog, [])

    return keywords_dict

def filter(json_array, string_label, catalog_keywords=None):
    filtered_objects = []
    
    for obj in json_array:
        obj['labels'] =[]
        if string_label.lower() == "static" or string_label.lower() == "dynamic" or string_label.lower() == "other":
            for keyword, labels in catalog_keywords.items():
                if keyword in obj['name'].lower():
                    unique_labels = list(set(labels))  # Remove duplicates
                    obj['labels'] = unique_labels
                filtered_objects.append(obj)
        
    
    return filtered_objects
def refine_bbs_worker(bb, sensor_info, catalog_keywords, K, world_to_camera):
    refined_bbox = {
        'name': "",
        'id': bb['id'],
        'min_x': bb['min_x'],
        'min_y': bb['min_y'],
        'max_x': bb['max_x'],
        'max_y': bb['max_y'],
        'base_label': bb['label'],
        'other_labels': []
    }
    #area= (bb['max_y']-bb['min_y'])*(bb['max_x']-bb['min_x'])
    w= sensor_info['camera_parameters']['image_size_x']
    h=sensor_info['camera_parameters']['image_size_y']
    best_distance = float('inf')
    best_matched_bbs = []
    best_name = ""

    filtered_objects = filter(sensor_info['filtered_objects'], bb['label'], catalog_keywords)
    bb_area = (bb['max_x'] - bb['min_x']) * (bb['max_y'] - bb['min_y'])
    best_intersection_area = 0.2* bb_area

    for sensor_bb in filtered_objects:
        # Extract corners and projected_corners as before
        sensor_id = sensor_bb['id']
            
        # Extract corners from bounding box info
        bbox_center = np.array([sensor_bb['bounding_box']['location']['x'],
                                sensor_bb['bounding_box']['location']['y'],
                                sensor_bb['bounding_box']['location']['z']])
        
        bbox_extent = np.array([sensor_bb['bounding_box']['extent']['x'],
                                sensor_bb['bounding_box']['extent']['y'],
                                sensor_bb['bounding_box']['extent']['z']])
        

        corners = sensor_bb['bounding_box']['vertices']

        projected_corners = [get_image_point(
            carla.Location(x=c['x'], y=c['y'], z=c['z']),
            K,
            world_to_camera
        ) for c in corners]
                
        #min_x_proj = min([p[0] for p in projected_corners])
        #min_y_proj = min([p[1] for p in projected_corners])
        #max_x_proj = max([p[0] for p in projected_corners])
        #max_y_proj = max([p[1] for p in projected_corners])
        #corners = [
        #    bbox_center + bbox_extent * np.array([-1, -1, -1]),
        #    bbox_center + bbox_extent * np.array([1, -1, -1]),
        #    bbox_center + bbox_extent * np.array([-1, 1, -1]),
        #    bbox_center + bbox_extent * np.array([1, 1, -1]),
        #    bbox_center + bbox_extent * np.array([-1, -1, 1]),
        #    bbox_center + bbox_extent * np.array([1, -1, 1]),
        #    bbox_center + bbox_extent * np.array([-1, 1, 1]),
        #    bbox_center + bbox_extent * np.array([1, 1, 1])
        #]
        
        #projected_corners = [get_image_point(carla.Location(x=c[0], y=c[1], z=c[2]), K, world_to_camera) for c in corners]
        #max(0,min(w,p[0]))
        min_x = min([p[0] for p in projected_corners])
        min_y = min([p[1] for p in projected_corners])
        max_x = max([p[0] for p in projected_corners])
        max_y = max([p[1] for p in projected_corners])
        if not ((max_x - min_x) >w or (max_y-min_y)>h):

            min_x= max(0,min(w,min_x))
            min_y= max(0,min(h,min_y))
            max_x= max(0,min(w,max_x))
            max_y= max(0,min(h,max_y))
            ar= (max_x-min_x)*(max_y-min_y)
            intersection_area = calculate_intersection_area(
                min_x, min_y, max_x, max_y,
                bb['min_x'], bb['min_y'], bb['max_x'], bb['max_y']
            )
            distance = sensor_bb['distance']
            
            if intersection_area>0.3*bb_area and intersection_area> 0.3*ar :

                if intersection_area > best_intersection_area:
                    best_intersection_area = intersection_area
                    best_distance = distance
                    best_matched_bbs = sensor_bb['labels']
                    best_name = sensor_bb['name']
                elif intersection_area == best_intersection_area and distance < best_distance:
                    best_distance = distance
                    best_matched_bbs = sensor_bb['labels']
                    best_name = sensor_bb['name']

    refined_bbox['other_labels'] = best_matched_bbs
    refined_bbox['name'] = best_name

        #sensor_bbox_area = (max_x - min_x) * (max_y - min_y)

        #if match(min_x, min_y, max_x, max_y, bb['min_x'], bb['min_y'], bb['max_x'], bb['max_y']):
        #    matched_bbs = sensor_bb['labels']
        #    name= sensor_bb['name']



    #refined_bbox['other_labels'] = matched_bbs
    #refined_bbox['name'] = name

    return refined_bbox

def refine_bbs(sensor_info_path, bounding_boxes_path, output_path, catalog_keywords=None):
    with open(sensor_info_path, 'r') as sensor_file:
        sensor_info = json.load(sensor_file)

    with open(bounding_boxes_path, 'r') as bb_file:
        bounding_boxes = json.load(bb_file)

    camera_transform = sensor_info['camera_transform']
    camera_params = sensor_info['camera_parameters']
    sensor_width = int(camera_params['image_size_x'])
    sensor_height = int(camera_params['image_size_y'])
    fov = float(camera_params['fov'])
    world_to_camera = np.array(carla.Transform(carla.Location(camera_transform['location']['x'], camera_transform['location']['y'], camera_transform['location']['z']),
                                               carla.Rotation(camera_transform['rotation']['pitch'], camera_transform['rotation']['yaw'], camera_transform['rotation']['roll'])).get_inverse_matrix())

    K = build_projection_matrix(sensor_width, sensor_height, fov)


    refined_bounding_boxes = []
    
    with ProcessPoolExecutor() as executor:
        refine_worker = partial(refine_bbs_worker, sensor_info=sensor_info, catalog_keywords=catalog_keywords, K=K, world_to_camera=world_to_camera)
        refined_bounding_boxes = list(executor.map(refine_worker, bounding_boxes))

    with open(output_path, 'w') as output_file:
        json.dump(refined_bounding_boxes, output_file, indent=4)
    
    return refined_bounding_boxes



def process_batch(batch_folder, keywords_dict):
    rgb_folder_path = os.path.join(batch_folder, 'rgb')
    for filename in os.listdir(rgb_folder_path):
        if filename.endswith(".png") and filename.startswith("image_"):
            frame_number = filename.split("_")[1].split(".")[0]

            # Load images
            rgb_image_path = os.path.join(batch_folder, 'rgb', filename)
            semantic_image_path = os.path.join(batch_folder, 'semantic_segmentation', filename)
            instance_image_path = os.path.join(batch_folder, 'instance_segmentation', filename)
            #rgb_image = Image.open(rgb_image_path)
            #semantic_image = Image.open(semantic_image_path)
            #instance_image = Image.open(instance_image_path)

            # Process instance and semantic segmentation images
            result = process_instance_semantic_segmentation_(instance_image_path, semantic_image_path, CLASS_MAPPING)

            # Save processed results to JSON
            result_json = f'output_{frame_number}.json'
            save_result_to_json(result, os.path.join(batch_folder, result_json))

            # Example: Refine bounding boxes using JSON files
            refine_result = refine_bbs(os.path.join(batch_folder,'simulation_objects', f'image_{frame_number}.json'), os.path.join(batch_folder, result_json), os.path.join(batch_folder, 'labels' ,f'refined_{result_json}'), keywords_dict)

            # Example: Draw bounding boxes on RGB image
            result_image = draw_bounding_boxes2(rgb_image_path, refine_result)
            result_image_path = os.path.join(batch_folder,'labels', f'output_image_{frame_number}.png')
            result_image.save(result_image_path)

def post_process(batch_folder, catalog_json_filename):
    with open(catalog_json_filename, 'r') as json_file:
        json_data = json.load(json_file)

    catalog_tree = json_data['Props catalogue']
    keywords_dict = preprocess_catalog_tree(catalog_tree)

    process_batch(batch_folder, keywords_dict)

if __name__ == '__main__':


    
    multiprocessing.freeze_support()

    batch_folder = r'images'
    catalog_json_filename = 'environment_object.json'
    with open(catalog_json_filename, 'r') as json_file:
        json_data = json.load(json_file)

    catalog_tree = json_data['Props catalogue']
    keywords_dict = preprocess_catalog_tree(catalog_tree)

    process_batch(batch_folder, keywords_dict)

# Call the function with your input paths and output path
#refine_bbs('sensor_and_objects_info.json', 'bounding_boxes.json', 'refined_bounding_boxes.json')
# Read the JSON tree from a file
    #catalog_json_filename = 'environment_object.json'
    #with open(catalog_json_filename, 'r') as json_file:
    #    json_data = json.load(json_file)
#
   # # Call the preprocessing function
    #catalog_tree = json_data['Props catalogue']
    #keywords_dict = preprocess_catalog_tree(catalog_tree)

    #print(keywords_dict)
    #rgb_image_path ="image_rgb_1095.png"

    #instance_image_path = "image_instance_1095.png"

    #semantic_image_path = "image_semantic_1095.png"

    #result =  process_instance_semantic_segmentation(instance_image_path, semantic_image_path, CLASS_MAPPING)

    #ll = extract_keyword("static.prop.box02")
    #print(ll)
    #print(result)
    #result_image = draw_bounding_boxes(rgb_image_path, result)
    #result_image.save("output_image.png")
    #save_result_to_json(result, 'output.json')
    #result2=refine_bbs('sensor_and_objects_info_1095.json', 'output.json', 'refined_bounding_boxes.json', keywords_dict)
    #print(result2)
    #result_image = draw_bounding_boxes2(rgb_image_path, result2)
    #result_image.save("output_image2.png")
