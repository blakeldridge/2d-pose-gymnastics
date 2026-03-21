from segmentation.background_composition import composite_background
from utils.visualisation import plot_skeleton
from segment_anything import SamPredictor, sam_model_registry
import os
import json
import cv2
import random
import time
import numpy as np

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BACKGROUNDS = os.path.join(DIR, "data/backgrounds/")
BACKGROUND_ANN = os.path.join(DIR, "segmentation/annotations/annotations.json")

def pick_background():
    background_paths = [os.path.join(BACKGROUNDS, f) for f in os.listdir(BACKGROUNDS)]

    return random.choice(background_paths)

def build_dataset(images_dir, annotations_path, background_data, results_dir):
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(DIR, "segmentation/vit-b.pth"))
    predictor = SamPredictor(sam)

    images_converted = 0
    failed_conversions = 0

    result_images = os.path.join(results_dir, "images/")
    result_annotations = os.path.join(results_dir, "annotations.json")

    os.makedirs(result_images, exist_ok=True)

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    for i in range(len(annotations["images"]))[0:5]:
        try:
            filename = annotations["images"][i]["file_name"]
            path = os.path.join(images_dir, filename)
            person_image = cv2.imread(path)
            if person_image is None:
                raise ValueError(f"Failed to load image: {path}")
            image_id = annotations["images"][i]["id"]
            print(f"Converting {path}")

            for j in range(len(annotations["annotations"])):
                if annotations["annotations"][j]["image_id"] == image_id:
                    keypoints = annotations["annotations"][j]["keypoints"]
                    bbox = annotations["annotations"][j]["bbox"]
                    break

            if bbox is None or keypoints is None:
                continue

            bg = background_data[random.randint(0, len(background_data)-1)]
            bg_image = cv2.imread(bg["image"])
            bg_foreground_mask = cv2.imread(bg["foreground_mask"])
            bg_placement_mask = cv2.imread(bg["placement_mask"])
            bg_min_height = bg["min_height"]
            bg_max_height = bg["max_height"]

            start_time = time.time()
            new_image, new_kps, new_bbox = composite_background(person_image, bbox, keypoints, bg_image, bg_placement_mask, bg_foreground_mask, predictor, [bg_min_height, bg_max_height], [-180, 180], 0.1, 0.01)

            annotations["annotations"][i]["keypoints"] = list(map(float, new_kps))
            annotations["annotations"][i]["bbox"] = list(map(float, new_bbox))

            cv2.imwrite(os.path.join(result_images, filename), new_image)
            end_time = time.time()

            print(f"Converted {path}")
            print(f"Time taken : {(end_time - start_time):.2f} secs\n")
            images_converted += 1
        except Exception as e:
            print(f"ERROR : {e}")
            print(f"Failed to convert Image\n")
            failed_conversions += 1

    with open(result_annotations, "w") as f:
        json.dump(annotations, f)

    print(f"Images Converted : {images_converted}")
    if failed_conversions > 0:
        print(f"Failed conversions : {failed_conversions}")

if __name__ == "__main__":

    with open(BACKGROUND_ANN, "r") as f:
        background_data = json.load(f)

    test_image_dir = os.path.join(DIR, "data/segmentation_images/images/")
    test_annotation_path = os.path.join(DIR, "data/segmentation_images/annotations.json")

    start_time = time.time()

    build_dataset(test_image_dir, test_annotation_path, background_data, os.path.join(DIR, "data/segmentation_images/results/"))

    end_time = time.time()

    print(f"Total time taken : {(end_time - start_time):.2f} secs")

    with open(os.path.join(DIR, "data/segmentation_images/results/annotations.json"), "r") as f:
        annotations = json.load(f)

    for i in range(len(annotations["images"])):
        img_path = os.path.join(os.path.join(DIR, "data/segmentation_images/results/images/"), annotations["images"][i]["file_name"])
        kps = np.array(annotations["annotations"][i]["keypoints"]).reshape(17, 3)[:,:2]

        plot_skeleton(img_path, [kps])