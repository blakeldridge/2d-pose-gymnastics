from segmentation.segmentation import composite_backgrounds
from utils.visualisation import plot_skeleton
from segment_anything import SamPredictor, sam_model_registry
import os
import json
import cv2
import random
import time

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BACKGROUNDS = os.path.join(DIR, "data/backgrounds/")

def pick_background():
    background_paths = [os.path.join(BACKGROUNDS, f) for f in os.listdir(BACKGROUNDS)]

    return random.choice(background_paths)

def build_dataset(images_dir, annotations_path, results_dir):
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(DIR, "segmentation/vit-b.pth"))
    predictor = SamPredictor(sam)

    images_converted = 0
    failed_conversions = 0

    result_images = os.path.join(results_dir, "images/")
    result_annotations = os.path.join(results_dir, "annotations.json")

    os.makedirs(result_images, exist_ok=True)

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    for i in range(len(annotations["images"])):
        try:
            filename = annotations["images"][i]["file_name"]
            path = os.path.join(images_dir, filename)
            image_id = annotations["images"][i]["id"]
            print(f"Converting {path}")

            for j in range(len(annotations["annotations"])):
                if annotations["annotations"][j]["id"] == image_id:
                    keypoints = annotations["annotations"][j]["keypoints"]
                    bbox = annotations["annotations"][j]["bbox"]

            bg_path = pick_background()

            start_time = time.time()
            new_image, new_kps, new_bbox = composite_backgrounds(path, keypoints, bbox, bg_path, predictor)

            annotations["annotations"][i]["keypoints"] = new_kps
            annotations["annotations"][i]["bbox"] = new_bbox

            cv2.imwrite(os.path.join(result_images, filename), new_image)
            end_time = time.time()

            print(f"Converted {path}")
            print(f"Time taken : {(end_time - start_time):.2f} secs\n")
            images_converted += 1
        except:
            print(f"Failed to convert Image\n")
            failed_conversions += 1

    with open(result_annotations, "w") as f:
        json.dump(annotations, f)

    print(f"Images Converted : {images_converted}")
    if failed_conversions > 0:
        print(f"Failed conversions : {failed_conversions}")

if __name__ == "__main__":
    test_image_dir = os.path.join(DIR, "data/segmentation_images/images/")
    test_annotation_path = os.path.join(DIR, "data/segmentation_images/annotations.json")

    start_time = time.time()

    build_dataset(test_image_dir, test_annotation_path, os.path.join(DIR, "data/segmentation_images/results/"))

    end_time = time.time()

    print(f"Total time taken : {(end_time - start_time):.2f} secs")

    with open(test_annotation_path, "r") as f:
        annotations = json.load(f)

    for i in range(len(annotations["images"])):
        img_path = annotations["images"][i]["file_name"]
        kps = annotations["annotations"][i]["keypoints"]

        plot_skeleton(img_path, [kps])