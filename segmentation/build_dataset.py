# This is the code to generate a synthetic dataset
# Chooses gymnastics backgrounds, and a COCO dataset pose
# Transforms (rotation, perspective warp, move) person before placing them within the scene
# Saves each image and updated annotation in COCO style dataset

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

# total number of synthetic images
CONVERSION_NUM = 20000

def build_dataset(images_dir, annotations_path, background_data, results_dir):
    # load segmentation model to extract person from dataset
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(DIR, "segmentation/vit-b.pth"))
    predictor = SamPredictor(sam)

    images_converted = 0
    failed_conversions = 0

    # synthetic image dirs
    result_images = os.path.join(results_dir, "images/")
    result_annotations = os.path.join(results_dir, "annotations.json")

    os.makedirs(result_images, exist_ok=True)

    body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # load annotations if they already exist (to append to)
    if os.path.exists(result_annotations):
        with open(result_annotations, "r") as f:
            dataset = json.load(f)

        conversion_count = len(dataset["images"])
    else:
        dataset = {
            "images": [],
            "annotations": [],
            "categories": annotations["categories"]
        }
        conversion_count = 0

    i = 0
    while conversion_count < CONVERSION_NUM:
        # go through images again if we havent created all images yet
        if i >= len(annotations["images"]):
            i = 0 
        try:
            filename = annotations["images"][i]["file_name"]
            path = os.path.join(images_dir, filename)
            person_image = cv2.imread(path)
            if person_image is None:
                raise ValueError(f"Failed to load image: {path}")
            image_id = annotations["images"][i]["id"]

            # find all annotations for image
            image_annotations = [
                ann for ann in annotations["annotations"]
                if ann["image_id"] == image_id
            ]

            if len(image_annotations) == 0:
                i += 1
                continue


            # find all annotations for the image that can be segmented
            valid_candidates = []

            for ann in image_annotations:
                keypoints = ann["keypoints"]
                bbox = ann["bbox"]

                kps = np.array(keypoints).reshape(17, 3)

                # require all keypoints visible
                if not np.all(kps[:, 2][body_idx] == 2):
                    continue

                # ensure size is at least ~100x100
                area = bbox[2] * bbox[3]
                if area < 10000:
                    continue

                valid_candidates.append((area, ann))

            # nothing valid in this image
            if len(valid_candidates) == 0:
                i += 1
                continue

            print(i, f"Converting {path}")

            # randomly choose person from the best 3 random candidates list
            top_k = valid_candidates[:min(3, len(valid_candidates))]
            _, best_ann = random.choice(top_k)

            keypoints = best_ann["keypoints"]
            bbox = best_ann["bbox"]

            # randomly choose background
            bg = background_data[random.randint(0, len(background_data)-1)]
            bg_image = cv2.imread(bg["image"])

            # load mask for foreground (what to place in front of person)
            bg_foreground_mask = cv2.imread(os.path.join(DIR, bg["foreground_mask"]), cv2.IMREAD_GRAYSCALE)
            # load mask for placement of person on background
            bg_placement_mask = cv2.imread(os.path.join(DIR, bg["placement_mask"]), cv2.IMREAD_GRAYSCALE)
            # load size limits person can be
            bg_min_height = bg["min_height"]
            bg_max_height = bg["max_height"]

            start_time = time.time()
            # create synthetic image, convert keypoints and bbox
            new_image, new_bbox, new_kps = composite_background(person_image, bbox, keypoints, bg_image, bg_placement_mask, bg_foreground_mask, predictor, [bg_min_height, bg_max_height], [-180, 180], 0.1, 0.01)

            # create annotation entry for new dataset
            fname = f"{conversion_count:06d}.jpg"
            new_image_entry = {
                "id": conversion_count,
                "file_name": fname,
                "width": new_image.shape[1],
                "height": new_image.shape[0]
            }
            dataset["images"].append(new_image_entry)

            new_annotation_entry = {
                "id": conversion_count,
                "image_id": conversion_count,
                "category_id": 1,
                "keypoints": list(map(float, new_kps)),
                "num_keypoints": int(np.sum(np.array(new_kps)[2::3] > 0)),
                "bbox": list(map(float, new_bbox)),
                "area": float(new_bbox[2] * new_bbox[3]),
                "iscrowd": 0
            }
            dataset["annotations"].append(new_annotation_entry)

            cv2.imwrite(os.path.join(result_images, fname), new_image)
            end_time = time.time()

            print(f"Converted {path}")
            print(f"Time taken : {(end_time - start_time):.2f} secs\n")
            conversion_count += 1
            images_converted += 1
        except Exception as e:
            print(f"ERROR : {e}")
            print(f"Failed to convert Image\n")
            failed_conversions += 1
        
        i += 1

    with open(result_annotations, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Images Converted : {images_converted}")
    if failed_conversions > 0:
        print(f"Failed conversions : {failed_conversions}")

if __name__ == "__main__":

    with open(BACKGROUND_ANN, "r") as f:
        background_data = json.load(f)

    test_image_dir = os.path.join(DIR, "data/coco/train2017/")
    test_annotation_path = os.path.join(DIR, "data/coco/annotations/person_keypoints_train2017.json")

    start_time = time.time()

    build_dataset(test_image_dir, test_annotation_path, background_data, os.path.join(DIR, "data/segmentation_images/"))

    end_time = time.time()

    print(f"Total time taken : {(end_time - start_time):.2f} secs")