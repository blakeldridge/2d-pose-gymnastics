import os
import json
import cv2

DIR = os.path.abspath(os.path.dirname(__file__))

IMAGE_DIR = os.path.join(DIR, "images")
ANN_PATH = os.path.join(DIR, "annotations.json")
OUT_PATH = os.path.join(DIR, "annotations_rescaled.json")

TARGET_HEIGHT = 800


def rescale_keypoints(kps, inv_scale):

    new_kps = kps.copy()

    for i in range(0, len(new_kps), 3):

        x = new_kps[i]
        y = new_kps[i+1]
        v = new_kps[i+2]

        if v > 0:
            new_kps[i]   = x * inv_scale
            new_kps[i+1] = y * inv_scale

    return new_kps


def rescale_bbox(bbox, inv_scale):

    return [
        bbox[0] * inv_scale,
        bbox[1] * inv_scale,
        bbox[2] * inv_scale,
        bbox[3] * inv_scale,
    ]


with open(ANN_PATH) as f:
    coco = json.load(f)


# build image lookup
images_by_id = {img["id"]: img for img in coco["images"]}


for ann in coco["annotations"]:

    img = images_by_id[ann["image_id"]]

    img_path = os.path.join(IMAGE_DIR, img["file_name"])

    image = cv2.imread(img_path)

    if image is None:
        print("Missing image:", img_path)
        continue

    original_h = image.shape[0]

    scale = TARGET_HEIGHT / original_h
    inv_scale = 1.0 / scale

    ann["keypoints"] = rescale_keypoints(ann["keypoints"], inv_scale)

    ann["bbox"] = rescale_bbox(ann["bbox"], inv_scale)

    ann["area"] = ann["bbox"][2] * ann["bbox"][3]


with open(OUT_PATH, "w") as f:
    json.dump(coco, f, indent=2)

print("Saved corrected annotations to:", OUT_PATH)