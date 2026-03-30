from plot_skeleton import plot_skeleton
import json
import numpy as np
import os
import glob
from PIL import Image

folder = "outputs/"
images_out = "images/"
result_annotations = "3d_person_annotations.json"

os.makedirs(images_out, exist_ok=True)

COCO_ORDER = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle"
]

CATEGORIES = [
    {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ],
        "skeleton": [
            [16,14], [14,12], [17,15], [15,13],
            [12,13],
            [6,12], [7,13],
            [6,7],
            [6,8], [8,10],
            [7,9], [9,11],
            [2,3],
            [1,2], [1,3],
            [2,4], [3,5],
            [4,6], [5,7]
        ]
    }
]

def get_bbox_from_image(img_path):
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    non_zero = np.argwhere(alpha > 0)
    if non_zero.shape[0] == 0:
        return [0,0,img.width,img.height]
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [int(x_min), int(y_min), int(width), int(height)]

def convert_to_coco(input_json, bbox_offset, image_width, image_height, image_id=0, ann_id=0):
    keypoints = []
    num_visible = 0
    x_offset, y_offset = bbox_offset
    for name in COCO_ORDER:
        if name in input_json:
            kp = input_json[name]
            x = kp["x"] * image_width - x_offset
            y = kp["y"] * image_height - y_offset
            v = 2 if kp["visible"] == 1 else 0
            if v > 0:
                num_visible += 1
            keypoints.extend([x, y, v])
        else:
            keypoints.extend([0, 0, 0])
    coco_annotation = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": keypoints,
        "num_keypoints": num_visible,
        "bbox": [0, 0, image_width, image_height],
        "area": image_width * image_height,
        "iscrowd": 0
    }
    return coco_annotation

if __name__ == "__main__":
    png_files = sorted(glob.glob(os.path.join(folder, "img_*.png")))
    ann_files = sorted(glob.glob(os.path.join(folder, "img_*.json")))
    pairs = [(p, a) for p, a in zip(png_files, ann_files)]

    if os.path.exists(result_annotations):
        with open(result_annotations, "r") as f:
            dataset = json.load(f)
        ann_id = max([ann["id"] for ann in dataset["annotations"]], default=0) + 1
        img_id = max([img["id"] for img in dataset["images"]], default=0) + 1
    else:
        dataset = {"images": [], "annotations": [], "categories": CATEGORIES}
        ann_id = 1
        img_id = 1

    for img_path, ann_path in pairs:
        bbox = get_bbox_from_image(img_path)
        x, y, w, h = bbox

        with open(ann_path, "r") as f:
            data = json.load(f)

        img = Image.open(img_path).convert("RGBA")
        cropped_img = img.crop((x, y, x + w, y + h))
        new_file_name = os.path.basename(img_path)
        new_img_path = os.path.join(images_out, new_file_name)
        cropped_img.save(new_img_path)

        dataset["images"].append({
            "id": img_id,
            "file_name": new_file_name,
            "width": w,
            "height": h
        })

        coco_ann = convert_to_coco(data, bbox_offset=(x, y), image_width=512, image_height=512, image_id=img_id, ann_id=ann_id)
        # update bbox to 0,0,width,height
        coco_ann["bbox"] = [0, 0, w, h]
        coco_ann["area"] = w * h
        dataset["annotations"].append(coco_ann)

        img_id += 1
        ann_id += 1

    with open(result_annotations, "w") as f:
        json.dump(dataset, f, indent=2)
        