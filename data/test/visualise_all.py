import os
import json
import numpy as np
from utils import visualisation

DIR = os.path.abspath(os.path.dirname(__file__))
image_dir = os.path.join(DIR, "deblurred_images")

# POI = [61, 102, 68, 113]

def get_keypoints_for_image(coco, filename):
    # 1. find image id
    image_id = None
    for img in coco["images"]:
        if img["file_name"] == filename:
            image_id = img["id"]
            break

    # if image_id not in POI:
    #     return None
    
    print("id", image_id)

    if image_id is None:
        print("Image not found")
        return None

    # 2. find annotation
    for ann in coco["annotations"]:
        if ann["image_id"] == image_id:
            return ann["keypoints"]

    print("Annotation not found")
    return None

image_paths = [{"name":f, "path":os.path.join(image_dir, f)} for f in os.listdir(image_dir)]

with open(os.path.join(DIR, "annotations.json"), "r") as f:
    annotations = json.load(f)

body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
body_connections = [
    [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
]

for img in image_paths:
    keypoints = get_keypoints_for_image(annotations, img["name"])
    if keypoints:
        keypoints = np.array(keypoints).reshape(17, 3)[:,:2]
        keypoints = keypoints[body_idx]
        visualisation.plot_skeleton(img["path"], [keypoints] ,body_connections, [1,2], True)