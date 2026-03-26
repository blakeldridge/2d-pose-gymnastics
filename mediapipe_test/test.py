# Compute AP for mediapipe inference

from mediapipe_test.model import MediaPipe
import os
import json
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
IMAGES_DIR = os.path.join(DIR, "data/test/images/")

def coco_evaluation(results_path, ann_file):
    coco_gt = COCO(ann_file)

    coco_dt = coco_gt.loadRes(results_path)

    img_ids = sorted(coco_gt.getImgIds())

    cocoEval = COCOeval(coco_gt, coco_dt, "keypoints")
    cocoEval.params.imgIds  = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def run_model(model, results_path, ann_file, batch_size=-1):
    start_time = time.time()

    with open(ann_file, "r") as f:
        data = json.load(f)

    max_images = len(data["images"])
    image_fnames = [os.path.join(IMAGES_DIR, img["file_name"]) for img in data["images"]]
    image_ids = [img["id"] for img in data["images"]]

    results = []

    if batch_size == -1:
        batch_size = max_images

    batch_time = time.time()
    for i in range(int(max_images / batch_size)):
        batch_start = batch_size * i
        batch = image_fnames[batch_start : batch_start + batch_size]

        outputs = model(batch, image_loc="device")

        for index, output in enumerate(outputs):
            if type(output) == list:
                for pose in output:
                    keypoints = pose["keypoints"]
                    score = pose["score"]
                    visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                    keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                    result = {"image_id": image_ids[batch_start + index], "category_id": 1, "keypoints":keypoints.tolist(), "score": float(score)}
                    results.append(result)
            else:
                keypoints = output["keypoints"]
                score = output["score"]
                visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                result = {"image_id": image_ids[batch_start + index], "category_id": 1, "keypoints":keypoints.tolist(), "score":float(score)}
                results.append(result)

        print(f"Batch {i + 1} Completed : {(time.time() - batch_time):.2f} secs")
        batch_time = time.time()

    results_json = json.dumps(results, indent=4)
    with open(results_path, "w") as f:
        f.write(results_json)
    end_time = time.time()

    print(f"Pose Estimation Complete : {(end_time - start_time):.2f} secs")

if __name__ == "__main__":
    import mediapipe as mp
    print(mp.__version__)
    print(mp.__file__)
    ANNOTATIONS_PATH = os.path.join(DIR, "data/test/annotations.json")
    RESULT_PATH = os.path.join(DIR, "mediapipe_test/results.json")
    model = MediaPipe()

    run_model(model, RESULT_PATH, ANNOTATIONS_PATH, 30)

    coco_evaluation(RESULT_PATH, ANNOTATIONS_PATH)