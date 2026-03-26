# Evaluates all different experiments for HRNet and ViTPose 
# gets metrics for all joints, and types of poses (static vs dynamic)
# compares motion-blurred, occluded to non motion blur and occluded poses

import pickle
import os
import json
import numpy as np
from utils.metrics import evaluate_pose
from utils.visualisation import plot_skeleton
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def convert_to_json(raw_file):
    with open(raw_file, "rb") as f:
        results = pickle.load(f)

    output = []

    for r in results:
        kps = r['pred_instances']['keypoints'][0]
        scores = r['pred_instances']['keypoint_scores'][0]

        keypoints = []
        for (x, y), s in zip(kps, scores):
            keypoints.extend([float(x), float(y), float(s)])

        output.append({
            "image_id": r['img_id'],
            "category_id": 1,
            "keypoints": keypoints,
            "score": float(scores.mean())
        })

    return output

def coco_evaluation(results_path, ann_file):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results_path)

    img_ids = sorted(coco_gt.getImgIds())

    cocoEval = COCOeval(coco_gt, coco_dt, "keypoints")
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def filter_annotations(pred_data, ann_dict, labels={}):
    """Filter predictions and GT based on labels (pose_type, motion_blur, self_occlusion, apparatus_occlusion)"""
    filtered_preds = []
    filtered_gts = []
    filtered_scales = []
    filtered_ids = []

    for pred, p in zip(pred_data, pred_data):
        image_id = p["image_id"]
        ann = ann_dict[image_id][0]

        keep = True
        for key, val in labels.items():
            if ann.get(key) != val:
                keep = False
                break

        if keep:
            pred_kp = np.array(p["keypoints"]).reshape(17, 3)[:, :2]
            gt_kp = np.array(ann["keypoints"]).reshape(17, 3)[:, :2]

            bbox = ann["bbox"]
            scale_val = np.sqrt(bbox[2]**2 + bbox[3]**2)

            filtered_preds.append(pred_kp)
            filtered_gts.append(gt_kp)
            filtered_scales.append(scale_val)
            filtered_ids.append(image_id)

    return np.array(filtered_preds), np.array(filtered_gts), np.array(filtered_scales), filtered_ids

if __name__ == "__main__":

    experiments = [
        "hrnet",
        "hrnet_coco_freeze",
        "hrnet_athlete_freeze",
        "hrnet_custom",
        "vitpose",
        "vitpose_coco"
    ]

    image_dir = os.path.join(DIR, "data/test/images/")
    annotation_path = os.path.join(DIR, "data/test/annotations.json")

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    image_dict = {img["id"]: img for img in annotations["images"]}
    ann_dict = {}
    for ann in annotations["annotations"]:
        ann_dict.setdefault(ann["image_id"], []).append(ann)

    # WITH NOSE POINTS

    body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    body_connections = [
        [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
    ]
    joint_names = [
        "nose", "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    # WITHOUT NOSE POINTS

    # body_idx = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # body_connections = [
    #     [0,1],[0,2],[2,4],[1,3],[3,5],
    #     [0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]
    # ]
    # joint_names = [
    #     "left_shoulder","right_shoulder","left_elbow","right_elbow",
    #     "left_wrist","right_wrist","left_hip","right_hip",
    #     "left_knee","right_knee","left_ankle","right_ankle"
    # ]

    sigmas = np.array([
        0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,
        0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089
    ])[body_idx]

    mask = np.ones((1, len(body_idx)))

    USE_COCO_EVAL = True

    for experiment in experiments:

        raw_output_file = f"results/{experiment}/outputs.pkl"
        raw_output_path = os.path.join(DIR, raw_output_file)

        pred_data = convert_to_json(raw_output_path)

        # If athlete experiment remove nose
        if "athlete" in experiment.lower():
            for p in pred_data:
                kp = p["keypoints"]
                kp[0:3] = [0,0,0]
                p["keypoints"] = kp

        print("\n" + "="*50)
        print(f"Experiment: {experiment}")
        print("="*50)
        print("Total images:", len(pred_data))

        temp_pred_path = os.path.join(DIR, f"results/{experiment}/temp_preds.json")
        with open(temp_pred_path, "w") as f:
            json.dump(pred_data, f)

        # Display / Compute COCO evalutation of entire output of model
        # - if athlete, compute without nose 
        if "athlete" in experiment.lower():
            temp_ann_path = os.path.join(DIR, f"results/{experiment}/temp_ann.json")
            temp_annotations = json.loads(json.dumps(annotations))
            for ann in temp_annotations["annotations"]:
                ann["keypoints"][0:3] = [0,0,0]
            with open(temp_ann_path, "w") as f:
                json.dump(temp_annotations, f)
            print("\n[COCO EVAL - NO NOSE]")
            coco_evaluation(temp_pred_path, temp_ann_path)
        else:
            print("\n[COCO EVAL]")
            coco_evaluation(temp_pred_path, annotation_path)

        # for each label compute all the metrics
        subset_labels = [
            {"pose_type":"dynamic"},
            {"pose_type":"static"},
            {"motion_blur":True},
            {"motion_blur":False},
            {"self-occlusion":True},
            {"self-occlusion":False},
            {"apparatus-occlusion":True},
            {"apparatus-occlusion":False},
        ]

        for label in subset_labels:
            label_name = ", ".join([f"{k}={v}" for k,v in label.items()])
            print("\n" + "-"*50)
            print(f"Subset evaluation: {label_name}")
            print("-"*50)

            preds_sub, gts_sub, scales_sub, _ = filter_annotations(pred_data, ann_dict, label)
            if len(preds_sub) == 0:
                print("No images for this subset!")
                continue

            evaluate_pose(preds_sub[:,body_idx,:], gts_sub[:,body_idx,:], scales_sub, mask, sigmas, joint_names)