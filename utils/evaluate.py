import pickle
import os
import json
import numpy as np
from utils.metrics import evaluate_pose
from utils.visualisation import plot_skeleton

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

if __name__ == "__main__":
    experiments = ["hrnet", "hrnet_coco", "hrnet_athlete", "vitpose"]
    # experiments = ["hrnet_athlete"]
    for experiment in experiments:
        raw_output_file = f"results/{experiment}/outputs.pkl"
        raw_output_path = os.path.join(DIR, raw_output_file)

        image_dir = os.path.join(DIR, "data/test/images/")
        annotation_dir = os.path.join(DIR, "data/test/annotations.json")
        with open(annotation_dir, "r") as f:
            annotations = json.load(f)
        
    
        body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        body_connections = [
            [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
        ]

        pred_data = convert_to_json(raw_output_path)

        scales = []
        preds = []
        gts = []

        for p in pred_data:
            image_id = p["image_id"]
            pred = p["keypoints"]

            for idx in range(len(annotations["images"])):
                if annotations["images"][idx]["id"] == image_id:
                    image_file = annotations["images"][idx]["file_name"]
                    gt = annotations["annotations"][idx]["keypoints"]
                    bbox = annotations["annotations"][idx]["bbox"]
                    break

            pred = np.array(pred).reshape(17, 3)[:, :2][body_idx]
            gt = np.array(gt).reshape(17, 3)[:, :2][body_idx]

            scale_val = np.sqrt(bbox[2]**2 + bbox[3]**2)
            scale = np.array([scale_val])

            preds.append(pred)
            gts.append(gt)
            scales.append(scale)

        mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        mask = np.array([mask])

        sigmas = np.array([0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089])
        sigmas = sigmas[body_idx]


        joint_names = ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]


        print("----------------------------------------------")
        print(f"{experiment}")
        print("----------------------------------------------")
        evaluate_pose(np.array(preds), np.array(gts), scale, mask, sigmas, joint_names)