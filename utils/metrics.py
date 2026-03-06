import numpy as np

def per_joint_error(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1)

def mpjpe(pred, gt):
    return per_joint_error(pred, gt).mean()

def pck(pred, gt, scale, threshold=0.2):
    errors = per_joint_error(pred, gt)
    correct = errors < (threshold * scale[:, None])
    return correct.mean()

def pck_per_joint(pred, gt, scale, threshold=0.2):
    errors = np.linalg.norm(pred - gt, axis=-1)
    correct = errors < (threshold * scale[:, None])
    return correct.mean(axis=0)

def auc_pck(pred, gt, scale, max_threshold=0.5, steps=100):
    thresholds = np.linspace(0, max_threshold, steps)
    scores = []

    for t in thresholds:
        score = pck(pred, gt, scale, threshold=t)
        scores.append(score)

    return np.trapezoid(scores, thresholds) / max_threshold

def oks(pred, gt, scale, sigmas, vis):
    d = np.linalg.norm(pred - gt, axis=-1)

    vars = (sigmas * 2)**2
    e = np.exp(-(d**2) / (2 * (scale[:,None]**2) * vars))

    oks = (e * vis).sum(axis=1) / vis.sum(axis=1)
    return oks.mean()

import numpy as np


def compute_oks_per_sample(pred, gt, scale, sigmas, vis):
    d = np.linalg.norm(pred - gt, axis=-1)                  # (N,J)
    vars = (sigmas * 2)**2                                  # (J,)
    e = np.exp(-(d**2) / (2 * (scale[:, None]**2) * vars)) # (N,J)
    oks = (e * vis).sum(axis=1) / vis.sum()                # (N,)
    return oks

def compute_ap_from_oks(oks_scores, thresholds):
    ap = []

    for t in thresholds:
        correct = oks_scores >= t
        ap.append(correct.mean())

    return np.array(ap)


def evaluate_pose(pred, gt, scale, vis, sigmas, joint_names, pck_thresholds=(0.1, 0.2), oks_thresholds=np.arange(0.5, 0.96, 0.05)):

    results = {}

    errors = per_joint_error(pred, gt)

    results["MPJPE"] = mpjpe(pred, gt)

    results["per_joint_mean_error"] = errors.mean(axis=0)
    results["per_joint_median_error"] = np.median(errors, axis=0)

    for t in pck_thresholds:
        results[f"PCK@{t}"] = pck(pred, gt, scale, threshold=t)

    results["AUC_PCK"] = auc_pck(pred, gt, scale)

    oks_scores = compute_oks_per_sample(pred, gt, scale, sigmas, vis)
    results["OKS_mean"] = oks_scores.mean()

    ap_scores = compute_ap_from_oks(oks_scores, oks_thresholds)

    print("\nMetric summary")
    print("----------------")

    print(f"MPJPE: {results['MPJPE']:.3f}")

    for t in pck_thresholds:
        print(f"PCK@{t}: {results[f'PCK@{t}']:.3f}")

    print(f"AUC PCK: {results['AUC_PCK']:.3f}")
    print(f"OKS mean: {results['OKS_mean']:.3f}")

    print("\nPer-joint mean error (pixels)")
    print("-------------------------------")

    for j, name in enumerate(joint_names):
        mean_err = results["per_joint_mean_error"][j]
        med_err = results["per_joint_median_error"][j]

        print(f"{name:<15} mean={mean_err:.3f}  median={med_err:.3f}")

    print("\nAP (OKS thresholds)")
    print("-------------------")

    for t, score in zip(oks_thresholds, ap_scores):
        print(f"AP@{t:.2f}: {score:.3f}")

    print(f"\nCOCO-style AP: {ap_scores.mean():.3f}")

    return results, ap_scores


"""if __name__ == "__main__":
    import os
    import json
    from utils import visualisation
    from models.vitpose import ViTPose

    DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    model = ViTPose()

    image_path = os.path.join(DIR, "data/test/images/image_1.png")
    gt_path = os.path.join(DIR, "data/test/annotations.json")

    with open(gt_path, "r") as f:
        annotations = json.load(f)

    result = model(image_path, "device")

    body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pred = result[0][0]["keypoints"]
    pred = pred[body_idx, :]

    gt = np.array(annotations["annotations"][0]["keypoints"]).reshape(13, 3)[:, :2]

    body_connections = [
    [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
    ]

    visualisation.plot_skeleton(image_path, [pred, gt], body_connections)

    bbox = annotations["annotations"][0]["bbox"]

    scale_val = np.sqrt(bbox[2]**2 + bbox[3]**2)
    scale = np.array([scale_val])

    pred = pred[None, :, :]
    gt   = gt[None, :, :]  

    mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    mask = np.array([mask])

    sigmas = np.array([0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089])
    sigmas = sigmas[body_idx]


    joint_names = ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    evaluate_pose(pred, gt, scale, mask, sigmas, joint_names)"""