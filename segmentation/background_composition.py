import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from utils.visualisation import plot_skeleton

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [x1, y1, x2-x1, y2-y1]

def transform_foreground(image, mask, keypoints, angle, scale=1.0, max_shear=0.05, max_persp=0.0005):
    h, w = image.shape[:2]

    if scale != 1:
        target_h, target_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        scaled_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i:i+3]

            if v == 0:
                scaled_keypoints.extend([0,0,0])
                continue

            scaled_keypoints.extend([x * scale, y * scale, v])

        keypoints = scaled_keypoints

    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M_rot = np.vstack([M_rot, [0, 0, 1]])

    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    def jitter(pt):
        x, y = pt
        x += np.random.uniform(-max_shear, max_shear) * w
        y += np.random.uniform(-max_shear, max_shear) * h
        return [x, y]

    dst_corners = np.array([jitter(pt) for pt in corners], dtype=np.float32)

    H_persp = cv2.getPerspectiveTransform(corners, dst_corners)
    H = H_persp @ M_rot

    corners_h = np.hstack([corners, np.ones((4,1))])
    warped_corners = (H @ corners_h.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

    x_min, y_min = warped_corners.min(axis=0)
    x_max, y_max = warped_corners.max(axis=0)

    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    H = T @ H

    warped_img = cv2.warpPerspective(image, H, (new_w, new_h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(mask.astype(np.uint8), H, (new_w, new_h), flags=cv2.INTER_NEAREST)

    warped_kps = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]

        if v == 0:
            warped_kps.extend([0,0,0])
            continue

        pt = np.array([x, y, 1.0])
        px, py, pz = H @ pt
        px /= pz
        py /= pz

        warped_kps.extend([px, py, v])

    return warped_img, warped_mask.astype(bool), warped_kps

def composite_background(image, person_bbox, keypoints, background_image, placement_mask, foreground_mask, predictor, size_limits, angle_limits=[-15, 15], max_shear=0.05, max_persp=0.0005):
    predictor.set_image(image)
    x, y, w, h = person_bbox
    sam_box = np.array([x, y, x+w, y+h])
    masks, scores, _ = predictor.predict(box=sam_box, multimask_output=True)
    mask = masks[np.argmax(scores)]
    masked_image = image.copy()
    masked_image[~mask] = 0

    person = image * mask[..., None]
    ys, xs = np.where(mask)
    my1, my2 = ys.min(), ys.max()
    mx1, mx2 = xs.min(), xs.max()

    foreground_crop = person[my1:my2, mx1:mx2]
    mask_crop = mask[my1:my2, mx1:mx2]
    
    cropped_kps = []
    for i in range(0, len(keypoints), 3):
        kx, ky, v = keypoints[i:i+3]
        if v == 0:
            cropped_kps.extend([0,0,0])
            continue
        cropped_kps.extend([kx - mx1, ky - my1, v])
        
    angle = np.random.uniform(angle_limits[0], angle_limits[1])
    scale = np.random.uniform(size_limits[0], size_limits[1]) / foreground_crop.shape[0]
    foreground_crop, mask_crop, rotated_kps = transform_foreground(
        foreground_crop,
        mask_crop,
        cropped_kps,
        angle,
        scale,
        max_shear,
        max_persp
    )

    mask_bbox = mask_to_bbox(mask_crop)
    h, w = foreground_crop.shape[:2]

    # PLACE IN NEW BACKGROUND
    placement_indices = np.argwhere(placement_mask > 0)
    y, x = placement_indices[np.random.choice(len(placement_indices))][:2]
    y = max(0, y - int(h/2))
    x = max(0, x - int(x/2))

    bbox = [x, y, mask_bbox[2], mask_bbox[3]]

    keypoints = []
    for i in range(0, len(rotated_kps), 3):
        kx, ky, v = rotated_kps[i:i+3]
        if v == 0:
            keypoints.extend([0,0,0])
            continue

        keypoints.extend([kx + x, ky + y, v])
    # Place foreground on background
    foreground = np.where(foreground_mask > 0, background_image, 0)

    H_bg, W_bg = background_image.shape[:2]
    h, w = foreground_crop.shape[:2]

    # Ensure placement stays inside bounds
    y_end = min(y + h, H_bg)
    x_end = min(x + w, W_bg)

    h_valid = y_end - y
    w_valid = x_end - x

    # Crop everything to the valid region
    roi = background_image[y:y_end, x:x_end]
    fg_crop = foreground_crop[:h_valid, :w_valid]
    mask_valid = mask_crop[:h_valid, :w_valid]

    # Apply mask
    roi[mask_valid.astype(bool)] = fg_crop[mask_valid.astype(bool)]

    # Write back
    background_image[y:y_end, x:x_end] = roi

    result = np.where(foreground != 0, foreground, background_image)

    v_keypoints = []
    for i in range(0, len(keypoints), 3):
        kx, ky, v = keypoints[i:i+3]
        if v == 0:
            v_keypoints.extend([0,0,0])
            continue

        ix = int(round(kx))
        iy = int(round(ky))

        if (foreground[iy, ix] > 0).any():
            v_keypoints.extend([kx, ky, 1])
        else:
            v_keypoints.extend([kx, ky, v])

    return result, bbox, v_keypoints

if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(DIR, "segmentation/vit-b.pth"))
    predictor = SamPredictor(sam)

    test_image_path = os.path.join(DIR, "data/test/images/00003.jpg")

    annotation_path = os.path.join(DIR, "segmentation/annotations/annotations.json")

    # LOAD BACKGROUND DETAILS
    with open(annotation_path, "r") as f:
        annotations = json.load(f)[0]

    background = cv2.imread(annotations["image"])
    foreground_mask = cv2.imread(annotations["foreground_mask"])
    placement_mask = cv2.imread(annotations["placement_mask"])
    min_height = annotations["min_height"]
    max_height = annotations["max_height"]

    # GET PERSON MASK FROM IMAGE
    test_image = cv2.imread(test_image_path)
    test_image_kps = [1137.6, 1708.8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1224.0, 1545.6, 2, 1070.3999999999999, 1545.6, 2, 1291.2, 1756.8, 2, 1099.2, 1756.8, 2, 1315.2, 1963.1999999999998, 2, 1051.2, 1915.1999999999998, 2, 1296.0, 1152.0, 2, 1142.3999999999999, 1142.3999999999999, 2, 1329.6, 787.1999999999999, 2, 1243.2, 801.6, 2, 1401.6, 465.59999999999997, 2, 1344.0, 470.4, 2]
    test_image_bbox = np.array([974.4, 321.6, 504.0, 1732.8])

    result, bbox, keypoints = composite_background(test_image, test_image_bbox, test_image_kps, background, placement_mask, foreground_mask, predictor, [min_height, max_height], [-180, 180], 0.1, 0.01)
    result_path = os.path.join(DIR, "segmentation/result.jpg")
    cv2.imwrite(result_path, result)

    body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    body_connections = [
        [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
    ]

    kps = np.array(keypoints).reshape(17, 3)[:,:2][body_idx]

    # Show Images
    plot_skeleton(result_path, [kps], body_connections, [1, 2])