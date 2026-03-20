import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from utils.visualisation import plot_skeleton
from mmpose.datasets.transforms.gymnastics_transforms import BlurLimbs, OccludeLimbs
from mmpose.datasets.transforms import LoadImage, GetBBoxCenterScale

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def scale_background(bg_image, min_h, min_w):
    bh, bw = bg_image.shape[:2]

    scale = max((min_w+1) / bw, (min_h+1) / bh)

    if scale < 1:
        scale = 1

    new_w = int(bw * scale)
    new_h = int(bh * scale)

    bg_image = cv2.resize(bg_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return bg_image

def keypoints_to_bbox_relative(keypoints, bbox):
    x, y, w, h = bbox
    rel = []

    for i in range(0, len(keypoints), 3):
        kx, ky, v = keypoints[i:i+3]

        if v == 0:
            rel.extend([0,0,0])
            continue

        rel_x = (kx - x) / w
        rel_y = (ky - y) / h

        rel.extend([rel_x, rel_y, v])

    return rel

def keypoints_from_bbox_relative(rel_keypoints, new_bbox):
    x, y, w, h = new_bbox
    out = []

    for i in range(0, len(rel_keypoints), 3):
        rx, ry, v = rel_keypoints[i:i+3]

        if v == 0:
            out.extend([0,0,0])
            continue

        kx = x + rx * w
        ky = y + ry * h

        out.extend([kx, ky, v])

    return out

def composite_backgrounds(image_path, keypoints, bbox, background_path, predictor):
    image = cv2.imread(image_path)

    x,y,w,h = bbox
    sam_box = np.array([x, y, x+w, y+h])

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=sam_box, multimask_output=True)
    mask = masks[np.argmax(scores)]

    masked_image = image.copy()
    masked_image[~mask] = 0

    person = image * mask[..., None]
    ys, xs = np.where(mask)
    my1, my2 = ys.min(), ys.max()
    mx1, mx2 = xs.min(), xs.max()

    # Crop the foreground
    foreground_crop = person[my1:my2, mx1:mx2]
    mask_crop = mask[my1:my2, mx1:mx2]

    # Convert keypoints to crop coordinates (absolute within crop)
    cropped_kps = []
    for i in range(0, len(keypoints), 3):
        kx, ky, v = keypoints[i:i+3]
        if v == 0:
            cropped_kps.extend([0,0,0])
            continue
        cropped_kps.extend([kx - mx1, ky - my1, v])

    # Rotate foreground + mask + keypoints
    angle = np.random.uniform(-35, 35)
    foreground_crop, mask_crop, rotated_kps = rotate_foreground(
        foreground_crop,
        mask_crop,
        cropped_kps,
        angle
    )

    # Recompute bbox from rotated mask
    mask_bbox = mask_to_bbox(mask_crop)
    h, w = foreground_crop.shape[:2]

    # Paste onto background
    bg_image = cv2.imread(background_path)
    bg_image = scale_background(bg_image, h, w)
    bh, bw = bg_image.shape[:2]

    y = np.random.randint(0, bh - h)
    x = np.random.randint(0, bw - w)
    new_bbox = [x, y, mask_bbox[2], mask_bbox[3]]

    # Offset rotated keypoints to new background location
    new_keypoints = []
    for i in range(0, len(rotated_kps), 3):
        kx, ky, v = rotated_kps[i:i+3]
        if v == 0:
            new_keypoints.extend([0,0,0])
            continue
        new_keypoints.extend([kx + x, ky + y, v])

    # Composite
    roi = bg_image[y:y+h, x:x+w]
    roi[mask_crop.astype(bool)] = foreground_crop[mask_crop.astype(bool)]
    bg_image[y:y+h, x:x+w] = roi

    return bg_image, new_keypoints, new_bbox

def blur_and_occlude_limbs(image_path, keypoints, bbox):
    results = dict(
        img_path=[image_path],
        keypoints=[keypoints],
        bbox=bbox
    )

    transform = LoadImage()
    results = transform(results)

    transform = GetBBoxCenterScale()
    results = transform(results)

    transform = BlurLimbs(blur_prob=0.3)
    results = transform(results)

    transform = OccludeLimbs(occlusion_prob=0.4, max_size_ratio=0.1)
    results = transform(results)

    return results["img"]

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [x1, y1, x2-x1, y2-y1]

def rotate_foreground(image, mask, keypoints, angle):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0,0])
    sin = abs(M[0,1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0,2] += (new_w / 2) - cx
    M[1,2] += (new_h / 2) - cy

    rotated_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (new_w, new_h), flags=cv2.INTER_NEAREST)

    rotated_kps = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]

        if v == 0:
            rotated_kps.extend([0,0,0])
            continue

        px = M[0,0]*x + M[0,1]*y + M[0,2]
        py = M[1,0]*x + M[1,1]*y + M[1,2]

        rotated_kps.extend([px, py, v])

    return rotated_img, rotated_mask.astype(bool), rotated_kps

# def rotate_foreground(image, mask, keypoints, angle):
#     h, w = image.shape[:2]
#     cx, cy = w / 2, h / 2

#     M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

#     cos = abs(M[0,0])
#     sin = abs(M[0,1])

#     new_w = int((h * sin) + (w * cos))
#     new_h = int((h * cos) + (w * sin))

#     M[0,2] += (new_w / 2) - cx
#     M[1,2] += (new_h / 2) - cy

#     rotated_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
#     rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (new_w, new_h), flags=cv2.INTER_NEAREST)

#     rotated_kps = []
#     for i in range(0, len(keypoints), 3):
#         x, y, v = keypoints[i:i+3]

#         if v == 0:
#             rotated_kps.extend([0,0,0])
#             continue

#         pt = np.array([x, y, 1.0])
#         px, py = M @ pt

#         rotated_kps.extend([px, py, v])

#     return rotated_img, rotated_mask.astype(bool), rotated_kps

if __name__ == "__main__":
    #keypoints_list = [1137.6, 1708.8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1224.0, 1545.6, 2, 1070.3999999999999, 1545.6, 2, 1291.2, 1756.8, 2, 1099.2, 1756.8, 2, 1315.2, 1963.1999999999998, 2, 1051.2, 1915.1999999999998, 2, 1296.0, 1152.0, 2, 1142.3999999999999, 1142.3999999999999, 2, 1329.6, 787.1999999999999, 2, 1243.2, 801.6, 2, 1401.6, 465.59999999999997, 2, 1344.0, 470.4, 2]
    #bbox = np.array([974.4, 321.6, 504.0, 1732.8])
    keypoints_list = [1576.8, 602.1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1660.5, 464.4, 2, 1809.0, 577.8, 2, 1765.8, 621.0, 2, 1968.3, 772.2, 2, 1695.6, 850.5, 2, 2111.4, 982.8, 2, 2154.6, 656.1, 2, 2130.3, 715.5, 2, 2543.4, 904.5, 2, 2521.8, 912.6, 2, 2940.3, 1166.4, 2, 2916.0, 1136.7, 2]
    bbox = np.array([1452.6000000000001, 332.1, 1755.0000000000002, 1093.5])
    image_path = os.path.join(DIR, "data/test/images/00010.jpg")
    bg_path = os.path.join(DIR, "data/test/images/00004.jpg")

    body_idx = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    body_connections = [
        [1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12]
    ]

    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(DIR, "segmentation/vit-b.pth"))
    predictor = SamPredictor(sam)

    image, kps, bbox = composite_backgrounds(image_path, keypoints_list, bbox, bg_path, predictor)

    result_path = os.path.join(DIR, "segmentation/result.jpg")
    cv2.imwrite(result_path, image)

    #image = blur_and_occlude_limbs(result_path, np.array(kps).reshape(17, 3)[:, :2], np.array(bbox))
    #cv2.imwrite(result_path, image)

    plot_skeleton(result_path, [np.array(kps).reshape(17, 3)[body_idx][:,:2]], body_connections, [1,2])