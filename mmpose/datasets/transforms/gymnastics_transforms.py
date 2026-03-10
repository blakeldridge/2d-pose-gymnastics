from typing import Dict, Optional, Tuple, List, Union
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
import numpy as np
import cv2
import albumentations as alb

@TRANSFORMS.register_module()
class BlurLimbs(BaseTransform):
    def __init__(self, blur_prob, max_kernel=5, max_size=20):
        self.blur_prob = blur_prob
        self.max_kernel = max_kernel
        self.max_size = max_size

    def transform(self, results):
        img = results["img"].copy()
        joints = results["keypoints"].copy()  # (17,2)

        limb_pairs = [
            (7, 9), (8, 10),     # elbow–wrist
            (5, 7), (6, 8),      # shoulder–elbow
            (11, 13),(12, 14), # hip-knee
            (13, 15), (14, 16)   # knee–ankle
        ]

        for j1, j2 in limb_pairs:
            if np.random.rand() >= self.blur_prob:
                continue

            x1, y1 = joints[j1].astype(int)
            x2, y2 = joints[j2].astype(int)

            # Compute limb length
            length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            if length < 5:
                continue

            # Kernel size proportional to limb length
            k = max(5, length // 4)
            if k % 2 == 0:
                k += 1  # must be odd

            kernel = np.zeros((k, k))

            direction = np.random.choice(
                ["horizontal", "vertical", "diag1", "diag2"]
            )

            if direction == "horizontal":
                kernel[k//2, :] = 1
            elif direction == "vertical":
                kernel[:, k//2] = 1
            elif direction == "diag1":  # \
                np.fill_diagonal(kernel, 1)
            elif direction == "diag2":  # /
                np.fill_diagonal(np.fliplr(kernel), 1)

            kernel /= kernel.sum()

            # ROI bounds based on limb extent
            padding = int(length * 0.3)

            x_min = max(min(x1, x2) - padding, 0)
            x_max = min(max(x1, x2) + padding, img.shape[1])
            y_min = max(min(y1, y2) - padding, 0)
            y_max = min(max(y1, y2) + padding, img.shape[0])

            roi = img[y_min:y_max, x_min:x_max]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            blurred = cv2.filter2D(roi, -1, kernel)
            img[y_min:y_max, x_min:x_max] = blurred

        results["img"] = img
        return results
    
@TRANSFORMS.register_module()
class OccludeLimbs(BaseTransform):
    def __init__(self, occlusion_prob=0.5, max_occluded_limbs=3, max_size_ratio=0.3):
        """
        Args:
            occlusion_prob (float): probability to occlude a limb
            max_occluded_limbs (int): maximum number of limbs to occlude per image
            max_size_ratio (float): padding around limb as a fraction of limb length
        """
        self.occlusion_prob = occlusion_prob
        self.max_occluded_limbs = max_occluded_limbs
        self.max_size_ratio = max_size_ratio

    def transform(self, results):
        img = results["img"].copy()
        joints = results["keypoints"].copy()  # (17,2) COCO

        # Limb pairs to target (elbow→wrist, shoulder→elbow, knee→ankle)
        limb_pairs = [
            (5, 7), (6, 8),    # shoulder→elbow
            (7, 9), (8, 10),   # elbow→wrist
            (11, 13),(12, 14), # hip-knee
            (11, 12),          # hip-hip    
            (13, 15), (14, 16) # knee→ankle
        ]

        # Randomly shuffle limbs so different limbs occluded each time
        np.random.shuffle(limb_pairs)
        occluded_count = 0

        for j1_idx, j2_idx in limb_pairs:
            if occluded_count >= self.max_occluded_limbs:
                break

            if np.random.rand() > self.occlusion_prob:
                continue

            x1, y1 = joints[j1_idx].astype(int)
            x2, y2 = joints[j2_idx].astype(int)

            # Compute limb length
            limb_length = np.linalg.norm([x2 - x1, y2 - y1])
            if limb_length < 5:
                continue

            # Padding around limb proportional to limb length
            padding = int(limb_length * self.max_size_ratio)

            # Bounding box for occlusion
            x_min = max(min(x1, x2) - padding, 0)
            x_max = min(max(x1, x2) + padding, img.shape[1])
            y_min = max(min(y1, y2) - padding, 0)
            y_max = min(max(y1, y2) + padding, img.shape[0])

            # Random grayscale intensity occlusion (0–50)
            occlusion_color = np.random.randint(0, 50)

            if img.ndim == 3:
                img[y_min:y_max, x_min:x_max, :] = occlusion_color
            else:
                img[y_min:y_max, x_min:x_max] = occlusion_color

            occluded_count += 1

        results["img"] = img
        return results
    
@TRANSFORMS.register_module()
class RotateImage(BaseTransform):
    def __init__(self, rotation_prob=0.5, rotation_limits=[-15, 15]):
        self.prob = rotation_prob
        self.limit_min = rotation_limits[0]
        self.limit_max = rotation_limits[1]

    def transform(self, results):
        keypoints = results["keypoints"]

        # Handle visibility
        if keypoints.shape[1] == 3:
            keypoints_xy = keypoints[:, :2]
            visibility = keypoints[:, 2:3]
        else:
            keypoints_xy = keypoints
            visibility = None

        # Keep track of points that are exactly at (0,0)
        zero_mask = np.all(keypoints_xy == 0, axis=1)

        # Convert keypoints to list of tuples for Albumentations
        keypoints_list = [tuple(k) for k in keypoints_xy]

        transform = alb.Compose(
            [alb.Rotate(limit=[self.limit_min, self.limit_max], p=1.0)],
            keypoint_params=alb.KeypointParams(format="xy", remove_invisible=False)
        )

        # Apply augmentation
        aug = transform(image=results["img"], keypoints=keypoints_list)
        results["img"] = aug["image"]

        aug_kps = np.array(aug["keypoints"])

        # Reset any originally zero points back to (0,0)
        aug_kps[zero_mask] = 0

        # Combine with visibility if it exists
        if visibility is not None:
            results["keypoints"] = np.hstack([aug_kps, visibility])
        else:
            results["keypoints"] = aug_kps

        return results