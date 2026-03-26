from typing import Dict, Optional, Tuple
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
import numpy as np
import cv2
import albumentations as alb


@TRANSFORMS.register_module()
class BlurLimbs(BaseTransform):
    """Blur limbs randomly in the image."""
    def __init__(self, blur_prob=0.5, max_kernel=5, max_size=20):
        self.blur_prob = blur_prob
        self.max_kernel = max_kernel
        self.max_size = max_size

    def transform(self, results: Dict):
        img = results["img"].copy()
        keypoints = results["keypoints"]

        if keypoints.shape[-1] == 3:
            joints = keypoints[..., :2]
            vis = keypoints[..., 2]
        else:
            joints = keypoints
            vis = np.ones(joints.shape[:2])

        # get limbs to blur whole limb (hips, arms, legs)
        limb_pairs = [
            (7, 9), (8, 10), (5, 7), (6, 8), (11, 13), (12, 14),(13, 15), (14, 16)
        ]

        for person in range(joints.shape[0]):
            person_joints = joints[person]
            person_vis = vis[person]

            for j1, j2 in limb_pairs:
                if np.random.rand() >= self.blur_prob:
                    continue
                if person_vis[j1] == 0 or person_vis[j2] == 0:
                    continue

                x1, y1 = person_joints[j1].astype(int)
                x2, y2 = person_joints[j2].astype(int)

                length = int(np.linalg.norm([x2-x1, y2-y1]))
                if length < 5:
                    continue

                k = max(5, length // 4)
                if k % 2 == 0:
                    k += 1

                kernel = np.zeros((k, k))
                direction = np.random.choice(["horizontal", "vertical", "diag1", "diag2"])
                if direction == "horizontal":
                    kernel[k//2, :] = 1
                elif direction == "vertical":
                    kernel[:, k//2] = 1
                elif direction == "diag1":
                    np.fill_diagonal(kernel, 1)
                else:
                    np.fill_diagonal(np.fliplr(kernel), 1)
                kernel /= kernel.sum()

                padding = int(length * 0.3)
                x_min = max(min(x1, x2) - padding, 0)
                x_max = min(max(x1, x2) + padding, img.shape[1])
                y_min = max(min(y1, y2) - padding, 0)
                y_max = min(max(y1, y2) + padding, img.shape[0])

                roi = img[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                blurred = cv2.filter2D(roi, -1, kernel)
                img[y_min:y_max, x_min:x_max] = blurred

        results["img"] = img
        return results


@TRANSFORMS.register_module()
class OccludeLimbs(BaseTransform):
    """Randomly occlude limbs in the image."""
    def __init__(self, occlusion_prob=0.5, max_occluded_limbs=3, max_size_ratio=0.3):
        self.occlusion_prob = occlusion_prob
        self.max_occluded_limbs = max_occluded_limbs
        self.max_size_ratio = max_size_ratio

    def transform(self, results: Dict):
        img = results["img"].copy()
        keypoints = results["keypoints"]

        if keypoints.shape[-1] == 3:
            joints = keypoints[..., :2]
            vis = keypoints[..., 2]
        else:
            joints = keypoints
            vis = np.ones(joints.shape[:2])

        # get limbs to occlude whole limb (hips, arms, legs)
        limb_pairs = [
            (5,7),(6,8),(7,9),(8,10), (11,13),(12,14),(11,12),(13,15),(14,16)
        ]


        for person in range(joints.shape[0]):
            person_joints = joints[person]
            person_vis = vis[person]

            pairs = limb_pairs.copy()
            np.random.shuffle(pairs)
            occluded_count = 0

            for j1, j2 in pairs:
                if occluded_count >= self.max_occluded_limbs:
                    break
                if np.random.rand() > self.occlusion_prob:
                    continue
                if person_vis[j1] == 0 or person_vis[j2] == 0:
                    continue

                x1, y1 = person_joints[j1].astype(int)
                x2, y2 = person_joints[j2].astype(int)
                limb_length = int(np.linalg.norm([x2-x1, y2-y1]))
                if limb_length < 5:
                    continue

                padding = int(limb_length * self.max_size_ratio)
                x_min = max(min(x1,x2)-padding, 0)
                x_max = min(max(x1,x2)+padding, img.shape[1])
                y_min = max(min(y1,y2)-padding, 0)
                y_max = min(max(y1,y2)+padding, img.shape[0])

                occlusion_color = np.random.randint(0,50)
                if img.ndim == 3:
                    img[y_min:y_max, x_min:x_max, :] = occlusion_color
                else:
                    img[y_min:y_max, x_min:x_max] = occlusion_color

                occluded_count += 1

        results["img"] = img
        return results


@TRANSFORMS.register_module()
class RotateImage(BaseTransform):
    """Randomly rotate image and keypoints."""
    def __init__(self, rotation_prob=0.5, rotation_limits=[-15,15]):
        self.prob = rotation_prob
        self.limit_min = rotation_limits[0]
        self.limit_max = rotation_limits[1]

        self.aug_transform = alb.Compose(
            [alb.Rotate(limit=[self.limit_min, self.limit_max], p=self.prob)],
            keypoint_params=alb.KeypointParams(format="xy", remove_invisible=False)
        )

    def transform(self, results: Dict):
        img = results["img"].copy()
        keypoints = results["keypoints"].copy()

        if keypoints.shape[-1] == 3:
            coords = keypoints[..., :2]
            vis = keypoints[..., 2:]
        else:
            coords = keypoints
            vis = None

        N, K, _ = coords.shape
        zero_mask = np.all(coords == 0, axis=-1)

        coords_flat = coords.reshape(-1,2)
        keypoints_list = [tuple(p) for p in coords_flat]

        aug = self.aug_transform(image=img, keypoints=keypoints_list)
        img = aug["image"]

        aug_coords = np.array(aug["keypoints"], dtype=np.float32).reshape(N,K,2)
        aug_coords[zero_mask] = 0

        if vis is not None:
            keypoints[..., :2] = aug_coords
        else:
            keypoints = aug_coords

        results["img"] = img
        results["keypoints"] = keypoints
        return results