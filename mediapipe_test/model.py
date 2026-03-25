import mediapipe as mp
from PIL import Image
import requests
import numpy as np
import os
MP_TO_COCO_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
DIR = os.path.abspath(os.path.dirname(__file__))

class MediaPipe:
    def __init__(self, model_path=os.path.join(DIR, "pose_landmarker_full.task")):
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1
        )

    def load_mp_image(self, path, image_loc):
        if image_loc == "device":
            image = Image.open(path).convert("RGB")
        else:
            image = Image.open(requests.get(path, stream=True).raw).convert("RGB")

        image_arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        return mp_image, image_arr

    def load_images(self, image_paths, image_loc):
        images = []
        image_arrays = []
        if isinstance(image_paths, list):
            for path in image_paths:
                mp_img, np_img = self.load_mp_image(path, image_loc)
                images.append(mp_img)
                image_arrays.append(np_img)
        else:
            mp_img, np_img = self.load_mp_image(image_paths, image_loc)
            images.append(mp_img)
            image_arrays.append(np_img)
        return images, image_arrays

    def crop_images(self, images, image_arrays, bboxes, pad=0):
        """
        Crop a batch of images and update both PIL/MediaPipe images and np arrays.
        Args:
            images: list of mp.Image
            image_arrays: list of np.ndarray
            bboxes: list of [x, y, w, h] COCO-style
            pad: int pixels to pad around bbox
        Returns:
            cropped_mp_images, cropped_np_images
        """
        cropped_mp_images = []
        cropped_np_images = []

        for img, img_arr, bbox in zip(images, image_arrays, bboxes):
            x, y, w, h = bbox
            h_img, w_img = img_arr.shape[:2]

            x1 = max(0, int(x - pad))
            y1 = max(0, int(y - pad))
            x2 = min(w_img, int(x + w + pad))
            y2 = min(h_img, int(y + h + pad))

            cropped_np = img_arr[y1:y2, x1:x2]
            cropped_np_images.append(cropped_np)

            cropped_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_np)
            cropped_mp_images.append(cropped_mp)

        return cropped_mp_images, cropped_np_images

    def estimate_pose(self, images):
        poses = []
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            for img in images:
                img_poses = []
                pose_landmarker_result = landmarker.detect(img)

                if not pose_landmarker_result.pose_landmarks:
                    # No poses detected → fallback
                    img_poses.append({
                        "keypoints": np.zeros((33, 2)),
                        "score": 0
                    })
                    poses.append(img_poses)
                    continue

                for person in pose_landmarker_result.pose_landmarks:
                    keypoints_list = []
                    confidence_list = []

                    for landmark in person:
                        keypoints_list.append([landmark.x, landmark.y])
                        confidence_list.append(landmark.visibility)

                    keypoints = np.array(keypoints_list)
                    score = float(np.mean(confidence_list))

                    img_poses.append({"keypoints": keypoints, "score": score})
                poses.append(img_poses)
        return poses
    
    def keypoints_to_pixels(self, keypoints, image_arrays, bboxes=None, pad=0):
        """
        Converts normalized keypoints [0,1] to pixel coordinates in the original image.
        If bboxes were used to crop the image, offset the keypoints back to the original image.
        
        Args:
            keypoints: list of list of dicts from convert_coco_format
            image_arrays: list of np.ndarray (original images or cropped images)
            bboxes: list of [x, y, w, h] used for cropping (optional)
            pad: padding used when cropping (default=0)
        """
        for i, image in enumerate(image_arrays):
            height, width = image.shape[:2]
            for pose in keypoints[i]:
                keypoints_norm = pose["keypoints"]
                keypoints_px = np.zeros_like(keypoints_norm)

                # convert normalized coords to cropped image pixels
                keypoints_px[:, 0] = keypoints_norm[:, 0] * width
                keypoints_px[:, 1] = keypoints_norm[:, 1] * height

                # if cropped, shift keypoints back to original image coordinates
                if bboxes is not None:
                    x_offset = bboxes[i][0] - pad
                    y_offset = bboxes[i][1] - pad
                    keypoints_px[:, 0] += x_offset
                    keypoints_px[:, 1] += y_offset

                pose["keypoints"] = keypoints_px

        return keypoints

    def convert_coco_format(self, mp_results):
        coco_results = []
        for image in mp_results:
            image_results = []
            for person in image:
                mp_keypoints = person["keypoints"] 
                score = person["score"]

                coco_keypoints = mp_keypoints[MP_TO_COCO_IDX]

                image_results.append({
                    "keypoints": coco_keypoints,
                    "score": score
                })
            coco_results.append(image_results)
        return coco_results

    def __call__(self, image_paths, bboxes=None, image_loc="device"):
        images, image_arrays = self.load_images(image_paths, image_loc)
        mp_keypoints = self.estimate_pose(images)
        keypoints = self.convert_coco_format(mp_keypoints)
        keypoints = self.keypoints_to_pixels(keypoints, image_arrays, bboxes=None)
        return keypoints