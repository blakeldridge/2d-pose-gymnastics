import cv2
import os
import json
import numpy as np
from tracking.vitpose import ViTPose
from tracking.smoothing import savgol_smoothing, kalman_smoothing
from utils.visualisation import plot_skeleton

DIR = os.path.dirname(os.path.abspath(__file__))

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(filename, frame)

        frame_id += 1

    cap.release()
    print(f"Extracted {frame_id} frames")

if __name__ == "__main__":
    video_path = os.path.join(DIR, "test.mp4")
    frame_dir = os.path.join(DIR, "test_frames")
    keypoints_path = os.path.join(DIR, "results.json")

    if not os.path.exists(frame_dir):
        extract_frames(video_path, frame_dir)

    if not os.path.exists(keypoints_path):
        model = ViTPose()

        frame_paths = sorted([os.path.join(frame_dir, fname) for fname in os.listdir(frame_dir)])
        
        num_images = len(frame_paths)
        batch_size = 10

        keypoints = {}
        for i in range(0, num_images, batch_size):
            batch_start = i
            batch_end = i+batch_size

            batch_paths = frame_paths[batch_start:batch_end]
            output = model(batch_paths, image_loc="device")
            for j in range(len(batch_paths)):
                keypoints[batch_paths[j]] = output[j][0]["keypoints"].astype(float).tolist()

            print(f"Extract keypoints {batch_start} to {batch_end}")
    
        with open(keypoints_path, "w") as f:
            json.dump(keypoints, f)
    else:
        with open(keypoints_path, "r") as f:
            keypoints = json.load(f)

    sorted_frames = sorted(keypoints.keys())
    keypoints = [np.array(keypoints[key]) for key in sorted_frames]

    max_jump = 800
    savgol_window = 7
    savgol_polyorder = 4

    savgol_kpts = savgol_smoothing(np.array(keypoints), max_jump, savgol_window, savgol_polyorder)
    kalman_kpts = kalman_smoothing(np.array(keypoints), max_jump)

    keyframes = ["frame_0430.jpg", "frame_0511.jpg", "frame_0537.jpg", "frame_0557.jpg", "frame_0579.jpg", "frame_0609.jpg"]
    begin = "/home/blake-eldridge/Projects/2d-pose-gymnastics/tracking/test_frames/"
    keyframe_idxs = [sorted_frames.index(begin + keyframe) for keyframe in keyframes]

    for idx in range(len(keyframes)):
        plot_skeleton(begin+keyframes[idx], [keypoints[keyframe_idxs[idx]], savgol_kpts[keyframe_idxs[idx]]])
