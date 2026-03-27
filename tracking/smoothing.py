import numpy as np
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter

def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    kf.P *= 1000.0
    kf.R = np.eye(2) * 5
    kf.Q = np.eye(4) * 0.01

    return kf

def smooth_keypoints_kalman(keypoints):
    frames, num_joints, _ = keypoints.shape
    smoothed = np.zeros_like(keypoints)

    filters = [create_kalman_filter() for _ in range(num_joints)]

    for j in range(num_joints):
        filters[j].x = np.array([
            keypoints[0, j, 0],
            keypoints[0, j, 1],
            0, 0
        ])

    for t in range(frames):
        for j in range(num_joints):
            kf = filters[j]
            kf.predict()

            measurement = keypoints[t, j]
            kf.update(measurement)

            smoothed[t, j] = kf.x[:2]

    return smoothed


def smooth_keypoints_savgol(keypoints, window_length=7, polyorder=2):
    smoothed = np.copy(keypoints)
    
    for j in range(keypoints.shape[1]):
        for dim in range(2):
            smoothed[:, j, dim] = savgol_filter(
                keypoints[:, j, dim],
                window_length=window_length,
                polyorder=polyorder
            )
    
    return smoothed

def clamp_jumps(keypoints, max_jump=50):
    clamped = keypoints.copy()
    
    for t in range(1, len(keypoints)):
        for j in range(keypoints.shape[1]):
            dist = np.linalg.norm(clamped[t, j] - clamped[t-1, j])
            if dist > max_jump:
                clamped[t, j] = clamped[t-1, j]
    
    return clamped

def savgol_smoothing(keypoints, max_jump, window, polyorder):
    keypoints = clamp_jumps(keypoints, max_jump)
    smoothed = smooth_keypoints_savgol(keypoints, window, polyorder)
    return smoothed

def kalman_smoothing(keypoints, max_jump):
    keypoints = clamp_jumps(keypoints, max_jump)
    smoothed = smooth_keypoints_kalman(keypoints)
    return smoothed