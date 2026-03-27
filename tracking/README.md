# Pose Smoothing Pipeline

## Overview
This folder contains code to smooth keypoints by tracking through a video and using it to smooth over keypoints in an attempt to see any accuracy improvements
- Savitzky-Golay filter
- Kalman filter

## Requirements
- numpy, scipy, filterpy, opencv, torch, huggingface transformers
- ViTPose (pretrained)

## Usage

Visualise keypoints for gt, Savitzky-Golay and kalman for keyframes
```bash
python -m tracking.test