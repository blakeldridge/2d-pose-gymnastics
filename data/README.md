# Data README

## Overview
This folder would contain all image and annotations for the data used within the project

This includes
- COCO dataset 2017
- AthletePose3D 2d pose dataset
- synthetic image dataset
- backgrounds used in synthetic dataset
- test dataset manually annotated
- deblurred version of test dataset
- merged dataset combining COCO and synthetic datasets
- frames for the videos used in test dataset for tracking experiments

It also contains many handy programs used to manually annotate images and combine datasets for mixed training

## Folder structure
- backgrounds : contains background images used in synthetic dataset generation
- coco
    - train2017 : training images 
    - val2017 : validation iamges
    - annotations
        - person_keypoints_train2017.json : training images annotations
        - person_keypoints_val2017.json : validation images annotations
- segmentation_images
    - images : contains generated synthetic images
    - annotations.json : contains annotations of synthetic images in coco format
- test
    - images : contains test images
    - deblurred_images : contains deblurred versions of test images
    - frames : contains video frames for all videos in test dataset
    - annotations.json : contains annotations for test images
    - pick_points.py : program used to manually annotate test images