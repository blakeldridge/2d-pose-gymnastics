# Synthetic Dataset Generation Pipeline

## Overview
This folder contains the files to generate a synthetic dataset through the following pipeline
- segment person from COCO background
- perform rotate and shear transforms on person and keypoints
- adjust blur, noise, brightness and contrast to match with background
- erode and feather blend edges to seamlessly be placed into the background
- place "within" the background by placing annotated person segment randomly on background and use foreground mask to keep apparatus occlusion

## Requirements
- numpy, opencv, matplotlib
- SAM (segment anything)
- COCO dataset images + annotations
- Background images + annotations

Background annotation example :
```json
    {
        "image": "data/backgrounds/background1.jpg",
        "placement_mask": "segmentation/annotations/background1_placement.png",
        "foreground_mask": "segmentation/annotations/background1_foreground.png",
        "min_height": 1150,
        "max_height": 1500
    },
```
> where forground mask is an image mask showing where the apparatus is, and placement mask is an image mask showing where a person can be placed in the scene

- Background images kept in data/backgrounds directory
- Background annotations kept in segmentation/annotations directory
- COCO dataset kept in data/coco/ directory (data/coco/train2017/ for images and data/coco/annotations/person_keypoints_train2017.json for annotaitons)

## Usage

1. Generate dataset:
```bash
python -m segmentation.build_dataset