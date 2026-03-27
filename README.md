# Improving Monocular 2D Pose Estimation in Gymnastics

## Overview
This repository contains the code used to improve monocular 2d pose estimation in gymnastics by experimenting with synthetic datasets, heavy data augmentation, preprocessing, state of the art models and 2D tracking to make up for the lack of pose data in sports.

Gymnastics contains many difficulties that make pose estimation accuracy suffer. For instance :
- Complex orientations and poses
- Heavy motion blur due to dynamic movements
- Apparatus occlusion of joints
- Self-occlusion

By using different augmentation and data, I have finetuned SOTA models HRNet and ViTPose to see how the accuracy can be improved upon.

This Repository is based on the mmpose repo.

## Alterations

New training configs added for finetuning using different data and data augmentations :
- configs/gymnastics/

Custom augmentations to rotate images, motion blur and occlude limbs :
- mmpose/datasets/transforms/gymnastics_transforms.py

Synthetic Dataset generation pipeline
- segmentation/

Deblurring images using NAFNet
- deblur/NAFNet_image_deblurring.ipynb

2D pose tracking / smoothing:
- tracking/

Extra metrics to aid visualisation of accuracy on different types of poses (dynamic vs static, occluded vs not occluded, motion blurred vs not motion blurred)
- utils/

Tested common baseline model Mediapipe on test dataset to unveil shortcoming of current out of the box solutions :
- mediapipe_test/

Manually annotated gymnastics test dataset for main 13 keypoints (excluding eyes and ears) and type of pose :
- data/test/pick_points.py

## Usage

1. Train model on gymnastics finetuning config:
```bash
python tools/train.py <config-path>
```
Example config path :
- configs/gymnastics/finetune_hrnet_coco.py

2. Test model results 
```bash
python tools/test.py <config-path> <path-to-checkpoint> --out <path-to-result-dir>/metrics.json --dump <path-to-result-dir>/outputs.pkl
```
> to test on deblurred images, update config test dataloader image directory to "*deblurred_images/*"

Example paths used :
- config-path -> configs/gymnastics/finetune_hrnet_coco.py
- path-to-checkpoint -> models/hrnet_coco/checkpoint.pth
- path-to-result-dir -> results/hrnet_coco/

3. Generate synthetic data
- Install segment_anything (SAM) Repo
- Download COCO dataset
- Download or gather background images
- use "*segmentation/background_annotation.py*" script to annotate background foreground and placement masks

```bash
python -m segmentation.build_dataset
```

4. Test tracking of keypoints
- Save video in question to "*tracking/test.mp4*"
- remove directory "*tracking/test_frames/*"
- remove annotations "*tracking/results.json*"
- update keyframes you wish to visualise

```bash
python -m tracking.test
```

5. Add more annotations to test dataset
- add images to "*data/test/images/*"

```bash
python -m data.test.pick_points
```

6. Deblur more images using google colab
- upload "*deblur/NAFNet_image_deblurring.ipynb*" to google colab
- run cloning of repo and add input and output folders
- upload all images to input folder
- run rest of the cells and move output zip to "*data/test/deblurred_images/*"

7. Getting metrics from output results for each experiment and each type of pose in dataset
- Run models for each experiment + add experiments names to "*utils/evaluate.py*"

```bash
python -m utils.evaluate
```