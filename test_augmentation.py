from mmpose.datasets.transforms import LoadImage, RandomBBoxTransform, GetBBoxCenterScale, Albumentation, RandomFlip
from mmpose.datasets.transforms.gymnastics_transforms import BlurLimbs, OccludeLimbs
import mmcv
import numpy as np

"""keypoints_list = [
    367,81,2, 374,73,2, 360,75,2, 386,78,2, 356,81,2,
    399,108,2, 358,129,2, 433,142,2, 341,159,2,
    449,165,2, 309,178,2, 424,203,2,
    393,214,2, 429,294,2, 367,273,2,
    466,362,2, 396,341,2
]"""

keypoints_list =  [
                297,
                111,
                2,
                299,
                106,
                2,
                0,
                0,
                0,
                314,
                108,
                2,
                0,
                0,
                0,
                329,
                141,
                2,
                346,
                125,
                2,
                295,
                164,
                2,
                323,
                130,
                2,
                266,
                155,
                2,
                279,
                143,
                2,
                329,
                225,
                2,
                331,
                221,
                2,
                327,
                298,
                2,
                283,
                269,
                2,
                398,
                327,
                2,
                288,
                349,
                2
            ],

bbox = [
    247.76,
    74.23,
    169.67,
    300.78
]
# Convert to numpy
keypoints_array = np.array(keypoints_list).reshape(17, 3)
bbox_array = np.array(bbox)

# Split coordinates and visibility
keypoints_xy = keypoints_array[:, :2]        # (17, 2)
visibility = keypoints_array[:, 2:3]        # (17, 1)

# MMPose expects shape: (num_instances, num_joints, 2)
keypoints_xy = keypoints_xy[np.newaxis, :, :][0]
visibility = visibility[np.newaxis, :, :]

# Load the original image from the path
results = dict(
    img_path='/home/blake-eldridge/Repos/mmpose/tests/data/coco/000000196141.jpg',
    keypoints=keypoints_xy,
    bbox=bbox_array
)

transform = LoadImage()
results = transform(results)

transform = GetBBoxCenterScale()
results = transform(results)

# transform = RandomFlip(direction="horizontal")
# results = transform(results)

transform = BlurLimbs(blur_prob=0.3)
results = transform(results)

transform = OccludeLimbs(occlusion_prob=0.4, max_size_ratio=0.1)
results = transform(results)

transform = Albumentation(transforms=[
    dict(type="Rotate", limit=(-180, 180), p=1),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=1),
    dict(type='Perspective', scale=(0.05, 0.15), p=1)
])
results = transform(results)

import matplotlib.pyplot as plt
plt.imshow(results['img'][:, :, ::-1])
plt.show()