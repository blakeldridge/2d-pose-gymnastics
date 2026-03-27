import json
import random
import os

DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_JSON = os.path.join(DIR, "annotations.json")
TRAIN_JSON = os.path.join(DIR, "train.json")
TEST_JSON = os.path.join(DIR, "test.json")
NUM_TRAIN = 100
SEED = 42

with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco.get("categories", [])

random.seed(SEED)
random.shuffle(images)

train_images = images[:NUM_TRAIN]
test_images = images[NUM_TRAIN:]

train_ids = set(img["id"] for img in train_images)
test_ids = set(img["id"] for img in test_images)

train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
test_annotations = [ann for ann in annotations if ann["image_id"] in test_ids]

train_coco = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}

test_coco = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

with open(TRAIN_JSON, "w") as f:
    json.dump(train_coco, f, indent=4)

with open(TEST_JSON, "w") as f:
    json.dump(test_coco, f, indent=4)

print(f"Done!")
print(f"Train images: {len(train_images)}, annotations: {len(train_annotations)}")
print(f"Test images: {len(test_images)}, annotations: {len(test_annotations)}")
