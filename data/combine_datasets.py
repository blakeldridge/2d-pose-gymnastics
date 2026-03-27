import json
import os
import shutil
import random
from tqdm import tqdm

synthetic_json = "segmentation_images/10Kannotations.json"
synthetic_img_dir = "segmentation_images/images"

coco_json = "coco/annotations/person_keypoints_train2017.json"
coco_img_dir = "coco/train2017"

output_dir = "merged_dataset"
output_img_dir = os.path.join(output_dir, "images")
output_json = os.path.join(output_dir, "annotations.json")

num_coco_samples = 1000
num_synthetic_samples = 2000
random.seed(42)

os.makedirs(output_img_dir, exist_ok=True)

with open(synthetic_json) as f:
    synth = json.load(f)

with open(coco_json) as f:
    coco = json.load(f)

if os.path.exists(output_json):
    print("Found existing merged dataset. Loading...")
    with open(output_json) as f:
        merged = json.load(f)
    new_image_id = max(img["id"] for img in merged["images"]) + 1
    new_ann_id = max(ann["id"] for ann in merged["annotations"]) + 1
else:
    merged = {
        "images": [],
        "annotations": [],
        "categories": synth["categories"]
    }
    new_image_id = 0
    new_ann_id = 0
def copy_image(src_path, dst_path):
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

print("Sampling COCO data...")

synth_anns_per_img = {}
for ann in synth["annotations"]:
    synth_anns_per_img.setdefault(ann["image_id"], []).append(ann)

valid_images = [img for img in synth["images"] if img["id"] in synth_anns_per_img]

sampled_images = random.sample(valid_images, num_synthetic_samples)

print("Adding synthetic data...")

img_id_map = {}

for img in tqdm(sampled_images):
    old_id = img["id"]

    new_filename = f"synth_{img['file_name']}"
    src = os.path.join(synthetic_img_dir, img["file_name"])
    dst = os.path.join(output_img_dir, new_filename)

    copy_image(src, dst)

    new_img = img.copy()
    new_img["id"] = new_image_id
    new_img["file_name"] = new_filename

    merged["images"].append(new_img)
    img_id_map[old_id] = new_image_id

    new_image_id += 1

    for ann in synth_anns_per_img[old_id]:
        new_ann = ann.copy()
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = img_id_map[old_id]

        merged["annotations"].append(new_ann)
        new_ann_id += 1

print("Sampling COCO data...")

coco_anns_per_img = {}
for ann in coco["annotations"]:
    coco_anns_per_img.setdefault(ann["image_id"], []).append(ann)

valid_images = [img for img in coco["images"] if img["id"] in coco_anns_per_img]

sampled_images = random.sample(valid_images, num_coco_samples)

print("Adding COCO data...")

img_id_map = {}

for img in tqdm(sampled_images):
    old_id = img["id"]

    new_filename = f"coco_{img['file_name']}"
    src = os.path.join(coco_img_dir, img["file_name"])
    dst = os.path.join(output_img_dir, new_filename)

    copy_image(src, dst)

    new_img = img.copy()
    new_img["id"] = new_image_id
    new_img["file_name"] = new_filename

    merged["images"].append(new_img)
    img_id_map[old_id] = new_image_id

    new_image_id += 1

    for ann in coco_anns_per_img[old_id]:
        new_ann = ann.copy()
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = img_id_map[old_id]

        merged["annotations"].append(new_ann)
        new_ann_id += 1

print("Saving merged dataset...")

with open(output_json, "w") as f:
    json.dump(merged, f)

print("Done!")
print(f"Total images: {len(merged['images'])}")
print(f"Total annotations: {len(merged['annotations'])}")

