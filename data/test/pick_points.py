import cv2
import numpy as np
import os
import json
from pathlib import Path

DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(DIR, "images")
ANN_PATH = os.path.join(DIR, "annotations.json")

COCO_KEYPOINTS = [
    "nose",
    "left_eye","right_eye",
    "left_ear","right_ear",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
]

# -----------------------------
# Annotation UI
# -----------------------------
def annotate_image(img, keypoint_names, zoom=8, zoom_window=200):

    target_h = 800
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, target_h))

    base_img = img.copy()
    h, w = img.shape[:2]

    base_img = img.copy()
    h, w = img.shape[:2]

    phase = "bbox"
    cursor_pos = None

    bbox_start = None
    bbox_end = None
    drawing_bbox = False

    keypoints = []
    kp_idx = 0
    done = False

    def draw_ui():

        display = base_img.copy()

        if bbox_start and bbox_end:
            cv2.rectangle(display, bbox_start, bbox_end, (255,0,0), 2)

        for p in keypoints:
            if p is not None:
                px = int(p[0] * scale)
                py = int(p[1] * scale)
                cv2.circle(display, (px,py),5,(0,0,255),-1)

        cv2.rectangle(display,(0,0),(700,60),(0,0,0),-1)

        if phase == "bbox":
            text = "Step1: Draw bbox (drag) → ENTER"
        else:
            text = f"Keypoint: {keypoint_names[kp_idx]}" if kp_idx < len(keypoint_names) else "Done"

        cv2.putText(display,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        if cursor_pos:
            x,y = cursor_pos
            cv2.line(display,(x,0),(x,h),(255,255,0),1)
            cv2.line(display,(0,y),(w,y),(255,255,0),1)

        cv2.imshow("image",display)

        if cursor_pos:
            x,y = cursor_pos
            x1 = max(0,x-zoom_window//(2*zoom))
            y1 = max(0,y-zoom_window//(2*zoom))
            x2 = min(w,x+zoom_window//(2*zoom))
            y2 = min(h,y+zoom_window//(2*zoom))

            crop = base_img[y1:y2,x1:x2]

            if crop.size:
                zoomed = cv2.resize(crop,(zoom_window,zoom_window),interpolation=cv2.INTER_NEAREST)
                zc = zoom_window//2
                cv2.line(zoomed,(zc,0),(zc,zoom_window),(0,255,0),1)
                cv2.line(zoomed,(0,zc),(zoom_window,zc),(0,255,0),1)
                cv2.imshow("zoom",zoomed)

    def mouse(event,x,y,flags,param):

        nonlocal bbox_start,bbox_end,drawing_bbox
        nonlocal kp_idx,done,cursor_pos

        cursor_pos = (x,y)

        if phase == "bbox":

            if event == cv2.EVENT_LBUTTONDOWN:
                bbox_start = (x,y)
                bbox_end = (x,y)
                drawing_bbox = True

            elif event == cv2.EVENT_MOUSEMOVE and drawing_bbox:
                bbox_end = (x,y)

            elif event == cv2.EVENT_LBUTTONUP:
                bbox_end = (x,y)
                drawing_bbox = False

        else:

            if kp_idx < len(keypoint_names) and keypoint_names[kp_idx] in ["left_eye","right_eye","left_ear","right_ear"]:
                keypoints.append(None)
                kp_idx += 1
                draw_ui()
                return

            if event == cv2.EVENT_LBUTTONDOWN:

                if kp_idx < len(keypoint_names):
                    orig_x = x / scale
                    orig_y = y / scale
                    keypoints.append([float(orig_x), float(orig_y)])
                    # keypoints.append([float(x),float(y)])
                    kp_idx += 1

                    if kp_idx == len(keypoint_names):
                        done = True

        draw_ui()

    cv2.namedWindow("image")
    cv2.namedWindow("zoom")
    cv2.setMouseCallback("image",mouse)

    while True:

        draw_ui()
        key = cv2.waitKey(10)

        if key == 27:
            break

        if phase == "bbox" and key == 13 and bbox_start and bbox_end:
            phase = "keypoints"

        elif phase == "keypoints":

            if done:
                break

            elif key == 32:  # occluded
                keypoints.append(None)
                kp_idx += 1

            elif key == 8 and kp_idx>0:  # undo
                kp_idx -= 1
                keypoints.pop()

    cv2.destroyAllWindows()

    if bbox_start and bbox_end:
        x1,y1 = bbox_start
        x2,y2 = bbox_end
        bbox = [
            min(x1,x2) / scale,
            min(y1,y2) / scale,
            abs(x2-x1) / scale,
            abs(y2-y1) / scale
        ]
        # bbox = [min(x1,x2),min(y1,y2),abs(x2-x1),abs(y2-y1)]
    else:
        bbox = None

    return keypoints,bbox


# -----------------------------
# COCO helpers
# -----------------------------
def to_coco_annotation(image_id,annotation_id,keypoints,bbox):

    kp_out=[]
    visible=0

    for p in keypoints:
        if p is None:
            kp_out.extend([0,0,0])
        else:
            kp_out.extend([p[0],p[1],2])
            visible+=1

    return {
        "id":annotation_id,
        "image_id":image_id,
        "category_id":1,
        "bbox":bbox,
        "area":bbox[2]*bbox[3],
        "iscrowd":0,
        "keypoints":kp_out,
        "num_keypoints":visible
    }


# -----------------------------
# Load or create COCO file
# -----------------------------
if os.path.exists(ANN_PATH):

    with open(ANN_PATH) as f:
        coco=json.load(f)

else:

    coco={
        "images":[],
        "annotations":[],
        "categories":[{
            "id":1,
            "name":"person",
            "keypoints":COCO_KEYPOINTS,
            "skeleton":[]
        }]
    }


annotated_names=set(i["file_name"] for i in coco["images"])

image_files=sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg",".png",".jpeg"))
])


image_id=len(coco["images"])+1
annotation_id=len(coco["annotations"])+1


# -----------------------------
# Annotation loop
# -----------------------------
for file in image_files:

    if file in annotated_names:
        continue

    old_path=os.path.join(IMAGE_DIR,file)

    new_name=f"{image_id:05d}.jpg"
    new_path=os.path.join(IMAGE_DIR,new_name)

    if file!=new_name:
        os.rename(old_path,new_path)

    img=cv2.imread(new_path)

    print("Annotating:",new_name)

    keypoints,bbox=annotate_image(img,COCO_KEYPOINTS)

    coco["images"].append({
        "id":image_id,
        "file_name":new_name,
        "width":img.shape[1],
        "height":img.shape[0]
    })

    coco["annotations"].append(
        to_coco_annotation(image_id,annotation_id,keypoints,bbox)
    )

    image_id+=1
    annotation_id+=1

    with open(ANN_PATH,"w") as f:
        json.dump(coco,f,indent=2)


print("Annotation finished.")