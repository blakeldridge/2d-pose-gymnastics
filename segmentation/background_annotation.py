import cv2
import numpy as np
import json
import os

# ===== CONFIG =====
DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
IMAGE_DIR = os.path.join(DIR, "data/backgrounds/")
OUTPUT_DIR = os.path.join(DIR, "segmentation/annotations")
ANN_PATH = os.path.join(OUTPUT_DIR, "annotations.json")

DISPLAY_MAX_HEIGHT = 900

FOREGROUND_BRUSH = 10
PLACEMENT_BRUSH = 25
MIN_BRUSH = 3
MAX_BRUSH = 100

ZOOM_SIZE = 200
ZOOM_SCALE = 3

# ===== CALLBACK =====
def draw(event, x, y, flags, param):
    global drawing, mouse_x, mouse_y

    mouse_x, mouse_y = x, y

    brush = FOREGROUND_BRUSH if stage == "foreground" else PLACEMENT_BRUSH

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if eraser:
            cv2.circle(mask, (x, y), brush, 0, -1)
        else:
            cv2.circle(mask, (x, y), brush, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if eraser:
                cv2.circle(mask, (x, y), brush, 0, -1)
            else:
                cv2.circle(mask, (x, y), brush, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Annotator")
cv2.setMouseCallback("Annotator", draw)

foreground_mask_full = None
placement_mask_full = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(ANN_PATH):

    with open(ANN_PATH) as f:
        annotations=json.load(f)
else:
    annotations = []

image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if os.path.join(IMAGE_DIR, f) not in [background["image"] for background in annotations]]

for image_path in image_paths:
    # ===== LOAD ORIGINAL =====
    orig = cv2.imread(image_path)
    orig_h, orig_w = orig.shape[:2]

    # ===== RESIZE =====
    scale = DISPLAY_MAX_HEIGHT / orig_h if orig_h > DISPLAY_MAX_HEIGHT else 1.0
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    img = cv2.resize(orig, (disp_w, disp_h))

    # ===== MASK =====
    mask = np.zeros((disp_h, disp_w), dtype=np.uint8)

    drawing = False
    stage = "foreground"
    eraser = False

    mouse_x, mouse_y = 0, 0

    # ===== LOOP =====
    while True:
        display = img.copy()

        # Overlay mask
        overlay = np.zeros_like(display)
        if stage == "foreground":
            overlay[:, :, 2] = mask  # red
            brush = FOREGROUND_BRUSH
        else:
            overlay[:, :, 1] = mask  # green
            brush = PLACEMENT_BRUSH

        display = cv2.addWeighted(display, 1.0, overlay, 0.5, 0)

        # UI text
        mode_text = "Eraser" if eraser else "Draw"
        cv2.putText(display, f"Stage: {stage} | Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Brush: {brush} (+/- to adjust)", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "e = toggle eraser | n = next stage | c = clear | s = save | ESC = quit",
                    (10, disp_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Annotator", display)

        # Zoom window
        x1 = max(0, mouse_x - ZOOM_SIZE // 2)
        y1 = max(0, mouse_y - ZOOM_SIZE // 2)
        x2 = min(disp_w, mouse_x + ZOOM_SIZE // 2)
        y2 = min(disp_h, mouse_y + ZOOM_SIZE // 2)

        zoom = display[y1:y2, x1:x2]
        zoom = cv2.resize(zoom, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Zoom", zoom)

        key = cv2.waitKey(1) & 0xFF

        # ===== NEXT STAGE =====
        if key == ord('n'):
            if stage == "foreground":
                foreground_mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                print("Foreground done → placement stage")
                mask[:] = 0
                stage = "placement"
            elif stage == "placement":
                placement_mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                print("Placement done → press 's' to save")

        # ===== CLEAR =====
        elif key == ord('c'):
            mask[:] = 0

        # ===== SAVE =====
        elif key == ord('s'):
            if foreground_mask_full is None or placement_mask_full is None:
                print("Finish both stages first")
                continue

            base = os.path.splitext(os.path.basename(image_path))[0]

            placement_path = os.path.join(OUTPUT_DIR, f"{base}_placement.png")
            foreground_path = os.path.join(OUTPUT_DIR, f"{base}_foreground.png")

            cv2.imwrite(placement_path, placement_mask_full)
            cv2.imwrite(foreground_path, foreground_mask_full)

            data = {
                "image": image_path,
                "placement_mask": placement_path,
                "foreground_mask": foreground_path,
                "min_scale": 0.3,
                "max_scale": 1.2
            }

            annotations.append(data)

            with open(ANN_PATH, "w") as f:
                json.dump(annotations, f, indent=4)
            print("Saved!")

        # ===== BRUSH SIZE ADJUST =====
        elif key == ord('+') or key == ord('='):
            if stage == "foreground":
                FOREGROUND_BRUSH = min(MAX_BRUSH, FOREGROUND_BRUSH + 2)
            else:
                PLACEMENT_BRUSH = min(MAX_BRUSH, PLACEMENT_BRUSH + 2)

        elif key == ord('-'):
            if stage == "foreground":
                FOREGROUND_BRUSH = max(MIN_BRUSH, FOREGROUND_BRUSH - 2)
            else:
                PLACEMENT_BRUSH = max(MIN_BRUSH, PLACEMENT_BRUSH - 2)

        # ===== ERASER TOGGLE =====
        elif key == ord('e'):
            eraser = not eraser
            print("Eraser mode:", eraser)

        # ===== ESC =====
        elif key == 27:
            break

    cv2.destroyAllWindows()