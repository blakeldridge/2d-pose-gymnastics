import cv2
import numpy as np
import json
import os

points = []
image = None
clone = None
image_list = []
current_index = 0

DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
JSON_PATH = os.path.join(DIR, "segmentation/annotations/annotations.json")


import cv2
import numpy as np
import json
import os

points = []
image = None
clone = None
image_list = []
current_index = 0
scale_factor = 1.0


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [item['image'] for item in data]


def resize_with_aspect(img, max_height=900):
    global scale_factor

    h, w = img.shape[:2]

    if h <= max_height:
        scale_factor = 1.0
        return img

    scale_factor = max_height / h
    new_w = int(w * scale_factor)
    new_h = max_height

    resized = cv2.resize(img, (new_w, new_h))
    return resized


def load_image(index):
    global image, clone, points

    if index < 0 or index >= len(image_list):
        print("No more images.")
        return False

    path = image_list[index]
    img = cv2.imread(path)

    if img is None:
        print(f"Failed to load: {path}")
        return False

    resized = resize_with_aspect(img)

    image = resized
    clone = resized.copy()
    points = []

    print(f"\nLoaded [{index+1}/{len(image_list)}]: {path}")
    print(f"Scale factor: {scale_factor:.4f}")

    return True


def click_event(event, x, y, flags, param):
    global points, image, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)} (display): ({x}, {y})")

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)

            p1 = np.array(points[0])
            p2 = np.array(points[1])
            display_distance = np.linalg.norm(p1 - p2)

            # Convert back to original image scale
            original_distance = display_distance / scale_factor

            print(f"Distance (display): {display_distance:.2f} px")
            print(f"Distance (original): {original_distance:.2f} px")

        cv2.imshow("Image", image)


def main():
    global current_index, image_list

    json_path = JSON_PATH

    if not os.path.exists(json_path):
        print("JSON file not found.")
        return

    image_list = load_json(json_path)

    if len(image_list) == 0:
        print("No images found in JSON.")
        return

    if not load_image(current_index):
        return

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_event)

    print("Instructions:")
    print("- Click two points to measure distance")
    print("- Press 'c' to clear")
    print("- Press 'n' for next image")
    print("- Press 'q' to quit")

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            load_image(current_index)
            print("Cleared points")

        elif key == ord('n'):
            current_index += 1
            if current_index >= len(image_list):
                print("Reached end of image list")
                current_index = len(image_list) - 1
            else:
                load_image(current_index)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
