import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

coco_connections = [
    [0,1], [0,2], [1,3], [2,4], [5,6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]
]
def plot_skeleton(frame_path, p2ds, connections=coco_connections, shoulders=[5, 6], side_by_side=False, bbox=None):
    frame = cv2.imread(frame_path)
    frame_rgb = frame[:, :, ::-1]

    colours = [
        "blue","orange","green","red","purple","brown","pink","gray","olive","cyan"
    ]

    # draw skeleton on graph next to frame
    if side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_img, ax_pose = axes
        ax_img.imshow(frame_rgb)
        ax_img.set_title("Image + Pose")
        ax_img.axis("off")

        ax_pose.set_title("Pose")
        ax_pose.set_aspect("equal")
        ax_pose.grid(True)
    else:
        fig, ax_img = plt.subplots()
        ax_img.imshow(frame_rgb)
        ax_img.axis("off")

    for colour, skeleton in zip(colours, p2ds):

        # scatter keypoints of skeleton
        ax_img.scatter(skeleton[:,0], skeleton[:,1], c=colour, s=5)
        if side_by_side:
            ax_pose.scatter(skeleton[:,0], skeleton[:,1], c=colour, s=10)

        # draw skeleton lines / connections
        for c in connections:
            x = [skeleton[c[0],0], skeleton[c[1],0]]
            y = [skeleton[c[0],1], skeleton[c[1],1]]

            ax_img.plot(x, y, colour, linewidth=1)

            if side_by_side:
                ax_pose.plot(x, y, colour, linewidth=1)

        # join nose to shoulder midpoint
        shoulder_mid = (skeleton[shoulders[0]] + skeleton[shoulders[1]]) / 2
        nose = skeleton[0]

        x = [nose[0], shoulder_mid[0]]
        y = [nose[1], shoulder_mid[1]]

        ax_img.plot(x, y, colour, linewidth=1)

        if side_by_side:
            ax_pose.plot(x, y, colour, linewidth=1)

    # draw bbox around person
    if bbox is not None:
        x, y, w, h = bbox
        x1, x2 = x, x + w
        y1, y2 = y, y + h
        ax_img.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "red", linewidth=1)

    if side_by_side:
        ax_pose.invert_yaxis()

    plt.tight_layout()
    plt.show()
    plt.show()

def image_with_joints(image_path, joints):
    image = cv2.imread(image_path)

    for pt in joints:
        cv2.circle(image, pt.astype(int), radius=2, color=(0,255,0), thickness=-1)

    cv2.imshow("Joint-Image Visualisation", image)
    cv2.waitKey(0)

def draw_comparison(image1_path, image2_path, keypoints1, keypoints2):
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    image1_with_keypoints = draw_keypoints(image1, keypoints1)
    image2_with_keypoints = draw_keypoints(image2, keypoints2)

    combined = concat_side_by_side(image1_with_keypoints, image2_with_keypoints)
    return combined

def draw_keypoints(img, keypoints, radius=4, color="red"):
    """Draw circles for keypoints onto a PIL image."""
    draw = ImageDraw.Draw(img)
    for (x, y) in keypoints:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color
        )
    return img

def concat_side_by_side(img1, img2):
    """Concatenate two PIL images horizontally."""
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_img = Image.new("RGB", (w1 + w2, max(h1, h2)))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w1, 0))
    return new_img

