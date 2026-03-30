import bpy
import random
import math
import os
import json
import bpy_extras
import re
import os

OUTPUT_DIR = "/home/blake-eldridge/Projects/cmu-pose/outputs/"

KEY_FRAMES = [163, 195, 197, 202, 212, 218, 256]

ANGLES_PER_POSE = 5

IMAGE_SIZE = 512

CAMERA_NAME = "Camera"
ARMATURE_NAME = "CharacterArmature"

scene = bpy.context.scene
camera = bpy.data.objects[CAMERA_NAME]
armature = bpy.data.objects[ARMATURE_NAME]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set resolution
scene.render.resolution_x = IMAGE_SIZE
scene.render.resolution_y = IMAGE_SIZE
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True

def look_at(cam, target):
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def get_hip_center():
    bones = armature.pose.bones

    if "LeftUpLeg" in bones and "RightUpLeg" in bones:
        l = armature.matrix_world @ bones["LeftUpLeg"].head
        r = armature.matrix_world @ bones["RightUpLeg"].head
        return (l + r) / 2

    elif "Hips" in bones:
        return armature.matrix_world @ bones["Hips"].head

    else:
        return armature.location

def randomize_camera(cam):
    radius = random.uniform(45, 60)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(35), math.radians(75))

    x = radius * math.cos(theta) * math.sin(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(phi)

    cam.location = (x, y, z)
    target = get_hip_center()
    look_at(cam, target)
    
BONE_MAP = {
    "Head": "nose",
    "Neck": "neck",
    "UpperArm.L": "l_shoulder",
    "UpperArm.R": "r_shoulder",
    "LowerArm.L": "l_elbow",
    "LowerArm.R": "r_elbow",
    "Hand.L": "l_wrist",
    "Hand.R": "r_wrist",
    "UpperLeg.L": "l_hip",
    "UpperLeg.R": "r_hip",
    "LowerLeg.L": "l_knee",
    "LowerLeg.R": "r_knee",
    "Foot.L": "l_ankle",
    "Foot.R": "r_ankle"
}

def get_keypoints():
    keypoints = {}

    for bone in armature.pose.bones:
        if bone.name not in BONE_MAP:
            continue

        mapped_name = BONE_MAP[bone.name]

        # get bone world head/tail
        head_world = armature.matrix_world @ bone.head
        tail_world = armature.matrix_world @ bone.tail
        direction = (tail_world - head_world).normalized()

        # default position = head
        world_pos = head_world.copy()

        # directional offsets
        if mapped_name == "nose":
            head_bone = armature.pose.bones["Head"]
            head = armature.matrix_world @ head_bone.head
            tail = armature.matrix_world @ head_bone.tail
            
            mid = (head + tail) / 2

            world_pos = mid 
            
        elif mapped_name in ["l_ankle", "r_ankle"]:
            knee_name = "LowerLeg.L" if mapped_name == "l_ankle" else "LowerLeg.R"
            knee_bone = armature.pose.bones[knee_name]

            knee_head = armature.matrix_world @ knee_bone.head
            knee_tail = armature.matrix_world @ knee_bone.tail

            leg_dir = (knee_tail - knee_head).normalized()
            offset_distance = 1
            world_pos = knee_tail - leg_dir * offset_distance

        else:
            world_pos = head_world.copy()

        # project to camera
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_pos)
        x = co_2d.x
        y = 1 - co_2d.y
        visible = 1 if 0 <= x <= 1 and 0 <= y <= 1 else 0

        keypoints[mapped_name] = {
            "x": float(x),
            "y": float(y),
            "visible": visible
        }

    return keypoints

existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("img_") and f.endswith((".png", ".json"))]

if existing_files:
    # extract numbers from filenames
    ids = [int(re.search(r"img_(\d+)", f).group(1)) for f in existing_files]
    img_id = max(ids) + 1
else:
    img_id = 0

for frame in KEY_FRAMES:
    scene.frame_set(frame)

    # small random rotation for variation
    armature.rotation_euler[2] = random.uniform(-0.3, 0.3)

    for i in range(ANGLES_PER_POSE):
        randomize_camera(camera)

        filename = f"img_{img_id:05d}"
        img_path = os.path.join(OUTPUT_DIR, filename + ".png")
        json_path = os.path.join(OUTPUT_DIR, filename + ".json")

        scene.render.filepath = img_path

        bpy.ops.render.render(write_still=True)

        keypoints = get_keypoints()

        with open(json_path, "w") as f:
            json.dump(keypoints, f, indent=2)

        img_id += 1

print(f"Done! Generated {img_id} images.")