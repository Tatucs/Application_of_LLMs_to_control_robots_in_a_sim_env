"""overhead_camera controller.

A plain Robot that owns one overhead Camera with Recognition. Once per
SAVE_PERIOD of sim time it writes into <repo>/data/camera/:
  image.jpg          raw RGB frame                       (L3)
  segmentation.png   objects painted in recognitionColors (L2)
  recognition.json   recognised objects, camera + world frame (L1)
No Supervisor: the world position comes from CAM_POS / CAM_ROT, which must
match the Robot node in the .wbt file.
"""
import json
import math
import os
import time
from pathlib import Path

from controller import Robot

# ---------------------------------------------------------------- Constants

CAMERA_NAME = "overhead_camera"            # Camera node name in the .wbt
CAM_POS = (0.5, 0.0, 2)                  # Robot translation in the .wbt, in metres
# Robot rotation in the .wbt (axis, angle): turns the camera view axis +x to world -z
CAM_ROT = (0.0, 1.0, 0.0, math.pi / 2)
CAMERA_STEPS = 4                           # camera + recognition sampled every this many time steps
SAVE_PERIOD = 0.25                         # s sim time between saved frames
JPEG_QUALITY = 90

# ---------------------------------------------------------------- Setup

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice(CAMERA_NAME)
camera.enable(CAMERA_STEPS * timestep)
if camera.hasRecognition():
    camera.recognitionEnable(CAMERA_STEPS * timestep)
    if camera.hasRecognitionSegmentation():
        camera.enableRecognitionSegmentation()
else:
    print("WARNING: camera has no Recognition node, only image.jpg is written")
print(f"Camera {camera.getWidth()}x{camera.getHeight()}, fov {camera.getFov():.3f} rad")

# ---------------------------------------------------------------- File I/O

# <repo>/data/camera, next to data/panda.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "camera"
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("Data dir:", DATA_DIR)

def replace(tmp, path, retries=5):
    """Swap tmp in as path. Retries while the reader has the file open (Windows)."""
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)
    os.replace(tmp, path)            # last attempt: a real failure should be visible -> raise

def write_json(path, data):
    """Writes to a temp file, then swaps it in, so the reader never sees a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    replace(tmp, path)

def save_image(save, path):
    """Same as write_json, for camera.saveImage-like functions.
    The temp name keeps the extension: Webots picks the image format from it."""
    tmp = path.with_name("tmp_" + path.name)
    if save(str(tmp), JPEG_QUALITY) != 0:
        print(f"WARNING: could not save {path.name}")
        return
    replace(tmp, path)

# ---------------------------------------------------------------- Camera frame -> world frame

def axis_angle_to_matrix(x, y, z, angle):
    """3x3 rotation matrix (list of rows) from a Webots rotation field (Rodrigues formula)."""
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:                                        # no rotation, return identity
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return [[t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]

CAM_R = axis_angle_to_matrix(*CAM_ROT)

def camera_position_to_world(p_cam):
    """World-frame point from a camera-frame point (x = view direction, y left, z up)."""
    return [CAM_POS[i] + sum(CAM_R[i][j] * p_cam[j] for j in range(3)) for i in range(3)]

def camera_rotation_to_world(R_cam):
    """World-frame rotation matrix from a camera-frame rotation matrix."""
    return [[sum(CAM_R[i][k] * R_cam[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

# ---------------------------------------------------------------- Recognition

def recognised_objects():
    """Every object the camera recognises right now, as JSON-ready dicts."""
    objects = []
    for obj in camera.getRecognitionObjects():
        p_cam = list(obj.getPosition())                        # object position, camera frame
        p_world = camera_position_to_world(p_cam)              # object position, world frame
        R_obj = axis_angle_to_matrix(*obj.getOrientation())    # object rotation, camera frame
        R_world = camera_rotation_to_world(R_obj)              # object rotation, world frame
        yaw_world = math.atan2(R_world[1][0], R_world[0][0])   # object yaw, world frame
        colours = obj.getColors()                              # flat r, g, b, r, g, b, ...
        objects.append({
            "id": obj.getId(),               # Webots node id, not the DEF name
            "model": obj.getModel(),
            "colours": [round(colours[i], 3) for i in range(3 * obj.getNumberOfColors())],
            "position_camera": [round(v, 4) for v in p_cam],
            "position_world": [round(v, 4) for v in p_world],
            "yaw_world": round(yaw_world, 4),
            "size": [round(v, 4) for v in obj.getSize()],
            "position_on_image": list(obj.getPositionOnImage()),   # bbox centre, pixels
            "size_on_image": list(obj.getSizeOnImage()),           # bbox width, height, pixels
        })
    return objects

def save_frame():
    """Write image, segmentation and recognition list for the current frame, print a summary."""
    save_image(camera.saveImage, DATA_DIR / "image.jpg")
    objects = []
    if camera.hasRecognition():
        objects = recognised_objects()
        if camera.isRecognitionSegmentationEnabled():
            save_image(camera.saveRecognitionSegmentationImage, DATA_DIR / "segmentation.png")
    write_json(DATA_DIR / "recognition.json",
               {"time": round(robot.getTime(), 3),
                "wall_time": time.time(),
                "objects": objects})
    summary = "; ".join(f"{o['colours'][:3]} @ {o['position_world']} @ {o['yaw_world']}" for o in objects) or "nothing"
    print(f"[{robot.getTime():6.2f} s] {len(objects)} object(s): {summary}")

# ---------------------------------------------------------------- Main loop

next_save = SAVE_PERIOD              # the first image exists only after one camera sampling period
while robot.step(timestep) != -1:
    if robot.getTime() >= next_save:
        save_frame()
        next_save = robot.getTime() + SAVE_PERIOD
