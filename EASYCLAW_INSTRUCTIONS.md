# Easyclaw Vision Pick System Instructions

## Task

You are controlling a **UFactory xArm6** robot arm with a **parallel jaw gripper** and a **RealSense D435** depth camera mounted on the end-effector. Your job is to **detect objects on a table using your own vision, plan grasps, and pick-and-place them to a release position**.

---

## Hardware Setup

- **Robot**: UFactory xArm6, IP `192.168.1.60`
- **Gripper**: Parallel jaw, max opening 84mm, mounted on flange
- **Camera**: Intel RealSense D435, mounted on end-effector (eye-in-hand), looking downward
- **Workspace**: Table surface at approximately Z=0 in robot base frame

### Coordinate System

- Robot base frame: X forward, Y left, Z up (mm)
- Default end-effector orientation: roll=180, pitch=0, yaw=0 (gripper pointing straight down)
- Camera is rigidly attached to the end-effector with a known transform

---

## Available Python APIs

All modules are importable from the project directory `c:\Users\72863\Desktop\vision_pick\`.

### 1. Robot Control (`robot.py`)

```python
from robot import Robot

robot = Robot(
    ip="192.168.1.60",
    safe_bounds={"x": [200, 600], "y": [-500, 500], "z": [0, 500]},
    gripper_open=800,
    gripper_close=0
)

# Movement (units: mm for XYZ, degrees for RPY)
robot.move_to(x, y, z, roll=180, pitch=0, yaw=0, speed=200)

# Gripper
robot.gripper_open()
robot.gripper_close()  # Returns True if object grasped, False if empty

# Read current pose: [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
pose = robot.get_eef_pose_m()

# Safety check
robot.is_within_bounds(x, y, z)  # Returns bool

# Cleanup
robot.disconnect()
```

**Safety bounds** (mm): X [200, 600], Y [-500, 500], Z [0, 500]. All moves are checked.

### 2. Camera (`camera.py`)

```python
from camera import RealSenseCamera

cam = RealSenseCamera(
    color_width=1920, color_height=1080,
    depth_width=1280, depth_height=720, fps=30
)

# Get aligned color + depth images
color_image, depth_image = cam.get_images()
# color_image: (1080, 1920, 3) uint8 BGR
# depth_image: (1080, 1920) float64 meters, NaN = invalid

# Or use this (returns depth, color in reversed order):
depth_image, color_image = cam.capture_for_detection()

# Camera intrinsics (after alignment to color frame)
intrinsics = cam.get_intrinsics()
# intrinsics.fx, intrinsics.fy  -- focal lengths (pixels)
# intrinsics.ppx, intrinsics.ppy -- principal point (pixels)

cam.stop()
```

**Depth format**: float64 meters. Valid range: 0.10m to 1.2m. Zero/invalid pixels are NaN.

### 3. Coordinate Transforms (`coord_transform.py`)

```python
from coord_transform import (
    lookup_depth,           # Robust depth at pixel: median in patch
    pixel_to_camera_frame,  # Pixel + depth -> 3D point in camera frame
    estimate_xy_from_height,# Pixel -> robot XY assuming table at Z=0
    pixel_offset_to_base_mm,# Pixel offset -> robot XY correction
    compute_grasp_target,   # Pixel + depth -> full grasp pose [x,y,z,r,p,y]
    euler2mat,              # 6D pose -> 4x4 homogeneous matrix
    DEPTH_MIN_M, DEPTH_MAX_M
)
```

**Key function signatures**:

```python
# Get depth at a pixel (returns float meters or None)
depth_m = lookup_depth(depth_image, px, py, radius=5)

# Pixel -> robot base XY (mm), assuming object on table at Z=0
x_mm, y_mm = estimate_xy_from_height(
    px, py, eef_pose_m, intrinsics,
    euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59],
    table_z_m=0.0
)

# Pixel + depth -> full grasp pose in robot base frame
grasp_pose = compute_grasp_target(
    px, py, depth_m, eef_pose_m, intrinsics,
    euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59],
    gripper_z_mm=152,       # gripper_z_mm(172) - grasp_depth_offset(20)
    grasping_min_z=0,
    yaw_deg=0
)
# Returns: [x_mm, y_mm, z_mm, 180, 0, yaw_deg]
```

### 4. Depth-based Object Segmentation (`cv_refine.py`)

```python
from cv_refine import depth_detect_object, compute_gripper_yaw

# Detect object closest to image center via depth segmentation
result = depth_detect_object(depth_image, color_image, search_radius=600)
# Returns RefinedDetection or None:
#   .cx, .cy         -- center pixel
#   .width, .height   -- bounding rect size (pixels)
#   .angle_deg        -- orientation angle
#   .aspect_ratio     -- length/width ratio
#   .obj_depth        -- object surface depth (meters)
#   .table_depth      -- table surface depth (meters)
#   .ref_hist         -- HSV color histogram for identity matching

# Compute gripper yaw for elongated objects
yaw = compute_gripper_yaw(angle_deg)  # Only if aspect_ratio >= 1.3
```

---

## Hand-Eye Calibration

The transform from end-effector frame to camera color frame is fixed:

```python
euler_eef_to_color = [0.067, -0.031, 0.022, -0.004, -0.008, 1.59]
# [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
```

This is required by all coordinate transform functions.

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `detect_xyz` | [300, -200, 400] | Overview scan position (mm) |
| `close_detect_z` | 300 | Visual servo height (mm) |
| `release_xyz` | [300, 350, 300] | Place/release position (mm) |
| `gripper_z_mm` | 172 | Flange-to-gripper-tip (mm) |
| `grasp_depth_offset` | 20 | TCP offset for grasp (mm) |
| `gripper_max_opening` | 84 | Max graspable width (mm) |
| `grasping_min_z` | 0 | Minimum grasp Z (mm) |

---

## Recommended Pick Pipeline

### Step 1: Move to overview position and capture

```python
robot.move_to(300, -200, 400)  # detect_xyz
robot.gripper_open()
import time; time.sleep(0.5)
depth_img, color_img = cam.capture_for_detection()
eef_pose = robot.get_eef_pose_m()
```

### Step 2: Detect objects (use your own vision)

Analyze `color_img` (1920x1080 BGR) to find objects. For each object, determine:
- Pixel location (px, py) of the best grasp point
- Bounding box if possible
- Object name / category
- Whether it's graspable given the 84mm gripper opening

### Step 3: Estimate rough robot XY for target object

```python
rough_x, rough_y = estimate_xy_from_height(
    px, py, eef_pose, cam.get_intrinsics(),
    euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59]
)
```

### Step 4: Visual servo to refine position

Move closer and iteratively center the object under the camera:

```python
close_z = 300  # mm
for i in range(5):
    robot.move_to(rough_x, rough_y, close_z)
    time.sleep(0.8)
    depth_close, color_close = cam.capture_for_detection()
    eef_pose = robot.get_eef_pose_m()

    # Detect object at close range via depth segmentation
    refined = depth_detect_object(depth_close, color_close, search_radius=600)
    if refined is None:
        break

    # Compute pixel offset from image center
    img_cx, img_cy = 960, 540  # center of 1920x1080
    dx_px = refined.cx - img_cx
    dy_px = refined.cy - img_cy

    if abs(dx_px) < 40 and abs(dy_px) < 40:
        break  # Centered enough

    # Convert pixel offset to robot XY correction
    depth_m = lookup_depth(depth_close, refined.cx, refined.cy, radius=10)
    dx_mm, dy_mm = pixel_offset_to_base_mm(
        dx_px, dy_px, depth_m, eef_pose, cam.get_intrinsics(),
        euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59]
    )

    gain = 0.7
    rough_x += dx_mm * gain
    rough_y += dy_mm * gain
```

### Step 5: Compute grasp pose

```python
depth_m = lookup_depth(depth_close, refined.cx, refined.cy, radius=10)
intrinsics = cam.get_intrinsics()

# Compute gripper yaw for elongated objects
gripper_yaw = 0
if refined.aspect_ratio >= 1.3:
    gripper_yaw = compute_gripper_yaw(refined.angle_deg)

# Full grasp pose
grasp_pose = compute_grasp_target(
    refined.cx, refined.cy, depth_m, eef_pose, intrinsics,
    euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59],
    gripper_z_mm=152,  # 172 - 20 (grasp_depth_offset)
    grasping_min_z=0,
    yaw_deg=gripper_yaw
)
grasp_x, grasp_y, grasp_z = grasp_pose[0], grasp_pose[1], grasp_pose[2]
```

### Step 6: Execute pick

```python
approach_z = max(close_z, grasp_z + 80)

# Approach
robot.move_to(grasp_x, grasp_y, approach_z, yaw=gripper_yaw)

# Descend slowly
robot.move_to(grasp_x, grasp_y, grasp_z, yaw=gripper_yaw, speed=80)

# Grasp
grasped = robot.gripper_close()
time.sleep(0.5)

if not grasped:
    robot.gripper_open()
    robot.move_to(grasp_x, grasp_y, approach_z, yaw=gripper_yaw)
    print("Empty grasp - object missed")
else:
    # Lift and release
    robot.move_to(grasp_x, grasp_y, 300, yaw=gripper_yaw)
    robot.move_to(300, 350, 300, yaw=gripper_yaw)  # release_xyz
    robot.gripper_open()
    time.sleep(0.5)
    print("Pick successful!")
```

---

## Object Height Estimation

To check if an object is real (not a stain or flat marking), estimate its height above the table:

```python
from pick_planner import estimate_object_height

height_mm = estimate_object_height(
    depth_img, px, py, eef_pose,
    euler_eef_to_color=[0.067, -0.031, 0.022, -0.004, -0.008, 1.59]
)
# Uses 5th percentile (object top) vs 80th percentile (table) of depth patch
# For tall objects (>80mm): grip at 40% from bottom, raise approach height
```

---

## Safety Rules

1. **Never move outside safe bounds**: X [200,600], Y [-500,500], Z [0,500] mm
2. **Always open gripper before scanning** to avoid occluding the camera
3. **Use slow speed (80 mm/s) for final grasp descent** to avoid collision
4. **Check `is_within_bounds()` before any move** if computing positions dynamically
5. **Always `robot.disconnect()` and `cam.stop()` on cleanup**
6. **Objects wider than 84mm cannot be grasped** by the parallel jaw gripper

---

## Quick Start

```python
import time
from robot import Robot
from camera import RealSenseCamera
from coord_transform import (
    lookup_depth, estimate_xy_from_height,
    compute_grasp_target, pixel_offset_to_base_mm
)
from cv_refine import depth_detect_object, compute_gripper_yaw

# Initialize hardware
robot = Robot("192.168.1.60", {"x":[200,600],"y":[-500,500],"z":[0,500]}, 800, 0)
cam = RealSenseCamera(1920, 1080, 1280, 720, 30)
intrinsics = cam.get_intrinsics()
EULER_EEF_TO_COLOR = [0.067, -0.031, 0.022, -0.004, -0.008, 1.59]

try:
    # Scan
    robot.move_to(300, -200, 400)
    robot.gripper_open()
    time.sleep(0.5)
    depth, color = cam.capture_for_detection()
    pose = robot.get_eef_pose_m()

    # >>> YOUR VISION HERE <<<
    # Analyze 'color' (1920x1080 BGR numpy array) to find objects
    # Return target pixel (px, py)

    # Estimate and pick (see pipeline above)
    ...

finally:
    cam.stop()
    robot.gripper_open()
    robot.move_to(300, -200, 400)
    robot.disconnect()
```
