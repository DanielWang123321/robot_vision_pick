import math
import numpy as np

# Depth validity range (meters) for RealSense D435.
# Min ~0.105 m (sensor limit); max 1.2 m (desktop workspace ceiling).
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 1.2


def rpy_to_rot(rpy):
    """RPY angles (radians) to 3x3 rotation matrix."""
    roll, pitch, yaw = rpy
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def euler2mat(euler_6d):
    """[x, y, z, roll, pitch, yaw] (meters, radians) to 4x4 homogeneous matrix."""
    mat = np.eye(4)
    mat[:3, :3] = rpy_to_rot(euler_6d[3:])
    mat[0, 3] = euler_6d[0]
    mat[1, 3] = euler_6d[1]
    mat[2, 3] = euler_6d[2]
    return mat


def mat2euler(mat):
    """4x4 homogeneous matrix to [x, y, z, roll, pitch, yaw]."""
    rot = mat[:3, :3]
    pitch = math.atan2(-rot[2, 0], math.sqrt(rot[2, 1]**2 + rot[2, 2]**2))
    sign = 1 if math.cos(pitch) >= 0 else -1
    roll = math.atan2(rot[2, 1] * sign, rot[2, 2] * sign)
    yaw = math.atan2(rot[1, 0] * sign, rot[0, 0] * sign)
    return [float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]), roll, pitch, yaw]


def convert_pose(pose_6d, T_source_in_target):
    """Transform a 6D pose from source frame to target frame."""
    mat_in_source = euler2mat(pose_6d)
    mat_in_target = T_source_in_target @ mat_in_source
    return mat2euler(mat_in_target)


def pixel_to_camera_frame(px, py, depth_m, fx, fy, cx, cy):
    """Pixel + depth to 3D point in camera frame (meters)."""
    x_cam = (px - cx) / fx * depth_m
    y_cam = (py - cy) / fy * depth_m
    z_cam = depth_m
    return x_cam, y_cam, z_cam


def lookup_depth(depth_image, px, py, radius=5):
    """Robust depth lookup: median of valid values in a patch around (px, py)."""
    h, w = depth_image.shape
    y_min = max(0, py - radius)
    y_max = min(h, py + radius + 1)
    x_min = max(0, px - radius)
    x_max = min(w, px + radius + 1)
    patch = depth_image[y_min:y_max, x_min:x_max]
    valid = patch[~np.isnan(patch)]
    valid = valid[(valid > DEPTH_MIN_M) & (valid < DEPTH_MAX_M)]
    if len(valid) < 3:
        return None
    return float(np.median(valid))


def estimate_xy_from_height(px, py, eef_pose_m, intrinsics, euler_eef_to_color, table_z_m=0.0):
    """Estimate object XY in robot base frame using known table height (no depth sensor needed).

    Assumes the object is on a flat table at table_z_m (default 0 = robot base Z=0).
    Computes the camera-to-table distance geometrically and uses it as estimated depth.

    Returns:
        (x_mm, y_mm) estimated position in robot base frame, or None on failure.
    """
    T_base_eef = euler2mat(eef_pose_m)
    T_eef_cam = euler2mat(euler_eef_to_color)
    T_base_cam = T_base_eef @ T_eef_cam

    # Camera position in base frame
    cam_pos_base = T_base_cam[:3, 3]
    cam_z_base = float(cam_pos_base[2])

    # Camera Z axis direction in base frame (column 2 of rotation)
    cam_z_dir = T_base_cam[:3, 2]

    # For a downward-looking camera, cam_z_dir should point roughly toward -Z in base
    # Estimated depth = distance along camera Z to the table plane
    if abs(cam_z_dir[2]) < 0.1:
        return None  # Camera not pointing down

    estimated_depth = (cam_z_base - table_z_m) / abs(cam_z_dir[2])
    if estimated_depth <= 0:
        return None

    x_cam, y_cam, z_cam = pixel_to_camera_frame(
        px, py, estimated_depth, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    )

    point_cam_6d = [x_cam, y_cam, z_cam, 0, 0, 0]
    point_base_6d = convert_pose(point_cam_6d, T_base_cam)

    x_mm = point_base_6d[0] * 1000
    y_mm = point_base_6d[1] * 1000
    return x_mm, y_mm


def pixel_offset_to_base_mm(dx_px, dy_px, depth_m, eef_pose_m, intrinsics, euler_eef_to_color):
    """Convert pixel offset from image center to base-frame XY correction (mm).

    Uses the full camera-to-base rotation matrix for accurate direction mapping.
    """
    dx_cam = dx_px / intrinsics.fx * depth_m
    dy_cam = dy_px / intrinsics.fy * depth_m

    T_base_eef = euler2mat(eef_pose_m)
    T_eef_cam = euler2mat(euler_eef_to_color)
    T_base_cam = T_base_eef @ T_eef_cam
    R = T_base_cam[:3, :3]

    d_cam = np.array([dx_cam, dy_cam, 0.0])
    d_base = R @ d_cam
    return float(d_base[0]) * 1000, float(d_base[1]) * 1000


def robot_xy_to_pixel(x_mm, y_mm, eef_pose_m, intrinsics, euler_eef_to_color, table_z_m=0.0):
    """Convert robot base XY (mm) on table plane to pixel coordinates.

    Inverse of estimate_xy_from_height(). Assumes object is at table_z_m height.
    Returns (px, py) or None if point is behind camera.
    """
    T_base_eef = euler2mat(eef_pose_m)
    T_eef_cam = euler2mat(euler_eef_to_color)
    T_base_cam = T_base_eef @ T_eef_cam
    T_cam_base = np.linalg.inv(T_base_cam)

    point_base = np.array([x_mm / 1000.0, y_mm / 1000.0, table_z_m, 1.0])
    point_cam = T_cam_base @ point_base

    x_cam, y_cam, z_cam = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    if z_cam <= 0.01:
        return None

    px = int(intrinsics.fx * x_cam / z_cam + intrinsics.ppx)
    py = int(intrinsics.fy * y_cam / z_cam + intrinsics.ppy)
    return px, py


def compute_grasp_target(px, py, depth_m, eef_pose_m, intrinsics, euler_eef_to_color, gripper_z_mm=0, grasping_min_z=0, yaw_deg=0):
    """Full pipeline: pixel + depth -> grasp pose in robot base frame (mm).

    Args:
        px, py: pixel coordinates
        depth_m: depth in meters
        eef_pose_m: [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
        intrinsics: pyrealsense2 intrinsics object (has fx, fy, ppx, ppy)
        euler_eef_to_color: hand-eye calibration [x,y,z,r,p,y] meters+radians
        gripper_z_mm: flange-to-tip offset in mm (0 if TCP set)
        grasping_min_z: minimum z in mm
        yaw_deg: gripper yaw in degrees (0 = default, adjusted for object orientation)

    Returns:
        [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg] or None
    """
    x_cam, y_cam, z_cam = pixel_to_camera_frame(
        px, py, depth_m, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    )

    T_base_eef = euler2mat(eef_pose_m)
    T_eef_cam = euler2mat(euler_eef_to_color)
    T_base_cam = T_base_eef @ T_eef_cam

    point_cam_6d = [x_cam, y_cam, z_cam, 0, 0, 0]
    point_base_6d = convert_pose(point_cam_6d, T_base_cam)

    x_mm = point_base_6d[0] * 1000
    y_mm = point_base_6d[1] * 1000
    z_mm = point_base_6d[2] * 1000 + gripper_z_mm
    z_mm = max(z_mm, grasping_min_z)

    return [x_mm, y_mm, z_mm, 180, 0, yaw_deg]
