from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np

from coord_transform import (
    DEPTH_MIN_M,
    DEPTH_MAX_M,
    compute_grasp_target,
    estimate_xy_from_height,
    euler2mat,
    lookup_depth,
    pixel_offset_to_base_mm,
)
from cv_refine import (
    compute_gripper_yaw,
    depth_detect_object,
    depth_detect_with_color,
    extract_color_ref,
    extract_color_ref_bbox,
)
from pick_models import CameraFrameError, RobotCommandError
from pick_models import GraspPlan, ImageCapture, RefinedDetection, VisualServoResult

if TYPE_CHECKING:
    from pick_models import DetectedObject


logger = logging.getLogger(__name__)

ASPECT_THRESHOLD = 1.3
TALL_OBJECT_THRESHOLD = 80
GRIP_HEIGHT_FRACTION = 0.4
MIN_CLEARANCE_ABOVE = 120


def estimate_object_height(depth_img, px, py, eef_pose_m, euler_eef_to_color):
    """Estimate object height above table from a high-position depth image."""
    height_px, width_px = depth_img.shape
    search_radius = 100
    y_min = max(0, py - search_radius)
    y_max = min(height_px, py + search_radius + 1)
    x_min = max(0, px - search_radius)
    x_max = min(width_px, px + search_radius + 1)
    patch = depth_img[y_min:y_max, x_min:x_max]
    valid = patch[~np.isnan(patch)]
    valid = valid[(valid > DEPTH_MIN_M) & (valid < DEPTH_MAX_M)]

    if len(valid) < 10:
        return 0

    min_depth = float(np.percentile(valid, 5))
    table_depth = float(np.percentile(valid, 80))

    transform_base_eef = euler2mat(eef_pose_m)
    transform_eef_cam = euler2mat(euler_eef_to_color)
    transform_base_cam = transform_base_eef @ transform_eef_cam
    cam_z_mm = float(transform_base_cam[2, 3]) * 1000

    obj_top_z = cam_z_mm - min_depth * 1000
    table_z = cam_z_mm - table_depth * 1000
    height = obj_top_z - table_z
    return max(0, height)


def _detect_refined_object(depth, color, ref_hist, search_radius):
    img_cx = color.shape[1] // 2
    img_cy = color.shape[0] // 2

    if ref_hist is not None:
        result = depth_detect_with_color(
            depth,
            color,
            img_cx,
            img_cy,
            ref_hist,
            search_radius=search_radius,
        )
    else:
        result = depth_detect_object(depth, img_cx, img_cy, search_radius=search_radius)
    return RefinedDetection.from_cv_result(result)


def visual_servo_center(
    robot,
    cam,
    target_x,
    target_y,
    close_z,
    intrinsics,
    euler_eef_to_color,
    ref_hist=None,
    max_iters=5,
    tolerance_px=40,
):
    """Center camera over object using visual servoing with adaptive gain."""
    cur_x, cur_y = target_x, target_y
    detection = None
    search_radius = 600
    prev_offset = None
    images = None

    for iteration in range(max_iters):
        logger.debug("[VS %d] Moving to (%.1f, %.1f, %.0f)", iteration + 1, cur_x, cur_y, close_z)
        robot.move_to(cur_x, cur_y, close_z, yaw=0)
        time.sleep(0.8)

        depth, color = cam.capture_for_detection()
        images = ImageCapture(depth=depth, color=color)
        detection = _detect_refined_object(depth, color, ref_hist, search_radius)
        if detection is None:
            logger.warning("[VS %d] Detection failed", iteration + 1)
            return VisualServoResult(cur_x, cur_y, None, images)

        img_cx = color.shape[1] // 2
        img_cy = color.shape[0] // 2
        dx_px = detection.cx - img_cx
        dy_px = detection.cy - img_cy
        offset = math.sqrt(dx_px**2 + dy_px**2)
        logger.debug(
            "[VS %d] Object at (%d, %d), offset=%.0fpx, %.0fx%.0f, aspect=%.2f",
            iteration + 1,
            detection.cx,
            detection.cy,
            offset,
            detection.width,
            detection.height,
            detection.aspect_ratio,
        )

        obj_size = max(detection.width, detection.height)
        new_search_radius = max(int(obj_size * 2), 250)
        if new_search_radius < search_radius:
            search_radius = new_search_radius

        if offset < tolerance_px:
            logger.info("[VS %d] Centered with %.0fpx residual", iteration + 1, offset)
            return VisualServoResult(cur_x, cur_y, detection, images)

        if offset > 200:
            gain = 0.9
        elif offset > 80:
            gain = 0.8
        else:
            gain = 0.6

        if prev_offset is not None and offset > prev_offset * 1.2:
            logger.info("[VS %d] Overshoot detected (%.0f -> %.0f), stopping", iteration + 1, prev_offset, offset)
            return VisualServoResult(cur_x, cur_y, detection, images)
        prev_offset = offset

        obj_depth = detection.obj_depth or lookup_depth(depth, detection.cx, detection.cy, radius=15)
        if obj_depth is None:
            return VisualServoResult(cur_x, cur_y, detection, images)

        eef_pose_m = robot.get_eef_pose_m()
        corr_x, corr_y = pixel_offset_to_base_mm(
            dx_px,
            dy_px,
            obj_depth,
            eef_pose_m,
            intrinsics,
            euler_eef_to_color,
        )
        cur_x += corr_x * gain
        cur_y += corr_y * gain
        logger.debug(
            "[VS %d] Correction (%.1f, %.1f)mm with gain %.1f -> (%.1f, %.1f)",
            iteration + 1,
            corr_x,
            corr_y,
            gain,
            cur_x,
            cur_y,
        )

    if images is None:
        depth, color = cam.capture_for_detection()
        images = ImageCapture(depth=depth, color=color)
    return VisualServoResult(cur_x, cur_y, detection, images)


def plan_grasp_for_object(detected_object, scan_images, cam, robot, rc, cc, intrinsics, dry_run=False, detector=None, diagnostics=None, scan_id=None):
    """Build a grasp plan for one detected object."""
    try:
        name = detected_object.name
        coarse_px, coarse_py = detected_object.px, detected_object.py

        if not dry_run:
            eef_pose_m = robot.get_eef_pose_m()
        else:
            detect_x, detect_y, detect_z = rc["detect_xyz"]
            eef_pose_m = [detect_x * 0.001, detect_y * 0.001, detect_z * 0.001, -3.14159, 0, 0]

        close_z = rc.get("close_detect_z", 200)
        gripper_z_mm = rc["gripper_z_mm"]
        grasp_depth_offset = rc.get("grasp_depth_offset", 0)
        grasping_min_z = rc["grasping_min_z"]

        height_est = estimate_object_height(
            scan_images.depth,
            coarse_px,
            coarse_py,
            eef_pose_m,
            cc["euler_eef_to_color"],
        )
        if height_est > TALL_OBJECT_THRESHOLD:
            close_z = max(close_z, height_est + MIN_CLEARANCE_ABOVE)
            logger.info("[Stage1] Tall object %.0fmm, close_z raised to %.0f", height_est, close_z)
        elif height_est > 0:
            logger.debug("[Stage1] Object height estimate %.0fmm", height_est)

        if detected_object.robot_xy_mm is not None:
            rough_x, rough_y = detected_object.robot_xy_mm
            logger.info("[Stage1] %s pre-computed XY: (%.1f, %.1f) mm", name, rough_x, rough_y)
        else:
            if detected_object.bbox is not None:
                bbox = detected_object.bbox
                center_px = (bbox.x1 + bbox.x2) // 2
                center_py = (bbox.y1 + bbox.y2) // 2
            else:
                center_px, center_py = coarse_px, coarse_py

            rough_xy = estimate_xy_from_height(
                center_px,
                center_py,
                eef_pose_m,
                intrinsics,
                cc["euler_eef_to_color"],
            )
            if rough_xy is None:
                logger.warning("[Stage1] Failed XY estimate for (%d, %d)", center_px, center_py)
                return None, "xy_estimation_failed"

            rough_x, rough_y = rough_xy
            logger.info("[Stage1] %s rough XY: (%.1f, %.1f) mm", name, rough_x, rough_y)

        if detected_object.bbox is not None:
            bbox = detected_object.bbox
            ref_hist = extract_color_ref_bbox(scan_images.color, bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        else:
            ref_hist = None
        if ref_hist is None:
            ref_hist = extract_color_ref(scan_images.color, coarse_px, coarse_py)

        if dry_run:
            logger.info("[Stage2] Dry run capture at planned close pose")
            depth2, color2 = cam.capture_for_detection()
            images = ImageCapture(depth=depth2, color=color2)
            refined = _detect_refined_object(depth2, color2, None, search_radius=600)
            servo_result = VisualServoResult(rough_x, rough_y, refined, images)
            if refined is None:
                logger.warning("[Stage2] Dry-run detection failed")
                return None, "detection_failed_dry"
        else:
            logger.info("[Stage2] Visual servo centering at z=%.0f", close_z)
            servo_result = visual_servo_center(
                robot,
                cam,
                rough_x,
                rough_y,
                close_z,
                intrinsics,
                cc["euler_eef_to_color"],
                ref_hist=ref_hist,
                max_iters=3,
            )
            if servo_result.detection is None:
                logger.warning("[Stage2] Detection failed")
                return None, "detection_failed"

        refined = servo_result.detection

        if detector is not None and not dry_run:
            verify_enabled = cc.get("verify_identity", False)
            if verify_enabled:
                crop_margin = 30
                crop_x1 = max(0, refined.cx - int(refined.width / 2) - crop_margin)
                crop_y1 = max(0, refined.cy - int(refined.height / 2) - crop_margin)
                crop_x2 = min(servo_result.images.color.shape[1], refined.cx + int(refined.width / 2) + crop_margin)
                crop_y2 = min(servo_result.images.color.shape[0], refined.cy + int(refined.height / 2) + crop_margin)
                crop = servo_result.images.color[crop_y1:crop_y2, crop_x1:crop_x2]
                match, actual = detector.verify_identity(crop, detected_object.name)
                if not match:
                    logger.warning(
                        "[Verify] Identity mismatch: expected '%s', got '%s'",
                        detected_object.name, actual,
                    )
                    return None, "identity_mismatch"

        depth_m = refined.obj_depth or lookup_depth(
            servo_result.images.depth,
            refined.cx,
            refined.cy,
            radius=10,
        )
        if depth_m is None:
            logger.warning("[Stage2] No usable depth")
            return None, "no_depth"

        avg_focal = (intrinsics.fx + intrinsics.fy) / 2
        short_dim_mm = refined.height * depth_m / avg_focal * 1000
        long_dim_mm = refined.width * depth_m / avg_focal * 1000
        logger.info("[Stage2] Physical size: %.0f x %.0f mm", long_dim_mm, short_dim_mm)

        # Save visualization of detected object outline
        try:
            vis = servo_result.images.color.copy()
            rect_center = (refined.cx, refined.cy)
            rect_size = (int(refined.width), int(refined.height))
            rect_angle = refined.angle_deg
            box = cv2.boxPoints(((rect_center), (rect_size), rect_angle))
            box = np.intp(box)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
            label = f"{long_dim_mm:.0f}x{short_dim_mm:.0f}mm"
            cv2.putText(vis, label, (refined.cx - 100, refined.cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if diagnostics is not None and scan_id:
                vis_path = diagnostics.save_image(scan_id, "grasp_detect.jpg", vis)
            else:
                vis_dir = os.path.join("diagnostics", "scan_images")
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_grasp_detect.jpg")
                cv2.imwrite(vis_path, vis)
            if vis_path:
                logger.info("[GRASP VIS] Saved: %s  (%s)", vis_path, label)
        except Exception as exc:
            logger.warning("Failed to save grasp visualization: %s", exc)

        gripper_yaw = 0.0
        if refined.aspect_ratio >= ASPECT_THRESHOLD:
            gripper_yaw = compute_gripper_yaw(refined.angle_deg)
        logger.debug("[Stage2] Planned gripper yaw %.1f", gripper_yaw)

        if not dry_run:
            eef_pose_m = robot.get_eef_pose_m()
        else:
            eef_pose_m = [rough_x * 0.001, rough_y * 0.001, close_z * 0.001, -3.14159, 0, 0]

        grasp_pose = compute_grasp_target(
            refined.cx,
            refined.cy,
            depth_m,
            eef_pose_m,
            intrinsics,
            cc["euler_eef_to_color"],
            gripper_z_mm=gripper_z_mm - grasp_depth_offset,
            grasping_min_z=grasping_min_z,
            yaw_deg=gripper_yaw,
        )
        grasp_x, grasp_y, grasp_z = grasp_pose[0], grasp_pose[1], grasp_pose[2]

        if refined.table_depth and refined.obj_depth:
            obj_height_close = (refined.table_depth - refined.obj_depth) * 1000
            if obj_height_close > TALL_OBJECT_THRESHOLD:
                obj_top_z = grasp_z - (gripper_z_mm - grasp_depth_offset)
                obj_bottom_z = obj_top_z - obj_height_close
                desired_grip_z = obj_bottom_z + obj_height_close * GRIP_HEIGHT_FRACTION
                grasp_z = max(desired_grip_z + gripper_z_mm, grasping_min_z)
                logger.info("[Stage2] Tall object %.0fmm, adjusted grasp z to %.0f", obj_height_close, grasp_z)

        logger.info(
            "[Grasp] Target x=%.1f y=%.1f z=%.1f yaw=%.1f",
            grasp_x,
            grasp_y,
            grasp_z,
            gripper_yaw,
        )
        approach_z = max(close_z, grasp_z + 80)

        return (
            GraspPlan(
                detected_object=detected_object,
                refined_detection=refined,
                rough_x_mm=rough_x,
                rough_y_mm=rough_y,
                close_detect_z_mm=close_z,
                grasp_x_mm=grasp_x,
                grasp_y_mm=grasp_y,
                grasp_z_mm=grasp_z,
                approach_z_mm=approach_z,
                gripper_yaw_deg=gripper_yaw,
                long_dim_mm=long_dim_mm,
                short_dim_mm=short_dim_mm,
                estimated_height_mm=height_est,
            ),
            "planned",
        )
    except (RobotCommandError, CameraFrameError) as exc:
        logger.error("Planning aborted: %s", exc)
        reason = getattr(exc, "reason", "camera_frame_failed")
        return None, reason
