from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import cv2
import numpy as np

from camera import RealSenseCamera
from coord_transform import estimate_xy_from_height, lookup_depth, DEPTH_MIN_M, DEPTH_MAX_M
from pick_models import PickResult, RobotCommandError, ScanResult
from llm_detector import LLMDetector
from pick_workflows import execute_grasp_plan, log_pick_result
from pick_planner import plan_grasp_for_object
from pick_selection import TargetTracker, assess_targets
from robot import Robot


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan filters (standalone for testability)
# ---------------------------------------------------------------------------

def filter_by_category(detections, category="fruit"):
    """Keep only detections matching the given category."""
    return [d for d in detections if d.get("category") == category]


def filter_by_physical_size(detections, depth_image, avg_focal, min_dim_mm=20, min_height_mm=10):
    """Remove detections that are too small (area) or too flat (height above table).

    Returns (kept, removed) where removed is a list of (detection, reason) tuples.
    """
    kept = []
    removed = []
    for d in detections:
        px, py = d["px"], d["py"]
        bbox = d.get("bbox")

        # Size check via bbox
        if bbox is not None:
            obj_depth = lookup_depth(depth_image, px, py, radius=10)
            if obj_depth is not None:
                bbox_w_mm = (bbox["x2"] - bbox["x1"]) * obj_depth / avg_focal * 1000
                bbox_h_mm = (bbox["y2"] - bbox["y1"]) * obj_depth / avg_focal * 1000
                if bbox_w_mm < min_dim_mm and bbox_h_mm < min_dim_mm:
                    removed.append((d, f"too small: {bbox_w_mm:.0f}x{bbox_h_mm:.0f}mm"))
                    continue

        # Height check: ring-based table sampling to avoid object contamination
        outer_r = 150
        inner_r = 40
        h_img, w_img = depth_image.shape
        y_min = max(0, py - outer_r)
        y_max = min(h_img, py + outer_r + 1)
        x_min = max(0, px - outer_r)
        x_max = min(w_img, px + outer_r + 1)
        patch = depth_image[y_min:y_max, x_min:x_max]

        # Object top: small center region, 5th percentile
        center_patch = depth_image[max(0, py - 10):min(h_img, py + 11),
                                   max(0, px - 10):min(w_img, px + 11)]
        center_valid = center_patch[~np.isnan(center_patch)]
        center_valid = center_valid[(center_valid > DEPTH_MIN_M) & (center_valid < DEPTH_MAX_M)]

        # Table: ring excluding object center, 80th percentile
        cy_local, cx_local = py - y_min, px - x_min
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        dist_sq = (yy - cy_local) ** 2 + (xx - cx_local) ** 2
        ring_mask = dist_sq >= inner_r ** 2
        ring_vals = patch[ring_mask]
        ring_valid = ring_vals[~np.isnan(ring_vals)]
        ring_valid = ring_valid[(ring_valid > DEPTH_MIN_M) & (ring_valid < DEPTH_MAX_M)]

        if len(center_valid) >= 3 and len(ring_valid) >= 10:
            obj_top = float(np.percentile(center_valid, 5))
            table_d = float(np.percentile(ring_valid, 80))
            if table_d > obj_top:
                height_mm = (table_d - obj_top) * 1000
                if height_mm < min_height_mm:
                    removed.append((d, f"too flat: height={height_mm:.0f}mm"))
                    continue

        kept.append(d)
    return kept, removed


def convert_to_robot_coords(detections, eef_pose_m, intrinsics, euler_eef_to_color, safe_bounds):
    """Convert pixel detections to robot XY and filter by safe bounds.

    Returns (results, skipped) where skipped is a list of (detection, reason) tuples.
    """
    results = []
    skipped = []
    for det in detections:
        px, py = det["px"], det["py"]
        xy = estimate_xy_from_height(px, py, eef_pose_m, intrinsics, euler_eef_to_color)
        if xy is None:
            skipped.append((det, f"XY conversion failed at ({px}, {py})"))
            continue
        robot_x, robot_y = xy
        if not (safe_bounds["x"][0] <= robot_x <= safe_bounds["x"][1] and
                safe_bounds["y"][0] <= robot_y <= safe_bounds["y"][1]):
            skipped.append((det, f"outside bounds at ({robot_x:.0f}, {robot_y:.0f})"))
            continue

        results.append({
            "name": det["name"],
            "canonical_name": det["canonical_name"],
            "category": det["category"],
            "px": px,
            "py": py,
            "bbox": det.get("bbox"),
            "graspable": det["graspable"],
            "grasp_reason": det.get("grasp_reason"),
            "confidence": det["confidence"],
            "robot_xy_mm": (robot_x, robot_y),
        })
    return results, skipped


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _serialize_bbox(bbox):
    if bbox is None:
        return None
    return {
        "x1": bbox.x1,
        "y1": bbox.y1,
        "x2": bbox.x2,
        "y2": bbox.y2,
        "width": bbox.width,
        "height": bbox.height,
        "area": bbox.area,
    }


def _serialize_detected_object(detected_object):
    return {
        "target_id": detected_object.display_id,
        "name": detected_object.name,
        "canonical_name": detected_object.canonical_name,
        "category": detected_object.category,
        "graspable": detected_object.graspable,
        "grasp_reason": detected_object.grasp_reason,
        "confidence": detected_object.confidence,
        "px": detected_object.px,
        "py": detected_object.py,
        "bbox": _serialize_bbox(detected_object.bbox),
    }


def _serialize_assessment(assessment):
    return {
        "target": _serialize_detected_object(assessment.detected_object),
        "total_score": assessment.total_score,
        "confidence_score": assessment.confidence_score,
        "graspability_score": assessment.graspability_score,
        "border_score": assessment.border_score,
        "center_score": assessment.center_score,
        "isolation_score": assessment.isolation_score,
        "depth_score": assessment.depth_score,
        "bbox_score": assessment.bbox_score,
        "reasons": list(assessment.reasons),
    }


def _serialize_plan(plan):
    if plan is None:
        return None
    return {
        "target": _serialize_detected_object(plan.detected_object),
        "rough_xy_mm": [plan.rough_x_mm, plan.rough_y_mm],
        "close_detect_z_mm": plan.close_detect_z_mm,
        "grasp_xyz_mm": [plan.grasp_x_mm, plan.grasp_y_mm, plan.grasp_z_mm],
        "approach_z_mm": plan.approach_z_mm,
        "gripper_yaw_deg": plan.gripper_yaw_deg,
        "long_dim_mm": plan.long_dim_mm,
        "short_dim_mm": plan.short_dim_mm,
        "estimated_height_mm": plan.estimated_height_mm,
        "refined_detection": {
            "cx": plan.refined_detection.cx,
            "cy": plan.refined_detection.cy,
            "width": plan.refined_detection.width,
            "height": plan.refined_detection.height,
            "angle_deg": plan.refined_detection.angle_deg,
            "aspect_ratio": plan.refined_detection.aspect_ratio,
            "area": plan.refined_detection.area,
            "obj_depth": plan.refined_detection.obj_depth,
            "table_depth": plan.refined_detection.table_depth,
            "color_score": plan.refined_detection.color_score,
        },
    }


def _serialize_pick_result(pick_result):
    return {
        "success": pick_result.success,
        "reason": pick_result.reason,
        "verified": pick_result.verified,
        "detected_object": None if pick_result.detected_object is None else _serialize_detected_object(pick_result.detected_object),
        "grasp_plan": _serialize_plan(pick_result.grasp_plan),
    }


class DiagnosticsRecorder:
    """Persist scan, selection, and execution artifacts for offline replay."""

    def __init__(self, diagnostics_cfg=None):
        diagnostics_cfg = diagnostics_cfg or {}
        self.enabled = diagnostics_cfg.get("enabled", True)
        self.output_dir = Path(diagnostics_cfg.get("output_dir", "diagnostics"))
        self.save_color = diagnostics_cfg.get("save_color", True)
        self.save_depth = diagnostics_cfg.get("save_depth", True)
        self.save_llm_raw = diagnostics_cfg.get("save_llm_raw", True)
        self.save_assessments = diagnostics_cfg.get("save_assessments", True)
        self._scan_counter = 0
        self._pick_counter = 0
        self.session_dir: Path | None = None

        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.output_dir / f"session_{timestamp}"
            self._ensure_dir(self.session_dir / "scans")
            self._ensure_dir(self.session_dir / "picks")

    def record_scan(self, scan_result):
        if not self.enabled or self.session_dir is None:
            return scan_result

        self._scan_counter += 1
        scan_id = f"scan_{self._scan_counter:04d}"
        scan_dir = self.session_dir / "scans" / scan_id
        self._ensure_dir(scan_dir)

        if self.save_color:
            cv2.imwrite(str(scan_dir / "color.jpg"), scan_result.images.color)
        if self.save_depth:
            np.save(scan_dir / "depth.npy", scan_result.images.depth)

        metadata = {
            "scan_id": scan_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "objects": [_serialize_detected_object(obj) for obj in scan_result.objects],
            "llm_raw_response": scan_result.llm_raw_response if self.save_llm_raw else None,
        }
        self._write_json(scan_dir / "scan.json", metadata)
        return replace(scan_result, scan_id=scan_id)

    def record_assessments(self, scan_result, assessments):
        if not self.enabled or self.session_dir is None or not scan_result.scan_id or not self.save_assessments:
            return

        scan_dir = self.session_dir / "scans" / scan_result.scan_id
        payload = {
            "scan_id": scan_result.scan_id,
            "assessments": [_serialize_assessment(assessment) for assessment in assessments],
        }
        self._write_json(scan_dir / "assessments.json", payload)

    def record_pick(self, scan_result, pick_result, selected_assessment=None):
        if not self.enabled or self.session_dir is None:
            return None

        self._pick_counter += 1
        pick_id = f"pick_{self._pick_counter:04d}"
        pick_dir = self.session_dir / "picks" / pick_id
        self._ensure_dir(pick_dir)

        payload = {
            "pick_id": pick_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scan_id": scan_result.scan_id,
            "selected_assessment": None if selected_assessment is None else _serialize_assessment(selected_assessment),
            "pick_result": _serialize_pick_result(pick_result),
        }
        self._write_json(pick_dir / "pick.json", payload)
        return pick_id

    def save_image(self, scan_id, filename, image):
        """Save an arbitrary image into the scan directory."""
        if not self.enabled or self.session_dir is None or not scan_id:
            return None
        scan_dir = self.session_dir / "scans" / scan_id
        self._ensure_dir(scan_dir)
        path = scan_dir / filename
        try:
            cv2.imwrite(str(path), image)
        except Exception as exc:
            logger.warning("Failed to save image %s: %s", path, exc)
            return None
        return str(path)

    def save_pick_image(self, pick_id, filename, image):
        """Save an arbitrary image into the pick directory."""
        if not self.enabled or self.session_dir is None or not pick_id:
            return None
        pick_dir = self.session_dir / "picks" / pick_id
        self._ensure_dir(pick_dir)
        path = pick_dir / filename
        try:
            cv2.imwrite(str(path), image)
        except Exception as exc:
            logger.warning("Failed to save image %s: %s", path, exc)
            return None
        return str(path)

    def _write_json(self, path: Path, payload):
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write diagnostics file %s: %s", path, exc)

    def _ensure_dir(self, path: Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create diagnostics directory %s: %s", path, exc)


class VisionPickSystem:
    """Manage camera, LLM detector, and robot lifecycle."""

    def __init__(self, cfg, dry_run=False):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_KEY not found in .env")

        self.cfg = cfg
        self.rc = cfg["robot"]
        self.cc = cfg["camera"]
        self.lc = cfg["llm"]
        self.dc = cfg.get("diagnostics", {})
        self.dry_run = dry_run
        self.target_tracker = TargetTracker()
        self.diagnostics = DiagnosticsRecorder(self.dc)

        logger.info("Starting camera")
        self.cam = RealSenseCamera(
            color_width=self.cc["color_width"],
            color_height=self.cc["color_height"],
            depth_width=self.cc["depth_width"],
            depth_height=self.cc["depth_height"],
            fps=self.cc["fps"],
        )
        self.intrinsics = self.cam.get_intrinsics()
        logger.info("Intrinsics: fx=%.1f, fy=%.1f", self.intrinsics.fx, self.intrinsics.fy)

        self.detector = LLMDetector(
            api_key=api_key,
            api_url=self.lc["openrouter_url"],
            model=self.lc["model"],
        )

        self.robot = None
        if not dry_run:
            logger.info("Connecting to robot")
            self.robot = Robot(
                self.rc["ip"],
                self.rc["safe_bounds"],
                self.rc["gripper_open"],
                self.rc["gripper_close"],
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Release all hardware resources."""
        self.cam.stop()
        if self.robot:
            try:
                self.robot.gripper_open()
                detect_x, detect_y, detect_z = self.rc["detect_xyz"]
                self.robot.move_to(detect_x, detect_y, detect_z)
            except (RobotCommandError, KeyboardInterrupt) as exc:
                logger.warning("Cleanup could not restore detect pose: %s", exc)
            self.robot.disconnect()

    def scan_objects(self):
        """LLM-based scan: detect all fruits in a single LLM call on the overview image."""
        if not self.dry_run:
            logger.info("Moving to detect pose")
            self.robot.move_to_detect(self.rc["detect_xyz"])
            self.robot.gripper_open()

        time.sleep(0.5)
        _ts = lambda msg: print(f"[{datetime.now().strftime('%H:%M:%S')}][SCAN] {msg}")
        _ts("Capturing overview image")
        depth_overview, color_overview = self.cam.capture_for_detection()

        # Phase 1: single LLM call to detect all objects
        _ts("LLM detect_multi call start")
        detections = self.detector.detect_multi(color_overview, max_retries=2)
        _ts(f"LLM detect_multi done, found {len(detections)} object(s)")
        for d in detections:
            _ts(f"  -> {d['name']} [category={d['category']}] conf={d['confidence']:.2f} px=({d['px']},{d['py']})")

        fruits = detections
        if not fruits:
            return self._build_scan_result([], depth_overview, color_overview)

        # Filter out small debris
        avg_focal = (self.intrinsics.fx + self.intrinsics.fy) / 2
        fruits, size_removed = filter_by_physical_size(
            fruits, depth_overview, avg_focal,
            min_dim_mm=self.rc.get("min_object_dim_mm", 20),
            min_height_mm=self.rc.get("min_object_height_mm", 10),
        )
        for d, reason in size_removed:
            _ts(f"  {d['name']} {reason}, skipping")
        _ts(f"After size/height filter: {len(fruits)} fruit(s)")
        if not fruits:
            return self._build_scan_result([], depth_overview, color_overview)

        # Phase 2: convert pixel centers to robot XY
        if not self.dry_run:
            eef_pose_m = self.robot.get_eef_pose_m()
        else:
            dx, dy, dz = self.rc["detect_xyz"]
            eef_pose_m = [dx * 0.001, dy * 0.001, dz * 0.001, -3.14159, 0, 0]

        final_results, coord_skipped = convert_to_robot_coords(
            fruits, eef_pose_m, self.intrinsics,
            self.cc["euler_eef_to_color"], self.rc["safe_bounds"],
        )
        for d, reason in coord_skipped:
            _ts(f"{d['name']} {reason}, skipping")

        _ts(f"{len(final_results)} fruit(s) within bounds")
        return self._build_scan_result(final_results, depth_overview, color_overview)

    def _build_scan_result(self, results, depth, color):
        """Create a tracked ScanResult from raw results."""
        raw_scan_result = ScanResult.from_llm_results(
            results, depth, color,
            llm_raw_response=self.detector.last_raw_response_text,
        )
        tracked_objects = self.target_tracker.assign_ids(raw_scan_result.objects)
        scan_result = ScanResult(
            objects=tracked_objects,
            images=raw_scan_result.images,
            llm_raw_response=raw_scan_result.llm_raw_response,
        )
        return self.diagnostics.record_scan(scan_result)

    def assess_targets(self, scan_result):
        """Score and rank detected objects for picking."""
        assessments = assess_targets(scan_result)
        self.diagnostics.record_assessments(scan_result, assessments)
        return assessments

    def pick_object(self, detected_object, scan_result, selected_assessment=None):
        """Plan and execute a pick for one detected object."""
        plan, reason = plan_grasp_for_object(
            detected_object,
            scan_result.images,
            self.cam,
            self.robot,
            self.rc,
            self.cc,
            self.intrinsics,
            self.dry_run,
            detector=self.detector,
            diagnostics=self.diagnostics,
            scan_id=scan_result.scan_id,
        )
        if plan is None:
            pick_result = PickResult(False, reason, detected_object=detected_object)
        elif self.dry_run:
            logger.info("[Dry Run] Would grasp %s at planned coordinates", detected_object.name)
            pick_result = PickResult(True, "dry_run", detected_object=detected_object, grasp_plan=plan)
        else:
            pick_result = execute_grasp_plan(self.robot, plan, self.rc["release_xyz"])

        # Post-place verification: re-scan to check if the picked object disappeared
        if pick_result.success and not self.dry_run:
            pick_result = self._verify_pick(pick_result, scan_result)

        log_pick_result(pick_result)
        self.diagnostics.record_pick(scan_result, pick_result, selected_assessment=selected_assessment)
        return pick_result

    def _verify_pick(self, pick_result, prev_scan_result):
        """Return to detect pose, re-scan, and check if the picked object disappeared."""
        picked_name = pick_result.detected_object.canonical_name or pick_result.detected_object.name
        prev_names = [obj.canonical_name or obj.name for obj in prev_scan_result.objects]

        try:
            logger.info("[Verify] Moving to detect pose for post-place check")
            self.robot.move_to_detect(self.rc["detect_xyz"])
            time.sleep(0.5)

            _depth, color = self.cam.capture_for_detection()
            after_detections = self.detector.detect_multi(color, max_retries=1)
            after_names = [d.get("canonical_name") or d.get("name", "") for d in after_detections]

            logger.info("[Verify] Before: %s", prev_names)
            logger.info("[Verify] After:  %s", after_names)

            if picked_name in after_names:
                logger.warning("[Verify] %s still visible after place — pick likely failed", picked_name)
                return replace(pick_result, success=False, reason="verify_still_present", verified=False)

            logger.info("[Verify] %s disappeared — pick confirmed", picked_name)
            return replace(pick_result, verified=True)

        except Exception as exc:
            logger.warning("[Verify] Verification failed, assuming success: %s", exc)
            return pick_result
