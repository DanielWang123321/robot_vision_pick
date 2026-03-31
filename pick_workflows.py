from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime

from pick_models import CameraFrameError, PickResult, RobotCommandError


logger = logging.getLogger(__name__)

GRASP_LOG_FILE = "grasp_log.csv"
GRASP_LOG_FIELDS = ["timestamp", "object_name", "confidence", "grasp_x", "grasp_y", "grasp_z", "success", "reason", "verified"]


def log_pick_result(pick_result, log_file=GRASP_LOG_FILE):
    """Append one grasp record to CSV log."""
    plan = pick_result.grasp_plan
    file_exists = os.path.exists(log_file)
    try:
        with open(log_file, "a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=GRASP_LOG_FIELDS)
            if not file_exists:
                writer.writeheader()

            detected_object = pick_result.detected_object
            if detected_object is None and plan is not None:
                detected_object = plan.detected_object
            verified_str = "" if pick_result.verified is None else str(pick_result.verified)
            writer.writerow(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object_name": "" if detected_object is None else detected_object.name,
                    "confidence": "" if detected_object is None else f"{detected_object.confidence:.2f}",
                    "grasp_x": "" if plan is None else f"{plan.grasp_x_mm:.1f}",
                    "grasp_y": "" if plan is None else f"{plan.grasp_y_mm:.1f}",
                    "grasp_z": "" if plan is None else f"{plan.grasp_z_mm:.1f}",
                    "success": pick_result.success,
                    "reason": pick_result.reason,
                    "verified": verified_str,
                }
            )
    except OSError as exc:
        logger.warning("Failed to write grasp log: %s", exc)


def execute_grasp_plan(robot, plan, release_xyz):
    """Execute a grasp plan and return a structured result."""
    try:
        if not robot.is_within_bounds(plan.grasp_x_mm, plan.grasp_y_mm, plan.grasp_z_mm):
            logger.warning("Target out of bounds, skipping %s", plan.detected_object.name)
            return PickResult(False, "out_of_bounds", plan.detected_object, plan)

        release_x, release_y, release_z = release_xyz
        logger.debug("[Grasp] Approach at z=%.0f yaw=%.1f", plan.approach_z_mm, plan.gripper_yaw_deg)
        robot.move_to(plan.grasp_x_mm, plan.grasp_y_mm, plan.approach_z_mm, yaw=plan.gripper_yaw_deg)

        logger.debug("[Grasp] Descend to z=%.1f", plan.grasp_z_mm)
        robot.move_to(plan.grasp_x_mm, plan.grasp_y_mm, plan.grasp_z_mm, yaw=plan.gripper_yaw_deg, speed=80)

        logger.debug("[Grasp] Closing gripper")
        grasped = robot.gripper_close()
        time.sleep(0.5)
        if not grasped:
            logger.warning("[Grasp] Empty grasp for %s", plan.detected_object.name)
            try:
                robot.gripper_open()
                robot.move_to(plan.grasp_x_mm, plan.grasp_y_mm, plan.approach_z_mm, yaw=plan.gripper_yaw_deg)
            except RobotCommandError as exc:
                logger.error("Recovery after empty grasp failed: %s", exc)
                return PickResult(False, f"empty_grasp_recovery_{exc.reason}", plan.detected_object, plan)
            return PickResult(False, "empty_grasp", plan.detected_object, plan)

        logger.debug("[Grasp] Lift to z=%.0f", release_z)
        robot.move_to(plan.grasp_x_mm, plan.grasp_y_mm, release_z, yaw=plan.gripper_yaw_deg)

        logger.debug("[Grasp] Move to release pose")
        robot.move_to(release_x, release_y, release_z, yaw=plan.gripper_yaw_deg)

        logger.debug("[Grasp] Release")
        robot.gripper_open()
        time.sleep(0.5)
        logger.info("[Done] %s picked and placed", plan.detected_object.name)
        return PickResult(True, "success", plan.detected_object, plan)
    except RobotCommandError as exc:
        logger.error("Execution aborted: %s", exc)
        return PickResult(False, exc.reason, plan.detected_object, plan)


def run_interactive(system):
    """Interactive mode: detect objects and let the user choose one to pick."""
    total_picked = 0

    try:
        while True:
            scan_result = system.scan_objects()
            if not scan_result.objects:
                print("No objects detected. Table is clear.")
                break

            assessments = system.assess_targets(scan_result)
            print(f"\n{'=' * 40}")
            print(f"  Detected {len(assessments)} object(s):")
            print(f"{'=' * 40}")
            for index, assessment in enumerate(assessments, start=1):
                detected_object = assessment.detected_object
                bbox = detected_object.bbox
                if bbox is None:
                    bbox_text = "bbox=n/a"
                else:
                    bbox_text = f"bbox=({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2})"
                print(
                    f"  {index}. [{detected_object.display_id}] {detected_object.name}  "
                    f"(category={detected_object.category}, score={assessment.total_score:.1f}, "
                    f"confidence={detected_object.confidence:.2f}, graspable={'yes' if detected_object.graspable else 'no'}, "
                    f"pixel=({detected_object.px},{detected_object.py}), {bbox_text})"
                )
                if detected_object.grasp_reason:
                    print(f"     hint: {detected_object.grasp_reason}")
            print("  0. Exit")
            print(f"{'=' * 40}")

            selected_assessment = None
            assessment_by_id = {assessment.detected_object.display_id.lower(): assessment for assessment in assessments}
            while selected_assessment is None:
                try:
                    choice = input("Choose object number to pick: ").strip()
                    if choice == "0" or choice.lower() in ("q", "quit", "exit"):
                        logger.info("User exited. Total picked: %d", total_picked)
                        return

                    choice_key = choice.lower()
                    if choice_key in assessment_by_id:
                        selected_assessment = assessment_by_id[choice_key]
                        continue

                    parsed = int(choice) - 1
                    if 0 <= parsed < len(assessments):
                        selected_assessment = assessments[parsed]
                    else:
                        print(f"Invalid selection. Enter 1-{len(assessments)}, a target ID, or 0 to exit.")
                except ValueError:
                    print(f"Invalid input. Enter 1-{len(assessments)}, a target ID, or 0 to exit.")

            detected_object = selected_assessment.detected_object
            print(f"\nPicking: {detected_object.name}...")
            logger.info(
                "--- Picking: [%s] %s [%s] (score=%.1f, graspable=%s) ---",
                detected_object.display_id,
                detected_object.name,
                detected_object.category,
                selected_assessment.total_score,
                detected_object.graspable,
            )
            pick_result = system.pick_object(detected_object, scan_result, selected_assessment=selected_assessment)
            if pick_result.success:
                total_picked += 1
                print(f"Picked successfully. Total: {total_picked}")
            elif pick_result.reason == "identity_mismatch":
                print(f"Could not confirm object is {detected_object.name}. May have locked onto a neighbor. Try again.")
            else:
                print(f"Failed to pick {detected_object.name}: {pick_result.reason}")

    except (RobotCommandError, CameraFrameError) as exc:
        print(f"Workflow aborted: {exc}")
        logger.error("Workflow aborted: %s", exc)
    except KeyboardInterrupt:
        print("\nStopping safely.")
    finally:
        print(f"\nSession total: {total_picked}")


def run_pick_all(system):
    """Detect and pick all objects on the table."""
    total_picked = 0
    max_scans = 1 if system.dry_run else 10

    try:
        for scan_round in range(1, max_scans + 1):
            logger.info("[Scan %d] Starting scan", scan_round)
            scan_result = system.scan_objects()

            if not scan_result.objects:
                logger.info("[Scan %d] No objects detected. Table is clear", scan_round)
                break

            assessments = system.assess_targets(scan_result)
            best_assessment = assessments[0]
            detected_object = best_assessment.detected_object
            logger.info(
                "[Scan %d] Found %d object(s), choosing [%s] %s [%s] with score %.1f",
                scan_round,
                len(assessments),
                detected_object.display_id,
                detected_object.name,
                detected_object.category,
                best_assessment.total_score,
            )
            logger.debug("[Scan %d] Selection reasons: %s", scan_round, ", ".join(best_assessment.reasons))

            pick_result = system.pick_object(detected_object, scan_result, selected_assessment=best_assessment)
            if pick_result.success:
                total_picked += 1
                logger.info("Total picked: %d", total_picked)
            else:
                logger.warning("Failed to pick %s: %s", detected_object.name, pick_result.reason)

        logger.info("Finished. Total objects picked: %d", total_picked)
        return total_picked > 0

    except (RobotCommandError, CameraFrameError) as exc:
        logger.error("Workflow aborted: %s", exc)
        return total_picked > 0
    except KeyboardInterrupt:
        logger.info("Stopping safely")
        return total_picked > 0


def run_loop(system):
    """Continuously run pick-and-place scans."""
    while True:
        run_pick_all(system)
        time.sleep(1)
