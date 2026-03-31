import unittest
from unittest.mock import patch

import numpy as np

from pick_models import RobotCommandError
from pick_models import DetectedObject, ImageCapture, RefinedDetection, VisualServoResult
from pick_planner import plan_grasp_for_object


class FakeIntrinsics:
    fx = 1000.0
    fy = 1000.0
    ppx = 960.0
    ppy = 540.0


class FakeRobot:
    def __init__(self, pose=None, error=None):
        self.pose = pose or [0.3, 0.0, 0.4, -3.14159, 0.0, 0.0]
        self.error = error

    def get_eef_pose_m(self):
        if self.error is not None:
            raise self.error
        return self.pose


class TestPlanGraspForObject(unittest.TestCase):
    def setUp(self):
        self.detected_object = DetectedObject(name="mangosteen", px=100, py=150, confidence=0.9)
        self.scan_images = ImageCapture(
            depth=np.full((20, 20), 0.4, dtype=np.float64),
            color=np.zeros((20, 20, 3), dtype=np.uint8),
        )
        self.cam = object()
        self.robot = FakeRobot()
        self.rc = {
            "detect_xyz": [300, 0, 400],
            "close_detect_z": 200,
            "gripper_z_mm": 172,
            "grasp_depth_offset": 20,
            "grasping_min_z": 0,
        }
        self.cc = {
            "euler_eef_to_color": [0.067, -0.031, 0.022, -0.004, -0.008, 1.59],
        }

    @patch("pick_planner.estimate_object_height", return_value=0)
    @patch("pick_planner.estimate_xy_from_height", return_value=None)
    def test_xy_estimation_failure(self, *_patches):
        plan, reason = plan_grasp_for_object(
            self.detected_object,
            self.scan_images,
            self.cam,
            self.robot,
            self.rc,
            self.cc,
            FakeIntrinsics(),
            dry_run=False,
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "xy_estimation_failed")

    def test_robot_pose_failure_is_propagated(self):
        robot = FakeRobot(error=RobotCommandError("pose_read_failed", "cannot read pose"))
        plan, reason = plan_grasp_for_object(
            self.detected_object,
            self.scan_images,
            self.cam,
            robot,
            self.rc,
            self.cc,
            FakeIntrinsics(),
            dry_run=False,
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "pose_read_failed")

    @patch("pick_planner.compute_grasp_target", return_value=[320.0, -15.0, 38.0, 180, 0, -45.0])
    @patch("pick_planner.compute_gripper_yaw", return_value=-45.0)
    @patch("pick_planner.visual_servo_center")
    @patch("pick_planner.extract_color_ref", return_value="hist")
    @patch("pick_planner.estimate_xy_from_height", return_value=(300.0, -10.0))
    @patch("pick_planner.estimate_object_height", return_value=0.0)
    def test_builds_grasp_plan(self, _height_patch, _xy_patch, _hist_patch, servo_patch, _yaw_patch, _grasp_patch):
        refined = RefinedDetection(
            cx=960,
            cy=540,
            width=60.0,
            height=30.0,
            angle_deg=15.0,
            aspect_ratio=2.0,
            area=1500.0,
            obj_depth=0.5,
            table_depth=0.55,
        )
        servo_patch.return_value = VisualServoResult(
            target_x_mm=300.0,
            target_y_mm=-10.0,
            detection=refined,
            images=self.scan_images,
        )

        plan, reason = plan_grasp_for_object(
            self.detected_object,
            self.scan_images,
            self.cam,
            self.robot,
            self.rc,
            self.cc,
            FakeIntrinsics(),
            dry_run=False,
        )

        self.assertEqual(reason, "planned")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.detected_object.name, "mangosteen")
        self.assertEqual(plan.grasp_x_mm, 320.0)
        self.assertEqual(plan.grasp_y_mm, -15.0)
        self.assertEqual(plan.grasp_z_mm, 38.0)
        self.assertEqual(plan.gripper_yaw_deg, -45.0)
        self.assertAlmostEqual(plan.long_dim_mm, 30.0)
        self.assertAlmostEqual(plan.short_dim_mm, 15.0)
        self.assertEqual(plan.approach_z_mm, 200.0)


if __name__ == "__main__":
    unittest.main()
