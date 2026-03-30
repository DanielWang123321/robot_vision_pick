import unittest

from pick_models import RobotCommandError
from pick_workflows import execute_grasp_plan
from pick_models import DetectedObject, GraspPlan, RefinedDetection


def make_plan():
    detected_object = DetectedObject(name="mangosteen", px=100, py=200, confidence=0.95)
    refined = RefinedDetection(
        cx=110,
        cy=210,
        width=60.0,
        height=50.0,
        angle_deg=0.0,
        aspect_ratio=1.2,
        area=1200.0,
        obj_depth=0.28,
        table_depth=0.33,
    )
    return GraspPlan(
        detected_object=detected_object,
        refined_detection=refined,
        rough_x_mm=310.0,
        rough_y_mm=-20.0,
        close_detect_z_mm=200.0,
        grasp_x_mm=320.0,
        grasp_y_mm=-15.0,
        grasp_z_mm=40.0,
        approach_z_mm=200.0,
        gripper_yaw_deg=0.0,
        long_dim_mm=45.0,
        short_dim_mm=35.0,
        estimated_height_mm=20.0,
    )


class FakeRobot:
    def __init__(self, move_fail_reason=None, grasped=True):
        self.move_fail_reason = move_fail_reason
        self.grasped = grasped
        self.calls = []

    def is_within_bounds(self, x, y, z):
        self.calls.append(("is_within_bounds", x, y, z))
        return True

    def move_to(self, x, y, z, yaw=0, speed=200, roll=180, pitch=0):
        self.calls.append(("move_to", x, y, z, yaw, speed))
        if self.move_fail_reason is not None:
            reason = self.move_fail_reason
            self.move_fail_reason = None
            raise RobotCommandError(reason, f"move failed: {reason}")
        return True

    def gripper_close(self):
        self.calls.append(("gripper_close",))
        return self.grasped

    def gripper_open(self):
        self.calls.append(("gripper_open",))
        return True


class TestExecuteGraspPlan(unittest.TestCase):
    def test_successful_execution(self):
        robot = FakeRobot()
        result = execute_grasp_plan(robot, make_plan(), release_xyz=[300, 350, 300])
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "success")
        self.assertEqual(result.detected_object.name, "mangosteen")

    def test_robot_failure_returns_reason(self):
        robot = FakeRobot(move_fail_reason="move_failed")
        result = execute_grasp_plan(robot, make_plan(), release_xyz=[300, 350, 300])
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "move_failed")

    def test_empty_grasp_returns_empty_grasp(self):
        robot = FakeRobot(grasped=False)
        result = execute_grasp_plan(robot, make_plan(), release_xyz=[300, 350, 300])
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "empty_grasp")


if __name__ == "__main__":
    unittest.main()
