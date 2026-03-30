import time
import logging
from datetime import datetime
from xarm.wrapper import XArmAPI

from pick_models import RobotCommandError

logger = logging.getLogger(__name__)


class Robot:
    def __init__(self, ip, safe_bounds, gripper_open=800, gripper_close=0):
        self.safe_bounds = safe_bounds
        self.gripper_open_val = gripper_open
        self.gripper_close_val = gripper_close

        self.arm = XArmAPI(ip, report_type='real')
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        self.arm.set_gripper_enable(True)

    def check_error(self):
        if self.arm.error_code != 0:
            logger.warning("Error code: %d, cleaning...", self.arm.error_code)
            self.arm.clean_error()
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.5)

    def is_within_bounds(self, x, y, z):
        bx, by, bz = self.safe_bounds['x'], self.safe_bounds['y'], self.safe_bounds['z']
        return bx[0] <= x <= bx[1] and by[0] <= y <= by[1] and bz[0] <= z <= bz[1]

    def move_to(self, x, y, z, roll=180, pitch=0, yaw=0, speed=200):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][ROBOT CMD] move_to  x={x:.1f}  y={y:.1f}  z={z:.1f}  roll={roll}  pitch={pitch}  yaw={yaw}  speed={speed}")
        if not self.is_within_bounds(x, y, z):
            message = f"Target ({x:.1f}, {y:.1f}, {z:.1f}) out of safety bounds"
            logger.warning(message)
            raise RobotCommandError("out_of_bounds", message)
        self.check_error()
        code = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                     speed=speed, wait=True)
        if code != 0:
            message = f"move_to({x:.1f}, {y:.1f}, {z:.1f}) failed with code={code}"
            logger.error(message)
            self.check_error()
            raise RobotCommandError("move_failed", message)
        return True

    def move_to_detect(self, detect_xyz):
        return self.move_to(detect_xyz[0], detect_xyz[1], detect_xyz[2])

    def move_to_release(self, release_xyz):
        return self.move_to(release_xyz[0], release_xyz[1], release_xyz[2])

    def gripper_open(self):
        code = self.arm.set_gripper_position(self.gripper_open_val, wait=True)
        if code != 0:
            raise RobotCommandError("gripper_open_failed", f"gripper_open failed with code={code}")
        return True

    def gripper_close(self):
        """Close gripper. Returns True if object grasped, False if empty grasp."""
        code = self.arm.set_gripper_position(self.gripper_close_val, wait=True)
        if code != 0:
            raise RobotCommandError("gripper_close_failed", f"gripper_close failed with code={code}")
        time.sleep(0.3)
        code, pos = self.arm.get_gripper_position()
        if code != 0 or pos is None:
            raise RobotCommandError(
                "gripper_position_read_failed",
                f"get_gripper_position failed with code={code}",
            )
        if pos <= self.gripper_close_val + 15:
            logger.warning("Empty grasp detected: gripper pos=%.0f (fully closed)", pos)
            return False
        logger.debug("Gripper pos=%.0f, object grasped", pos)
        return True

    def get_eef_pose_m(self):
        """Get current EEF pose as [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]."""
        code, pos = self.arm.get_position(is_radian=True)
        if code != 0 or pos is None:
            raise RobotCommandError("pose_read_failed", f"get_position failed with code={code}")
        return [pos[0] * 0.001, pos[1] * 0.001, pos[2] * 0.001, pos[3], pos[4], pos[5]]

    def disconnect(self):
        self.arm.disconnect()


if __name__ == '__main__':
    safe_bounds = {'x': [200, 600], 'y': [-400, 400], 'z': [0, 500]}
    robot = Robot("192.168.1.60", safe_bounds)
    print("Moving to detect position...")
    robot.move_to(300, 0, 400)
    print("Opening gripper...")
    robot.gripper_open()
    time.sleep(1)
    print("Closing gripper...")
    robot.gripper_close()
    print("EEF pose (m):", robot.get_eef_pose_m())
    robot.disconnect()
