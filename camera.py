import math
import numpy as np
import pyrealsense2 as rs

from pick_models import CameraFrameError


class RealSenseCamera:
    def __init__(self, color_width=1920, color_height=1080, depth_width=1280, depth_height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = None

    def get_intrinsics(self):
        if self.intrinsics is None:
            frames = self.align.process(self.pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise CameraFrameError("Failed to read color frame for intrinsics")
            self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        return self.intrinsics

    def get_images(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            raise CameraFrameError("Failed to read color frame")
        if not depth_frame:
            raise CameraFrameError("Failed to read depth frame")
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float64) * 0.001
        depth_image[depth_image == 0] = math.nan
        return color_image, depth_image

    def capture_for_detection(self):
        """Capture aligned color and depth images for detection."""
        color_image, depth_image = self.get_images()
        return depth_image, color_image

    def stop(self):
        self.pipeline.stop()


if __name__ == '__main__':
    import cv2
    cam = RealSenseCamera()
    intrin = cam.get_intrinsics()
    print(f"Color resolution: {intrin.width}x{intrin.height}")
    print(f"fx={intrin.fx:.2f}, fy={intrin.fy:.2f}, cx={intrin.ppx:.2f}, cy={intrin.ppy:.2f}")
    fov_h = 2 * math.degrees(math.atan(intrin.width / (2 * intrin.fx)))
    fov_v = 2 * math.degrees(math.atan(intrin.height / (2 * intrin.fy)))
    print(f"FOV: horizontal={fov_h:.1f}°, vertical={fov_v:.1f}°")
    while True:
        color, depth = cam.get_images()
        display = cv2.resize(color, (960, 540))
        cv2.imshow('Color (1920x1080 -> 960x540)', display)
        depth_vis = cv2.convertScaleAbs(np.nan_to_num(depth) * 1000, alpha=0.03)
        depth_vis = cv2.resize(depth_vis, (960, 540))
        cv2.imshow('Depth (aligned to color)', depth_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            break
