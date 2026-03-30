"""Unit tests for coord_transform module."""
import math
import unittest
import numpy as np
from coord_transform import (
    rpy_to_rot, euler2mat, mat2euler, convert_pose,
    pixel_to_camera_frame, lookup_depth, pixel_offset_to_base_mm,
)


class TestRpyToRot(unittest.TestCase):
    def test_identity(self):
        R = rpy_to_rot([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonal(self):
        R = rpy_to_rot([0.5, 0.3, 0.7])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_roll_90(self):
        R = rpy_to_rot([math.pi / 2, 0, 0])
        # Roll 90: y -> z, z -> -y
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_yaw_90(self):
        R = rpy_to_rot([0, 0, math.pi / 2])
        # Yaw 90: x -> y, y -> -x
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_returns_ndarray(self):
        R = rpy_to_rot([0.1, 0.2, 0.3])
        self.assertIsInstance(R, np.ndarray)
        self.assertNotIsInstance(R, np.matrix)


class TestEuler2Mat(unittest.TestCase):
    def test_identity(self):
        m = euler2mat([0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(m, np.eye(4), atol=1e-12)

    def test_translation_only(self):
        m = euler2mat([1.0, 2.0, 3.0, 0, 0, 0])
        self.assertAlmostEqual(m[0, 3], 1.0)
        self.assertAlmostEqual(m[1, 3], 2.0)
        self.assertAlmostEqual(m[2, 3], 3.0)
        np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-12)

    def test_shape_and_type(self):
        m = euler2mat([0, 0, 0, 0.1, 0.2, 0.3])
        self.assertEqual(m.shape, (4, 4))
        self.assertIsInstance(m, np.ndarray)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_allclose(m[3, :], [0, 0, 0, 1], atol=1e-12)


class TestMat2Euler(unittest.TestCase):
    def test_roundtrip(self):
        """euler -> mat -> euler should be identity."""
        poses = [
            [0, 0, 0, 0, 0, 0],
            [1.0, -0.5, 0.3, 0, 0, 0],
            [0, 0, 0, 0.5, 0.3, 0.7],
            [0.3, -0.1, 0.4, math.pi, 0, 0.5],
            [0.1, 0.2, 0.3, -0.5, 0.2, -1.0],
        ]
        for pose in poses:
            m = euler2mat(pose)
            result = mat2euler(m)
            np.testing.assert_allclose(result, pose, atol=1e-10,
                                       err_msg=f"Roundtrip failed for {pose}")


class TestConvertPose(unittest.TestCase):
    def test_identity_transform(self):
        T = np.eye(4)
        pose = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        result = convert_pose(pose, T)
        np.testing.assert_allclose(result, pose, atol=1e-10)

    def test_translation_only(self):
        T = euler2mat([1.0, 0, 0, 0, 0, 0])
        pose = [0.5, 0, 0, 0, 0, 0]
        result = convert_pose(pose, T)
        self.assertAlmostEqual(result[0], 1.5, places=10)

    def test_yaw_90_rotation(self):
        T = euler2mat([0, 0, 0, 0, 0, math.pi / 2])
        pose = [1.0, 0, 0, 0, 0, 0]
        result = convert_pose(pose, T)
        # Point (1,0,0) rotated 90 around Z -> (0,1,0)
        self.assertAlmostEqual(result[0], 0.0, places=10)
        self.assertAlmostEqual(result[1], 1.0, places=10)


class TestPixelToCameraFrame(unittest.TestCase):
    def test_center_pixel(self):
        # Center pixel should give (0, 0, depth)
        x, y, z = pixel_to_camera_frame(320, 240, 1.0, 500, 500, 320, 240)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 1.0)

    def test_offset_pixel(self):
        # 100px right of center at depth=2m, fx=500
        x, y, z = pixel_to_camera_frame(420, 240, 2.0, 500, 500, 320, 240)
        self.assertAlmostEqual(x, 100 / 500 * 2.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 2.0)


class TestLookupDepth(unittest.TestCase):
    def test_valid_region(self):
        depth = np.full((100, 100), 0.5)
        result = lookup_depth(depth, 50, 50, radius=5)
        self.assertAlmostEqual(result, 0.5)

    def test_nan_region(self):
        depth = np.full((100, 100), np.nan)
        result = lookup_depth(depth, 50, 50, radius=5)
        self.assertIsNone(result)

    def test_mixed_with_outliers(self):
        depth = np.full((100, 100), 0.4)
        depth[48:52, 48:52] = 0.0  # invalid zeros
        result = lookup_depth(depth, 50, 50, radius=5)
        self.assertAlmostEqual(result, 0.4)

    def test_edge_pixel(self):
        depth = np.full((100, 100), 0.3)
        result = lookup_depth(depth, 0, 0, radius=5)
        self.assertAlmostEqual(result, 0.3)


class TestPixelOffsetToBaseMm(unittest.TestCase):
    def test_zero_offset(self):
        """Zero pixel offset should give zero correction."""

        class FakeIntrinsics:
            fx = 500.0
            fy = 500.0
            ppx = 320.0
            ppy = 240.0

        eef = [0.3, 0, 0.4, math.pi, 0, 0]
        euler_eef_to_color = [0.067, -0.031, 0.022, -0.004, -0.008, 1.59]
        corr_x, corr_y = pixel_offset_to_base_mm(0, 0, 0.3, eef, FakeIntrinsics(), euler_eef_to_color)
        self.assertAlmostEqual(corr_x, 0.0, places=5)
        self.assertAlmostEqual(corr_y, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
