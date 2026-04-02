"""Unit tests for YoloDetector.

The YOLO model is mocked so tests run without downloading weights.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestYoloDetectorInterface(unittest.TestCase):
    """Verify YoloDetector produces output compatible with the downstream pipeline."""

    def _make_detector(self, mock_yolo_cls):
        """Build a YoloDetector with a mocked YOLO model."""
        mock_yolo_cls.return_value = MagicMock()
        with patch.dict("sys.modules", {"ultralytics": MagicMock()}):
            from yolo_detector import YoloDetector
            # Patch the YOLO class used inside __init__
            with patch("yolo_detector.YOLO", mock_yolo_cls, create=True):
                pass
        # Construct manually to avoid import-time YOLO() call
        det = object.__new__(YoloDetector)
        det.model = mock_yolo_cls.return_value
        det.conf_threshold = 0.5
        det.iou_threshold = 0.5
        det.device = "cpu"
        det.last_raw_response_text = None
        return det

    def _make_mock_result(self, boxes_data):
        """Create a mock ultralytics result object with given box data.

        boxes_data: list of (cls_id, conf, x1, y1, x2, y2)
        """
        import torch

        mock_result = MagicMock()
        if boxes_data is None:
            mock_result.boxes = None
            return mock_result

        mock_boxes = []
        for cls_id, conf, x1, y1, x2, y2 in boxes_data:
            box = MagicMock()
            box.cls = torch.tensor([cls_id])
            box.conf = torch.tensor([conf])
            box.xyxy = torch.tensor([[x1, y1, x2, y2]])
            mock_boxes.append(box)

        mock_result.boxes = mock_boxes
        return mock_result

    @patch("yolo_detector.YOLO", create=True)
    def test_detect_multi_output_schema(self, mock_yolo_cls):
        """Detection results must have all required keys for the downstream pipeline."""
        import torch

        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result([
            (0, 0.92, 100, 200, 300, 400),  # green_grape_bunch
            (1, 0.85, 500, 100, 700, 250),  # brown_kiwi
        ])
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = det.detect_multi(color)

        self.assertEqual(len(results), 2)

        required_keys = {"name", "canonical_name", "category", "px", "py",
                         "bbox", "graspable", "grasp_reason", "confidence"}
        for r in results:
            self.assertTrue(required_keys.issubset(r.keys()),
                            f"Missing keys: {required_keys - r.keys()}")

        # First detection: green_grape_bunch
        self.assertEqual(results[0]["canonical_name"], "green_grape_bunch")
        self.assertEqual(results[0]["name"], "\u4e00\u4e32\u7eff\u8272\u8461\u8404")
        self.assertEqual(results[0]["category"], "fruit")
        self.assertEqual(results[0]["px"], 200)  # (100+300)//2
        self.assertEqual(results[0]["py"], 300)  # (200+400)//2
        self.assertAlmostEqual(results[0]["confidence"], 0.92, places=2)
        self.assertTrue(results[0]["graspable"])
        bbox = results[0]["bbox"]
        self.assertEqual(bbox, {"x1": 100, "y1": 200, "x2": 300, "y2": 400})

        # Second detection: brown_kiwi
        self.assertEqual(results[1]["canonical_name"], "brown_kiwi")
        self.assertEqual(results[1]["name"], "\u7315\u7334\u6843")

    @patch("yolo_detector.YOLO", create=True)
    def test_detect_multi_no_objects(self, mock_yolo_cls):
        """Empty results when no objects are detected."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result(None)
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = det.detect_multi(color)

        self.assertEqual(results, [])
        self.assertIsNone(det.last_raw_response_text)

    @patch("yolo_detector.YOLO", create=True)
    def test_detect_multi_unknown_class_filtered(self, mock_yolo_cls):
        """Unknown class IDs should be silently filtered out."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result([
            (99, 0.9, 10, 20, 100, 200),  # unknown class
        ])
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = det.detect_multi(color)

        self.assertEqual(results, [])

    @patch("yolo_detector.YOLO", create=True)
    def test_detect_multi_inference_error_returns_empty(self, mock_yolo_cls):
        """Inference errors should return empty list, not raise."""
        det = self._make_detector(mock_yolo_cls)
        det.model.predict.side_effect = RuntimeError("CUDA OOM")

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = det.detect_multi(color)

        self.assertEqual(results, [])

    @patch("yolo_detector.YOLO", create=True)
    def test_detect_multi_bbox_clamped(self, mock_yolo_cls):
        """Bounding boxes should be clamped to image dimensions."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result([
            (0, 0.8, -10, -5, 2000, 1200),  # exceeds 1920x1080
        ])
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = det.detect_multi(color)

        self.assertEqual(len(results), 1)
        bbox = results[0]["bbox"]
        self.assertGreaterEqual(bbox["x1"], 0)
        self.assertGreaterEqual(bbox["y1"], 0)
        self.assertLessEqual(bbox["x2"], 1919)
        self.assertLessEqual(bbox["y2"], 1079)

    @patch("yolo_detector.YOLO", create=True)
    def test_verify_identity_match(self, mock_yolo_cls):
        """verify_identity returns (True, name) when the expected object is detected."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result([
            (1, 0.9, 10, 10, 100, 100),  # brown_kiwi
        ])
        det.model.predict.return_value = [mock_result]

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        match, actual = det.verify_identity(crop, "brown_kiwi")

        self.assertTrue(match)
        self.assertEqual(actual, "brown_kiwi")

    @patch("yolo_detector.YOLO", create=True)
    def test_verify_identity_mismatch(self, mock_yolo_cls):
        """verify_identity returns (False, actual) when a different object is detected."""
        import torch

        det = self._make_detector(mock_yolo_cls)

        # Create mock boxes with conf attribute for max() comparison
        box = MagicMock()
        box.cls = torch.tensor([0])
        box.conf = torch.tensor([0.88])
        box.xyxy = torch.tensor([[10.0, 10.0, 100.0, 100.0]])
        mock_result = MagicMock()
        mock_result.boxes = [box]
        det.model.predict.return_value = [mock_result]

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        match, actual = det.verify_identity(crop, "brown_kiwi")

        self.assertFalse(match)
        self.assertEqual(actual, "green_grape_bunch")

    @patch("yolo_detector.YOLO", create=True)
    def test_verify_identity_empty_crop(self, mock_yolo_cls):
        """verify_identity returns (False, None) for empty crop."""
        det = self._make_detector(mock_yolo_cls)

        match, actual = det.verify_identity(None, "brown_kiwi")
        self.assertFalse(match)
        self.assertIsNone(actual)

    @patch("yolo_detector.YOLO", create=True)
    def test_verify_identity_fail_open(self, mock_yolo_cls):
        """verify_identity returns (True, None) when inference fails."""
        det = self._make_detector(mock_yolo_cls)
        det.model.predict.side_effect = RuntimeError("error")

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        match, actual = det.verify_identity(crop, "brown_kiwi")

        self.assertTrue(match)
        self.assertIsNone(actual)

    @patch("yolo_detector.YOLO", create=True)
    def test_last_raw_response_text_set_on_detection(self, mock_yolo_cls):
        """last_raw_response_text should be set after a successful detection."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result([
            (0, 0.7, 50, 50, 200, 200),
        ])
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        det.detect_multi(color)

        self.assertIsNotNone(det.last_raw_response_text)

    @patch("yolo_detector.YOLO", create=True)
    def test_max_retries_ignored(self, mock_yolo_cls):
        """max_retries parameter is accepted but does not cause retries."""
        det = self._make_detector(mock_yolo_cls)
        mock_result = self._make_mock_result(None)
        det.model.predict.return_value = [mock_result]

        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        det.detect_multi(color, max_retries=5)

        # predict should be called exactly once regardless of max_retries
        det.model.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
