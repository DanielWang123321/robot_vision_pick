import json
import unittest
from unittest.mock import patch

import numpy as np

from llm_detector import LLMDetector


class TestLLMDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LLMDetector(api_key="test", api_url="https://example.com", model="mock-model")

    def test_parse_enriched_response(self):
        text = """```json
        [
          {
            "name": "山竹",
            "canonical_name": "mangosteen",
            "category": "fruit",
            "grasp_point": {"px": 320, "py": 180},
            "bbox": {"x1": 280, "y1": 140, "x2": 360, "y2": 230},
            "graspable": "false",
            "grasp_reason": "occluded by another object",
            "confidence": 0.91
          }
        ]
        ```"""

        results = self.detector._parse_multi_response(text)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["name"], "山竹")
        self.assertEqual(result["canonical_name"], "mangosteen")
        self.assertEqual(result["category"], "fruit")
        self.assertEqual((result["px"], result["py"]), (320, 180))
        self.assertEqual(result["bbox"], {"x1": 280, "y1": 140, "x2": 360, "y2": 230})
        self.assertFalse(result["graspable"])
        self.assertEqual(result["grasp_reason"], "occluded by another object")

    def test_parse_legacy_response_is_still_supported(self):
        text = '[{"name": "纸巾", "px": 100, "py": 120, "confidence": 0.55}]'
        results = self.detector._parse_multi_response(text)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["category"], "unknown")
        self.assertIsNone(result["bbox"])
        self.assertTrue(result["graspable"])
        self.assertEqual(result["canonical_name"], "纸巾")

    def test_detect_multi_scales_bbox_and_point(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_results = [
            {
                "name": "苹果",
                "canonical_name": "apple",
                "category": "fruit",
                "px": 320,
                "py": 180,
                "bbox": {"x1": 300, "y1": 150, "x2": 340, "y2": 210},
                "graspable": True,
                "grasp_reason": "clear top surface",
                "confidence": 0.8,
            }
        ]

        with patch.object(self.detector, "_call_api", return_value=json.dumps(mock_results)):
            results = self.detector.detect_multi(color, max_retries=0)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual((result["px"], result["py"]), (960, 540))
        self.assertEqual(result["bbox"], {"x1": 900, "y1": 450, "x2": 1020, "y2": 630})
        self.assertEqual(result["canonical_name"], "apple")
        self.assertEqual(result["category"], "fruit")


if __name__ == "__main__":
    unittest.main()
