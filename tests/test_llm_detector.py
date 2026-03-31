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
            "name": "kiwi",
            "canonical_name": "brown_kiwi",
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
        self.assertEqual(result["name"], "\u7315\u7334\u6843")
        self.assertEqual(result["canonical_name"], "brown_kiwi")
        self.assertEqual(result["category"], "fruit")
        self.assertEqual((result["px"], result["py"]), (320, 180))
        self.assertEqual(result["bbox"], {"x1": 280, "y1": 140, "x2": 360, "y2": 230})
        self.assertFalse(result["graspable"])
        self.assertEqual(result["grasp_reason"], "occluded by another object")

    def test_non_target_response_is_filtered_out(self):
        text = '[{"name": "apple", "canonical_name": "apple", "px": 100, "py": 120, "confidence": 0.55}]'
        results = self.detector._parse_multi_response(text)
        self.assertEqual(results, [])

    def test_detect_multi_scales_bbox_and_point(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_results = [
            {
                "name": "green grape bunch",
                "canonical_name": "green_grape_bunch",
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
        self.assertEqual(result["name"], "\u4e00\u4e32\u7eff\u8272\u8461\u8404")
        self.assertEqual(result["canonical_name"], "green_grape_bunch")
        self.assertEqual(result["category"], "fruit")


if __name__ == "__main__":
    unittest.main()
