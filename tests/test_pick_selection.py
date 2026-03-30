import unittest

import numpy as np

from pick_models import DetectedObject, ImageCapture, ScanResult
from pick_selection import TargetTracker, assess_targets


class TestTargetTracker(unittest.TestCase):
    def test_assigns_stable_ids_across_scans(self):
        tracker = TargetTracker(max_match_distance_px=80)

        first_scan = tracker.assign_ids(
            [
                DetectedObject(name="apple", px=100, py=100, confidence=0.9),
                DetectedObject(name="orange", px=400, py=300, confidence=0.8),
            ]
        )
        second_scan = tracker.assign_ids(
            [
                DetectedObject(name="orange", px=395, py=295, confidence=0.82),
                DetectedObject(name="apple", px=108, py=104, confidence=0.88),
            ]
        )

        self.assertEqual(first_scan[0].display_id, second_scan[1].display_id)
        self.assertEqual(first_scan[1].display_id, second_scan[0].display_id)

    def test_new_object_gets_new_id(self):
        tracker = TargetTracker(max_match_distance_px=80)
        first_scan = tracker.assign_ids([DetectedObject(name="apple", px=100, py=100, confidence=0.9)])
        second_scan = tracker.assign_ids(
            [
                DetectedObject(name="apple", px=105, py=98, confidence=0.88),
                DetectedObject(name="banana", px=300, py=250, confidence=0.7),
            ]
        )

        self.assertEqual(first_scan[0].display_id, second_scan[0].display_id)
        self.assertNotEqual(second_scan[0].display_id, second_scan[1].display_id)


class TestAssessTargets(unittest.TestCase):
    def test_prefers_confident_centered_target_with_depth(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.full((1080, 1920), np.nan, dtype=np.float64)
        depth[530:551, 950:971] = 0.35
        depth[50:71, 50:71] = 0.35

        scan_result = ScanResult(
            objects=[
                DetectedObject(name="center", px=960, py=540, confidence=0.95, target_id="T001"),
                DetectedObject(name="edge", px=60, py=60, confidence=0.30, target_id="T002"),
            ],
            images=ImageCapture(depth=depth, color=color),
        )

        assessments = assess_targets(scan_result)
        self.assertEqual(assessments[0].detected_object.name, "center")
        self.assertGreater(assessments[0].total_score, assessments[1].total_score)
        self.assertIn("depth=ok", assessments[0].reasons)

    def test_missing_depth_is_penalized(self):
        color = np.zeros((600, 800, 3), dtype=np.uint8)
        depth = np.full((600, 800), np.nan, dtype=np.float64)
        depth[295:306, 395:406] = 0.4

        scan_result = ScanResult(
            objects=[
                DetectedObject(name="with_depth", px=400, py=300, confidence=0.6, target_id="T001"),
                DetectedObject(name="no_depth", px=410, py=310, confidence=0.95, target_id="T002"),
            ],
            images=ImageCapture(depth=depth, color=color),
        )

        assessments = assess_targets(scan_result)
        by_name = {assessment.detected_object.name: assessment for assessment in assessments}
        self.assertGreater(by_name["with_depth"].depth_score, by_name["no_depth"].depth_score)

    def test_ungraspable_object_is_penalized(self):
        color = np.zeros((600, 800, 3), dtype=np.uint8)
        depth = np.full((600, 800), 0.35, dtype=np.float64)

        scan_result = ScanResult(
            objects=[
                DetectedObject(name="easy", px=400, py=300, confidence=0.7, graspable=True, target_id="T001"),
                DetectedObject(name="blocked", px=395, py=295, confidence=0.95, graspable=False, target_id="T002"),
            ],
            images=ImageCapture(depth=depth, color=color),
        )

        assessments = assess_targets(scan_result)
        self.assertEqual(assessments[0].detected_object.name, "easy")
        self.assertLess(assessments[1].graspability_score, 0)


if __name__ == "__main__":
    unittest.main()
