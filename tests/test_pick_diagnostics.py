import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pick_system import DiagnosticsRecorder
from pick_models import BoundingBox, DetectedObject, ImageCapture, PickResult, ScanResult, TargetAssessment


class TestDiagnosticsRecorder(unittest.TestCase):
    def test_records_scan_assessments_and_pick_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DiagnosticsRecorder(
                {
                    "enabled": True,
                    "output_dir": temp_dir,
                    "save_color": False,
                    "save_depth": False,
                    "save_llm_raw": True,
                    "save_assessments": True,
                }
            )

            detected_object = DetectedObject(
                name="apple",
                px=320,
                py=240,
                confidence=0.9,
                canonical_name="apple",
                category="fruit",
                graspable=True,
                grasp_reason="clear top view",
                bbox=BoundingBox(280, 200, 360, 280),
                target_id="T001",
            )
            scan_result = ScanResult(
                objects=[detected_object],
                images=ImageCapture(
                    depth=np.full((480, 640), 0.35, dtype=np.float64),
                    color=np.zeros((480, 640, 3), dtype=np.uint8),
                ),
                llm_raw_response='[{"name":"apple"}]',
            )

            recorded_scan = recorder.record_scan(scan_result)
            self.assertIsNotNone(recorded_scan.scan_id)

            scan_dir = Path(temp_dir) / recorder.session_dir.name / "scans" / recorded_scan.scan_id
            scan_payload = json.loads((scan_dir / "scan.json").read_text(encoding="utf-8"))
            self.assertEqual(scan_payload["scan_id"], recorded_scan.scan_id)
            self.assertEqual(scan_payload["objects"][0]["target_id"], "T001")
            self.assertEqual(scan_payload["llm_raw_response"], '[{"name":"apple"}]')

            assessment = TargetAssessment(
                detected_object=detected_object,
                total_score=88.0,
                confidence_score=40.5,
                graspability_score=12.0,
                border_score=10.0,
                center_score=12.0,
                isolation_score=8.0,
                depth_score=10.0,
                bbox_score=5.5,
                reasons=("confidence=0.90", "depth=ok"),
            )
            recorder.record_assessments(recorded_scan, [assessment])
            assessments_payload = json.loads((scan_dir / "assessments.json").read_text(encoding="utf-8"))
            self.assertEqual(assessments_payload["assessments"][0]["target"]["name"], "apple")

            pick_result = PickResult(True, "success", detected_object=detected_object)
            recorder.record_pick(recorded_scan, pick_result, selected_assessment=assessment)

            picks_dir = Path(temp_dir) / recorder.session_dir.name / "picks"
            pick_files = list(picks_dir.glob("*/pick.json"))
            self.assertEqual(len(pick_files), 1)
            pick_payload = json.loads(pick_files[0].read_text(encoding="utf-8"))
            self.assertEqual(pick_payload["scan_id"], recorded_scan.scan_id)
            self.assertEqual(pick_payload["selected_assessment"]["target"]["target_id"], "T001")
            self.assertEqual(pick_payload["pick_result"]["reason"], "success")


if __name__ == "__main__":
    unittest.main()
