"""YOLO-based local object detector for fruit picking.

Drop-in replacement for LLMDetector — produces the same output format
so that all downstream code (physical filtering, coordinate conversion,
target assessment, visual servo) works without modification.
"""

import logging

import cv2
import numpy as np

from llm_detector import (
    TARGET_DISPLAY_NAMES,
    _normalize_target_canonical_name,
)

logger = logging.getLogger(__name__)

# YOLO class index -> canonical name (must match data.yaml training config)
CLASS_NAMES = {
    0: "green_grape_bunch",
    1: "brown_kiwi",
}


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


class YoloDetector:
    """Local YOLO-based object detector.

    Public API mirrors ``LLMDetector`` so the two are interchangeable:
      - ``detect_multi(color_image, max_retries=2) -> list[dict]``
      - ``verify_identity(crop_image, expected_name) -> (bool, str|None)``
      - ``last_raw_response_text``  (str|None, for diagnostics compatibility)
    """

    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.5, device=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or "cpu"
        self.last_raw_response_text = None

    # ------------------------------------------------------------------
    # Primary detection (replaces LLMDetector.detect_multi)
    # ------------------------------------------------------------------

    def detect_multi(self, color_image, max_retries=2):
        """Detect target fruits in *color_image*.

        Returns a list of dicts identical in schema to those produced by
        ``LLMDetector.detect_multi`` — coordinates are in the original
        image space, **not** a downscaled version.

        *max_retries* is accepted for interface compatibility but unused
        because local inference does not require retries.
        """
        self.last_raw_response_text = None
        orig_h, orig_w = color_image.shape[:2]

        try:
            results = self.model.predict(
                source=color_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = _clamp(int(x1), 0, orig_w - 1)
                    y1 = _clamp(int(y1), 0, orig_h - 1)
                    x2 = _clamp(int(x2), 0, orig_w - 1)
                    y2 = _clamp(int(y2), 0, orig_h - 1)

                    # Grasp point = bbox centre
                    px = (x1 + x2) // 2
                    py = (y1 + y2) // 2

                    canonical_name = CLASS_NAMES.get(cls_id)
                    if canonical_name is None:
                        continue

                    detections.append({
                        "name": TARGET_DISPLAY_NAMES.get(canonical_name, canonical_name),
                        "canonical_name": canonical_name,
                        "category": "fruit",
                        "px": px,
                        "py": py,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "graspable": True,
                        "grasp_reason": "YOLO detection",
                        "confidence": round(conf, 4),
                    })

        if detections:
            logger.info(
                "YOLO found %d object(s): %s",
                len(detections),
                ", ".join(
                    f"{d['name']}[{d['category']}]({d['px']},{d['py']}) "
                    f"conf={d['confidence']:.2f}"
                    for d in detections
                ),
            )
        else:
            logger.info("YOLO: no objects detected")

        self.last_raw_response_text = str(detections) if detections else None
        return detections

    # ------------------------------------------------------------------
    # Identity verification (replaces LLMDetector.verify_identity)
    # ------------------------------------------------------------------

    def verify_identity(self, crop_image, expected_name):
        """Verify that *crop_image* contains the expected object.

        Returns ``(match: bool, actual_name: str | None)``.
        Fail-open: returns ``(True, None)`` on any error so the grasp is
        not blocked by a verification failure.
        """
        if crop_image is None or crop_image.size == 0:
            return False, None

        try:
            results = self.model.predict(
                source=crop_image,
                conf=self.conf_threshold * 0.8,  # slightly lower for verification
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("[Verify] YOLO inference failed: %s", exc)
            return True, None  # fail-open

        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Normalise expected name via the same alias map used by the LLM path
            expected_canonical = _normalize_target_canonical_name(expected_name)

            # Check whether any detection matches
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                detected_name = CLASS_NAMES.get(cls_id, "unknown")
                if detected_name == expected_canonical:
                    logger.info("[Verify] Confirmed: %s", expected_name)
                    return True, detected_name

            # Detections exist but none match
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            actual = CLASS_NAMES.get(int(best_box.cls[0]), "unknown")
            logger.info("[Verify] Expected '%s', YOLO sees '%s'", expected_name, actual)
            return False, actual

        # Nothing detected in the crop — fail-open
        logger.warning("[Verify] No detection in crop, assuming match")
        return True, None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python yolo_detector.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        sys.exit(1)

    detector = YoloDetector(model_path=model_path, conf_threshold=0.5)
    results = detector.detect_multi(img)
    print(f"Found {len(results)} object(s):")
    for r in results:
        print(
            f"  {r['name']}[{r['category']}]: ({r['px']}, {r['py']}) "
            f"conf={r['confidence']:.2f} graspable={r['graspable']}"
        )
