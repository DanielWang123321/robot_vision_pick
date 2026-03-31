import base64
import json
import logging
import re
import time

import cv2
import requests

from pick_models import _to_bool

logger = logging.getLogger(__name__)

LLM_WIDTH = 640
LLM_HEIGHT = 360
CATEGORIES = ("fruit", "tissue", "container", "tool", "stationery", "toy", "other", "unknown")

SYSTEM_PROMPT = (
    f"You are a robot vision assistant. You analyze top-down images from a robot arm camera "
    f"and locate target objects. Image resolution: {LLM_WIDTH}x{LLM_HEIGHT}. "
    f"Return ONLY valid JSON, no other text."
)

MULTI_PROMPT = (
    f"Find ALL standalone objects on the table in this top-down image. Ignore the table itself, robot hardware, "
    f"background clutter, and anything inside containers. Objects can be fruits, tissues, cups, bottles, tools, "
    f"stationery, toys, and other common desktop items.\n"
    f"注意区分外观相似的水果：猕猴桃(kiwi)棕色/深绿色毛糙表皮有细小绒毛；"
    f"柠檬(lemon)亮黄色光滑表皮两端有尖突；葡萄(grape)一整串绿色小圆粒。\n"
    f"Return a JSON array. One object per item. Use this schema exactly:\n"
    "["
    "{"
    '"name": "<common Chinese object name>", '
    '"canonical_name": "<short normalized object id>", '
    f'"category": "<one of {", ".join(CATEGORIES)}>", '
    '"px": <integer grasp point x>, '
    '"py": <integer grasp point y>, '
    '"bbox": {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}, '
    '"graspable": <true_or_false>, '
    '"grasp_reason": "<short reason>", '
    '"confidence": <0_to_1>'
    "}"
    "]\n"
    f"Coordinates must use the {LLM_WIDTH}x{LLM_HEIGHT} image space. "
    f"px and py must point near the best top-down grasp point for that object. "
    f"bbox must tightly cover the visible object. "
    f"Return [] if no objects are found."
)

TARGET_SYSTEM_PROMPT = (
    f"You are a robot vision assistant for fruit picking. "
    f"Detect only the allowed targets in a {LLM_WIDTH}x{LLM_HEIGHT} top-down image. "
    f"Return JSON only."
)

TARGET_MULTI_PROMPT = (
    "Detect only these targets: "
    "green_grape_bunch = one whole bunch of green grapes, "
    "brown_kiwi = one kiwi fruit (can appear brown, golden, or yellowish, with rough or fuzzy skin). "
    "For grapes, return the whole bunch as one object, not single grapes. "
    "If unsure whether an object is a target, include it with a lower confidence score. "
    "Return a JSON array using "
    '[{"name":"<english name>","canonical_name":"<green_grape_bunch|brown_kiwi>","category":"fruit","px":<int>,"py":<int>,"bbox":{"x1":<int>,"y1":<int>,"x2":<int>,"y2":<int>},"graspable":<true_or_false>,"grasp_reason":"<short reason>","confidence":<0_to_1>}]. '
    f"Use {LLM_WIDTH}x{LLM_HEIGHT} coordinates. Return [] if no target is found."
)

VERIFY_SYSTEM_PROMPT = (
    "You verify fruit identity for a picking robot. "
    "Allowed targets: green_grape_bunch and brown_kiwi. "
    "Return JSON only."
)


def _normalize_canonical_name(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


TARGET_DISPLAY_NAMES = {
    "green_grape_bunch": "\u4e00\u4e32\u7eff\u8272\u8461\u8404",
    "brown_kiwi": "\u7315\u7334\u6843",
}

TARGET_ENGLISH_NAMES = {
    "green_grape_bunch": "green grape bunch",
    "brown_kiwi": "brown kiwi",
}

TARGET_CANONICAL_ALIASES = {
    "green_grape_bunch": "green_grape_bunch",
    "green_grapes": "green_grape_bunch",
    "green_grape": "green_grape_bunch",
    "grape_bunch": "green_grape_bunch",
    "bunch_of_green_grapes": "green_grape_bunch",
    "bunch_of_grapes": "green_grape_bunch",
    "grapes": "green_grape_bunch",
    "\u4e00\u4e32\u7eff\u8272\u8461\u8404": "green_grape_bunch",
    "\u8461\u8404": "green_grape_bunch",
    "brown_kiwi": "brown_kiwi",
    "kiwi": "brown_kiwi",
    "kiwifruit": "brown_kiwi",
    "brown_kiwifruit": "brown_kiwi",
    "\u7315\u7334\u6843": "brown_kiwi",
    "\u5947\u5f02\u679c": "brown_kiwi",
}


def _normalize_target_canonical_name(value):
    canonical = _normalize_canonical_name(value)
    return TARGET_CANONICAL_ALIASES.get(canonical, canonical)


def _display_name_for_target(canonical_name, fallback_name):
    return TARGET_DISPLAY_NAMES.get(canonical_name, fallback_name)


def _english_name_for_target(canonical_name, fallback_name):
    return TARGET_ENGLISH_NAMES.get(canonical_name, fallback_name)


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


class LLMDetector:
    def __init__(self, api_key, api_url, model):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.last_raw_response_text = None

    def detect_multi(self, color_image, max_retries=2):
        """Detect allowed target fruits in color_image and scale results to the original image."""
        self.last_raw_response_text = None
        orig_h, orig_w = color_image.shape[:2]

        llm_img = cv2.resize(color_image, (LLM_WIDTH, LLM_HEIGHT))
        _, jpeg_buf = cv2.imencode(".jpg", llm_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_image = base64.b64encode(jpeg_buf).decode("utf-8")

        for attempt in range(max_retries + 1):
            try:
                text = self._call_api(
                    b64_image,
                    TARGET_MULTI_PROMPT,
                    system_prompt=TARGET_SYSTEM_PROMPT,
                )
                results = self._parse_multi_response(text)
                if results is not None:
                    scaled = [self._scale_detection(result, orig_w, orig_h) for result in results]
                    if scaled:
                        logger.info(
                            "Found %d object(s): %s",
                            len(scaled),
                            ", ".join(f"{r['name']}[{r['category']}]({r['px']},{r['py']})" for r in scaled),
                        )
                    else:
                        logger.info("No objects detected")
                    return scaled
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (401, 403, 404):
                    logger.error("Non-retriable HTTP error %d: %s", status, exc)
                    break
                logger.warning("Attempt %d failed (HTTP %d): %s", attempt + 1, status, exc)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                logger.warning("Attempt %d failed (network): %s", attempt + 1, exc)
            except Exception as exc:
                logger.warning("Attempt %d failed: %s", attempt + 1, exc)

            if attempt < max_retries:
                delay = min(2 ** attempt, 8)
                logger.info("Retrying in %ds...", delay)
                time.sleep(delay)
        return []

    def _call_api(self, b64_image, prompt, system_prompt=SYSTEM_PROMPT, max_tokens=700, timeout=30):
        """Send an image + prompt to the LLM and return raw response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        self.last_raw_response_text = text
        return text

    def verify_identity(self, crop_image, expected_name):
        """Verify that a close-up crop shows the expected object.

        Returns (match: bool, actual_name: str or None).
        """
        if crop_image is None or crop_image.size == 0:
            return False, None

        h, w = crop_image.shape[:2]
        if max(h, w) > 320:
            scale = 320 / max(h, w)
            crop_image = cv2.resize(crop_image, (int(w * scale), int(h * scale)))

        _, jpeg_buf = cv2.imencode(".jpg", crop_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(jpeg_buf).decode("utf-8")
        expected_canonical = _normalize_target_canonical_name(expected_name)
        expected_label = _english_name_for_target(expected_canonical, str(expected_name))

        prompt = (
            "You see a close-up from a robot camera. "
            "Allowed targets: green_grape_bunch (one whole bunch of green grapes) "
            "and brown_kiwi (one brown kiwi with rough or fuzzy skin). "
            f'Is this object "{expected_label}"?\n'
            'Reply ONLY with JSON: {"match": true} or {"match": false, "actual": "<english label>"}'
        )

        try:
            text = self._call_api(
                b64_image, prompt,
                system_prompt=VERIFY_SYSTEM_PROMPT,
                max_tokens=60,
                timeout=15,
            )
            logger.debug("[Verify] Raw response: %s", text)

            text_clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
            data = json.loads(text_clean)
            match = bool(data.get("match", False))
            actual = data.get("actual")
            if not match:
                logger.info("[Verify] Expected '%s', LLM sees '%s'", expected_name, actual)
            else:
                logger.info("[Verify] Confirmed: %s", expected_name)
            return match, actual
        except Exception as exc:
            logger.warning("[Verify] Verification failed: %s", exc)
            return True, None  # fail-open: don't block grasp on verification error

    def _scale_detection(self, result, orig_w, orig_h):
        px_orig = _clamp(int(result["px"] * orig_w / LLM_WIDTH), 0, orig_w - 1)
        py_orig = _clamp(int(result["py"] * orig_h / LLM_HEIGHT), 0, orig_h - 1)
        scaled = {
            "name": result.get("name", "unknown"),
            "canonical_name": result.get("canonical_name") or _normalize_canonical_name(result.get("name", "unknown")),
            "category": result.get("category", "unknown"),
            "px": px_orig,
            "py": py_orig,
            "graspable": _to_bool(result.get("graspable", True), default=True),
            "grasp_reason": result.get("grasp_reason"),
            "confidence": float(result.get("confidence", 0.0)),
        }

        bbox = result.get("bbox")
        if bbox is not None:
            scaled["bbox"] = {
                "x1": _clamp(int(bbox["x1"] * orig_w / LLM_WIDTH), 0, orig_w - 1),
                "y1": _clamp(int(bbox["y1"] * orig_h / LLM_HEIGHT), 0, orig_h - 1),
                "x2": _clamp(int(bbox["x2"] * orig_w / LLM_WIDTH), 0, orig_w - 1),
                "y2": _clamp(int(bbox["y2"] * orig_h / LLM_HEIGHT), 0, orig_h - 1),
            }
        return scaled

    def _extract_json_payload(self, text):
        text_clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        try:
            return json.loads(text_clean)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text_clean, re.DOTALL)
            if not match:
                logger.warning("Cannot parse response: %s", text[:200])
                return None
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("JSON parse failed: %s", text[:200])
                return None

    def _parse_bbox(self, item):
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            try:
                x1 = int(bbox["x1"])
                y1 = int(bbox["y1"])
                x2 = int(bbox["x2"])
                y2 = int(bbox["y2"])
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(value) for value in bbox]
            except (TypeError, ValueError):
                return None
        else:
            return None

        left = _clamp(min(x1, x2), 0, LLM_WIDTH - 1)
        right = _clamp(max(x1, x2), 0, LLM_WIDTH - 1)
        top = _clamp(min(y1, y2), 0, LLM_HEIGHT - 1)
        bottom = _clamp(max(y1, y2), 0, LLM_HEIGHT - 1)
        if right <= left or bottom <= top:
            return None
        return {"x1": left, "y1": top, "x2": right, "y2": bottom}

    def _parse_point(self, item):
        point = item.get("grasp_point")
        if isinstance(point, dict) and "px" in point and "py" in point:
            try:
                return int(point["px"]), int(point["py"])
            except (TypeError, ValueError):
                return None
        try:
            return int(item["px"]), int(item["py"])
        except (KeyError, TypeError, ValueError):
            return None

    def _parse_multi_response(self, text):
        """Parse LLM response expecting a JSON array of object detections."""
        data = self._extract_json_payload(text)
        if data is None:
            return None
        if not isinstance(data, list):
            logger.warning("Expected array, got %s", type(data).__name__)
            return None

        results = []
        for item in data:
            try:
                point = self._parse_point(item)
                if point is None:
                    logger.debug("Skipping item without valid point")
                    continue
                px, py = point
                if not (0 <= px < LLM_WIDTH and 0 <= py < LLM_HEIGHT):
                    logger.debug("Skipping out-of-range: (%d, %d)", px, py)
                    continue

                category = str(item.get("category", "unknown")).strip().lower()
                if category not in CATEGORIES:
                    category = "unknown"

                canonical_name = _normalize_target_canonical_name(item.get("canonical_name") or item.get("name"))
                if canonical_name not in TARGET_DISPLAY_NAMES:
                    logger.debug("Skipping non-target item: %s", item)
                    continue

                raw_name = str(item.get("name", "unknown"))
                results.append(
                    {
                        "name": _display_name_for_target(canonical_name, raw_name),
                        "canonical_name": canonical_name,
                        "category": "fruit",
                        "px": px,
                        "py": py,
                        "bbox": self._parse_bbox(item),
                        "graspable": _to_bool(item.get("graspable", True), default=True),
                        "grasp_reason": None if item.get("grasp_reason") is None else str(item.get("grasp_reason")),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                )
            except ValueError as exc:
                logger.debug("Skipping invalid item: %s", exc)
                continue

        return results


if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("OPENROUTER_KEY")
    detector = LLMDetector(
        api_key=key,
        api_url="https://openrouter.ai/api/v1/chat/completions",
        model="anthropic/claude-sonnet-4.6",
    )
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Cannot read image: {sys.argv[1]}")
            sys.exit(1)
        results = detector.detect_multi(img, max_retries=2)
        print(f"Found {len(results)} object(s):")
        for result in results:
            print(
                f"  {result['name']}[{result['category']}]: ({result['px']}, {result['py']}) "
                f"conf={result['confidence']:.2f} graspable={result['graspable']}"
            )
    else:
        print("Usage: python llm_detector.py <image_path>")
