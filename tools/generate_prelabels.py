"""Generate YOLO-format pre-labels from existing LLM detection results.

Reads scan.json files in diagnostics sessions and converts the LLM bounding
boxes into YOLO annotation format (.txt files: class_id cx cy w h, normalized).

These pre-labels should be reviewed and corrected with a labeling tool before
training.

Usage:
    python tools/generate_prelabels.py [--diag-dir diagnostics] [--dataset-dir datasets/grape_kiwi]
"""

import argparse
import hashlib
import json
from pathlib import Path


# Must match data.yaml and yolo_detector.py CLASS_NAMES
CANONICAL_TO_CLASS_ID = {
    "green_grape_bunch": 0,
    "brown_kiwi": 1,
}

# Image resolution used during capture (from config.yaml defaults)
DEFAULT_IMG_WIDTH = 1920
DEFAULT_IMG_HEIGHT = 1080


def parse_scan_json(scan_json_path: Path) -> list[dict]:
    """Parse a scan.json and return list of {class_id, bbox} dicts."""
    try:
        data = json.loads(scan_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Warning: cannot read {scan_json_path}: {exc}")
        return []

    objects = data.get("objects", [])
    results = []
    for obj in objects:
        canonical = obj.get("canonical_name", "")
        class_id = CANONICAL_TO_CLASS_ID.get(canonical)
        if class_id is None:
            continue

        bbox = obj.get("bbox")
        if bbox is None:
            continue

        # bbox can be a dict with x1/y1/x2/y2 or with x1/y1/width/height
        if isinstance(bbox, dict):
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", bbox.get("x1", 0) + bbox.get("width", 0))
            y2 = bbox.get("y2", bbox.get("y1", 0) + bbox.get("height", 0))
        else:
            continue

        results.append({
            "class_id": class_id,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
        })

    return results


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel bbox to YOLO normalized format: cx cy w h."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def find_matching_image(scan_dir: Path, dataset_dir: Path) -> Path | None:
    """Find the dataset image file that corresponds to this scan directory."""
    # Reconstruct the unique filename used by prepare_dataset.py
    parts = scan_dir.parts
    session_name = ""
    scan_name = ""
    for p in parts:
        if p.startswith("session_"):
            session_name = p
        if p.startswith("scan_"):
            scan_name = p

    color_path = scan_dir / "color.jpg"
    if not color_path.exists():
        return None

    # The hash is computed from the relative path like prepare_dataset.py does
    try:
        rel = color_path.relative_to(color_path.parents[3])
    except ValueError:
        return None
    name_hash = hashlib.md5(str(rel).encode()).hexdigest()[:8]
    unique_name = f"{session_name}_{scan_name}_{name_hash}"

    # Search in both train and val
    for split in ("train", "val"):
        img_path = dataset_dir / split / "images" / f"{unique_name}.jpg"
        if img_path.exists():
            return img_path

    return None


def write_yolo_label(image_path: Path, detections: list[dict], img_w: int, img_h: int):
    """Write a YOLO label .txt file next to the image (in ../labels/)."""
    labels_dir = image_path.parent.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / image_path.with_suffix(".txt").name

    lines = []
    for det in detections:
        cx, cy, w, h = bbox_to_yolo(det["x1"], det["y1"], det["x2"], det["y2"], img_w, img_h)
        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return label_path


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO pre-labels from LLM detection results")
    parser.add_argument("--diag-dir", default="diagnostics", help="Diagnostics root directory")
    parser.add_argument("--dataset-dir", default="datasets/grape_kiwi", help="Dataset directory")
    parser.add_argument("--img-width", type=int, default=DEFAULT_IMG_WIDTH, help="Image width")
    parser.add_argument("--img-height", type=int, default=DEFAULT_IMG_HEIGHT, help="Image height")
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    dataset_dir = Path(args.dataset_dir)

    if not diag_dir.exists():
        print(f"Error: diagnostics directory '{diag_dir}' does not exist.")
        return

    scan_jsons = sorted(diag_dir.glob("session_*/scans/*/scan.json"))
    print(f"Found {len(scan_jsons)} scan.json files")

    stats = {"matched": 0, "labels_written": 0, "no_image": 0, "no_objects": 0}

    for scan_json in scan_jsons:
        scan_dir = scan_json.parent
        detections = parse_scan_json(scan_json)

        if not detections:
            stats["no_objects"] += 1
            continue

        image_path = find_matching_image(scan_dir, dataset_dir)
        if image_path is None:
            stats["no_image"] += 1
            continue

        stats["matched"] += 1
        label_path = write_yolo_label(image_path, detections, args.img_width, args.img_height)
        stats["labels_written"] += 1
        print(f"  {label_path.name}: {len(detections)} object(s)")

    print()
    print(f"Results: {stats['labels_written']} labels written, "
          f"{stats['matched']} scans matched, "
          f"{stats['no_image']} scans without dataset image, "
          f"{stats['no_objects']} scans without target objects")
    print()
    print("IMPORTANT: These are pre-labels generated from LLM detection results.")
    print("You MUST review and correct them with a labeling tool before training.")
    print("Recommended tools: LabelImg, CVAT, or Roboflow")


if __name__ == "__main__":
    main()
