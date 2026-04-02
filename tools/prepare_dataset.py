"""Extract training images from diagnostics sessions.

Scans all diagnostics/session_*/scans/*/color.jpg files and copies them
into the YOLO dataset directory with unique filenames.

Usage:
    python tools/prepare_dataset.py [--diag-dir diagnostics] [--out-dir datasets/grape_kiwi] [--val-ratio 0.2]
"""

import argparse
import hashlib
import random
import shutil
from pathlib import Path


def collect_images(diag_dir: Path) -> list[Path]:
    """Find all color.jpg files under diagnostics sessions."""
    images = sorted(diag_dir.glob("session_*/scans/*/color.jpg"))
    print(f"Found {len(images)} color images in {diag_dir}")
    return images


def copy_images(images: list[Path], out_dir: Path, val_ratio: float):
    """Copy images into train/images and val/images with stable split."""
    train_dir = out_dir / "train" / "images"
    val_dir = out_dir / "val" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(images)

    val_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
    val_set = set(range(val_count))

    stats = {"train": 0, "val": 0, "skipped": 0}

    for idx, img_path in enumerate(images):
        # Generate a unique filename from the session/scan path
        rel = img_path.relative_to(img_path.parents[3])  # e.g. session_.../scans/scan_.../color.jpg
        name_hash = hashlib.md5(str(rel).encode()).hexdigest()[:8]
        # Extract session and scan identifiers for readability
        parts = img_path.parts
        session_name = ""
        scan_name = ""
        for i, p in enumerate(parts):
            if p.startswith("session_"):
                session_name = p
            if p.startswith("scan_"):
                scan_name = p
        unique_name = f"{session_name}_{scan_name}_{name_hash}.jpg"

        target_dir = val_dir if idx in val_set else train_dir
        dest = target_dir / unique_name

        if dest.exists():
            stats["skipped"] += 1
            continue

        shutil.copy2(img_path, dest)
        if idx in val_set:
            stats["val"] += 1
        else:
            stats["train"] += 1

    print(f"Copied: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped (already exist)")
    print(f"Train images: {train_dir}")
    print(f"Val images:   {val_dir}")
    print()
    print("Next steps:")
    print("  1. Run: python tools/generate_prelabels.py  (generate initial labels from LLM results)")
    print("  2. Review and correct labels with a labeling tool (e.g. LabelImg)")
    print("  3. Run: python tools/train_yolo.py  (start training)")


def main():
    parser = argparse.ArgumentParser(description="Extract training images from diagnostics")
    parser.add_argument("--diag-dir", default="diagnostics", help="Diagnostics root directory")
    parser.add_argument("--out-dir", default="datasets/grape_kiwi", help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images for validation")
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)

    if not diag_dir.exists():
        print(f"Error: diagnostics directory '{diag_dir}' does not exist.")
        print("Run the system first to collect images, or specify --diag-dir.")
        return

    images = collect_images(diag_dir)
    if not images:
        print("No images found. Run the system to collect diagnostic images first.")
        return

    copy_images(images, out_dir, args.val_ratio)


if __name__ == "__main__":
    main()
