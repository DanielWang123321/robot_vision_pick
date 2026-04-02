"""Train a YOLOv8 model on the grape/kiwi dataset.

Usage:
    python tools/train_yolo.py [--data datasets/grape_kiwi/data.yaml] [--epochs 100] [--device cpu]

After training, the best model is saved to runs/detect/<name>/weights/best.pt.
Copy it to models/grape_kiwi_best.pt for use with the pick system:
    cp runs/detect/grape_kiwi_v1/weights/best.pt models/grape_kiwi_best.pt
"""

import argparse
import shutil
from pathlib import Path


def check_dataset(data_yaml: str):
    """Verify the dataset directory has images and labels."""
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: data.yaml not found at {data_path}")
        return False

    dataset_root = data_path.parent
    for split in ("train", "val"):
        img_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            print(f"Error: {img_dir} does not exist")
            return False

        img_count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        print(f"  {split}: {img_count} images, {label_count} labels")
        if img_count == 0:
            print(f"Error: no images in {img_dir}")
            print("Run 'python tools/prepare_dataset.py' first to collect images.")
            return False
        if label_count == 0:
            print(f"Warning: no labels in {label_dir}")
            print("Run 'python tools/generate_prelabels.py' first, then review with a labeling tool.")
            return False
        if label_count < img_count:
            print(f"Warning: only {label_count}/{img_count} images have labels")

    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on grape/kiwi dataset")
    parser.add_argument("--data", default="datasets/grape_kiwi/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu', '0' (GPU 0), '0,1' (multi-GPU). Auto-detect if not set.")
    parser.add_argument("--name", default="grape_kiwi_v1", help="Run name")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--export-to", default="models/grape_kiwi_best.pt",
                        help="Copy best.pt to this path after training")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLOv8 Training — grape_kiwi dataset")
    print("=" * 60)
    print()
    print(f"  Data:     {args.data}")
    print(f"  Model:    {args.model}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {args.device or 'auto'}")
    print(f"  Name:     {args.name}")
    print(f"  Patience: {args.patience}")
    print()

    print("Checking dataset...")
    if not check_dataset(args.data):
        return

    print()
    print("Starting training...")
    print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed.")
        print("Install with: pip install ultralytics")
        return

    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "patience": args.patience,
        "augment": True,
        "verbose": True,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    # Copy best model to target path
    best_pt = Path(f"runs/detect/{args.name}/weights/best.pt")
    if best_pt.exists() and args.export_to:
        export_path = Path(args.export_to)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, export_path)
        print()
        print(f"Best model copied to: {export_path}")
        print()
        print("To use this model, set in config.yaml:")
        print("  detection:")
        print("    backend: yolo")
        print("    yolo:")
        print(f"      model_path: \"{export_path}\"")
    else:
        print(f"Warning: best.pt not found at {best_pt}")
        print("Check runs/detect/ for training output.")

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
