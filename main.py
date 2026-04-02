import argparse
import logging
import os
import signal
import sys

import yaml


_ctrl_c_count = 0


def _sigint_handler(sig, frame):
    global _ctrl_c_count
    _ctrl_c_count += 1
    if _ctrl_c_count >= 2:
        print("\n[Force Exit] Second Ctrl+C received, terminating immediately.")
        os._exit(1)
    print("\n[Ctrl+C] Stopping... press Ctrl+C again to force exit.")
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _sigint_handler)

from pick_system import VisionPickSystem
from pick_workflows import run_interactive, run_loop, run_pick_all


logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def validate_config(cfg):
    """Validate config has required fields with sane values. Raises ValueError on failure."""
    rc = cfg.get("robot")
    if not rc:
        raise ValueError("Missing 'robot' section in config")
    for key in ("ip", "safe_bounds", "detect_xyz", "release_xyz", "gripper_z_mm", "grasping_min_z"):
        if key not in rc:
            raise ValueError(f"Missing robot.{key} in config")

    safe_bounds = rc["safe_bounds"]
    for axis in ("x", "y", "z"):
        if axis not in safe_bounds or len(safe_bounds[axis]) != 2 or safe_bounds[axis][0] >= safe_bounds[axis][1]:
            raise ValueError(f"Invalid robot.safe_bounds.{axis}: must be [min, max] with min < max")

    for position_key in ("detect_xyz", "release_xyz"):
        position = rc[position_key]
        if len(position) != 3:
            raise ValueError(f"robot.{position_key} must have 3 elements [x, y, z]")
        x_pos, y_pos, z_pos = position
        x_bounds, y_bounds, z_bounds = safe_bounds["x"], safe_bounds["y"], safe_bounds["z"]
        if not (
            x_bounds[0] <= x_pos <= x_bounds[1]
            and y_bounds[0] <= y_pos <= y_bounds[1]
            and z_bounds[0] <= z_pos <= z_bounds[1]
        ):
            raise ValueError(f"robot.{position_key} {position} is outside safe_bounds")

    if rc["gripper_z_mm"] <= 0:
        raise ValueError(f"robot.gripper_z_mm must be > 0, got {rc['gripper_z_mm']}")

    cc = cfg.get("camera")
    if not cc:
        raise ValueError("Missing 'camera' section in config")
    for key in ("color_width", "color_height", "depth_width", "depth_height", "fps", "euler_eef_to_color"):
        if key not in cc:
            raise ValueError(f"Missing camera.{key} in config")
    if len(cc["euler_eef_to_color"]) != 6:
        raise ValueError("camera.euler_eef_to_color must have 6 elements [x,y,z,r,p,y]")

    detection_cfg = cfg.get("detection", {})
    backend = detection_cfg.get("backend", "llm")

    if backend == "yolo":
        yolo_cfg = detection_cfg.get("yolo", {})
        model_path = yolo_cfg.get("model_path", "")
        if not model_path:
            raise ValueError("Missing detection.yolo.model_path in config")
    else:
        lc = cfg.get("llm")
        if not lc:
            raise ValueError("Missing 'llm' section in config")
        for key in ("openrouter_url", "model"):
            if key not in lc:
                raise ValueError(f"Missing llm.{key} in config")

    dc = cfg.get("diagnostics")
    if dc:
        if "output_dir" in dc and not isinstance(dc["output_dir"], str):
            raise ValueError("diagnostics.output_dir must be a string path")
        for key in ("enabled", "save_color", "save_depth", "save_llm_raw", "save_assessments"):
            if key in dc and not isinstance(dc[key], bool):
                raise ValueError(f"diagnostics.{key} must be a boolean")


def main():
    parser = argparse.ArgumentParser(description="Vision Pick: LLM-guided robotic grasping")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Compute coordinates without moving robot")
    parser.add_argument("--loop", action="store_true", help="Continuously run pick and place")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: choose which object to pick")
    parser.add_argument("--detector", choices=["yolo", "llm"], default=None,
                        help="Override detection backend (yolo or llm)")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity (-v for INFO, -vv for DEBUG)")
    args = parser.parse_args()

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    # CLI --detector overrides config file
    if args.detector:
        cfg.setdefault("detection", {})["backend"] = args.detector

    try:
        validate_config(cfg)
    except ValueError as exc:
        logger.error("Config validation failed: %s", exc)
        sys.exit(1)

    with VisionPickSystem(cfg, dry_run=args.dry_run) as system:
        if args.interactive:
            run_interactive(system)
        elif args.loop:
            run_loop(system)
        else:
            run_pick_all(system)


if __name__ == "__main__":
    main()
