from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("true", "yes", "y", "1"):
        return True
    if text in ("false", "no", "n", "0"):
        return False
    return default


@dataclass(slots=True, frozen=True)
class ImageCapture:
    depth: np.ndarray
    color: np.ndarray


@dataclass(slots=True, frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(slots=True, frozen=True)
class DetectedObject:
    name: str
    px: int
    py: int
    confidence: float = 0.0
    canonical_name: str | None = None
    category: str = "unknown"
    graspable: bool = True
    grasp_reason: str | None = None
    bbox: BoundingBox | None = None
    target_id: str | None = None
    robot_xy_mm: tuple[float, float] | None = None

    @classmethod
    def from_llm_result(cls, result: dict[str, Any]) -> "DetectedObject":
        bbox_data = result.get("bbox")
        bbox = None
        if isinstance(bbox_data, dict):
            try:
                bbox = BoundingBox(
                    x1=int(bbox_data["x1"]),
                    y1=int(bbox_data["y1"]),
                    x2=int(bbox_data["x2"]),
                    y2=int(bbox_data["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                bbox = None
        robot_xy = result.get("robot_xy_mm")
        if robot_xy is not None:
            robot_xy = (float(robot_xy[0]), float(robot_xy[1]))
        return cls(
            name=str(result.get("name", "unknown")),
            px=int(result["px"]),
            py=int(result["py"]),
            confidence=float(result.get("confidence", 0.0)),
            canonical_name=None if result.get("canonical_name") is None else str(result.get("canonical_name")),
            category=str(result.get("category", "unknown")),
            graspable=_to_bool(result.get("graspable", True), default=True),
            grasp_reason=None if result.get("grasp_reason") is None else str(result.get("grasp_reason")),
            bbox=bbox,
            robot_xy_mm=robot_xy,
        )

    @property
    def display_id(self) -> str:
        return self.target_id or "untracked"

    @property
    def tracking_name(self) -> str:
        if self.canonical_name:
            return self.canonical_name
        return self.name


@dataclass(slots=True, frozen=True)
class ScanResult:
    objects: list[DetectedObject]
    images: ImageCapture
    scan_id: str | None = None
    llm_raw_response: str | None = None

    @classmethod
    def from_llm_results(
        cls,
        results: list[dict[str, Any]],
        depth: np.ndarray,
        color: np.ndarray,
        llm_raw_response: str | None = None,
    ) -> "ScanResult":
        objects = [DetectedObject.from_llm_result(result) for result in results]
        return cls(objects=objects, images=ImageCapture(depth=depth, color=color), llm_raw_response=llm_raw_response)


@dataclass(slots=True, frozen=True)
class RefinedDetection:
    cx: int
    cy: int
    width: float
    height: float
    angle_deg: float
    aspect_ratio: float
    area: float
    obj_depth: float | None
    table_depth: float | None
    color_score: float | None = None

    @classmethod
    def from_cv_result(cls, result: dict[str, Any] | None) -> "RefinedDetection | None":
        if result is None:
            return None
        return cls(
            cx=int(result["cx"]),
            cy=int(result["cy"]),
            width=float(result["width"]),
            height=float(result["height"]),
            angle_deg=float(result["angle_deg"]),
            aspect_ratio=float(result["aspect_ratio"]),
            area=float(result["area"]),
            obj_depth=None if result.get("obj_depth") is None else float(result["obj_depth"]),
            table_depth=None if result.get("table_depth") is None else float(result["table_depth"]),
            color_score=None if result.get("color_score") is None else float(result["color_score"]),
        )


@dataclass(slots=True, frozen=True)
class VisualServoResult:
    target_x_mm: float
    target_y_mm: float
    detection: RefinedDetection | None
    images: ImageCapture


@dataclass(slots=True, frozen=True)
class GraspPlan:
    detected_object: DetectedObject
    refined_detection: RefinedDetection
    rough_x_mm: float
    rough_y_mm: float
    close_detect_z_mm: float
    grasp_x_mm: float
    grasp_y_mm: float
    grasp_z_mm: float
    approach_z_mm: float
    gripper_yaw_deg: float
    long_dim_mm: float
    short_dim_mm: float
    estimated_height_mm: float = 0.0


@dataclass(slots=True, frozen=True)
class PickResult:
    success: bool
    reason: str
    detected_object: DetectedObject | None = None
    grasp_plan: GraspPlan | None = None


@dataclass(slots=True, frozen=True)
class TargetAssessment:
    detected_object: DetectedObject
    total_score: float
    confidence_score: float
    graspability_score: float
    border_score: float
    center_score: float
    isolation_score: float
    depth_score: float
    bbox_score: float
    reasons: tuple[str, ...]


class CameraFrameError(RuntimeError):
    pass


class RobotCommandError(RuntimeError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
