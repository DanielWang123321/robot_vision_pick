from __future__ import annotations

import math
from dataclasses import replace

from coord_transform import lookup_depth
from pick_models import DetectedObject, ScanResult, TargetAssessment


def _normalize_name(name: str) -> str:
    return "".join(name.lower().split())


def _bbox_iou(first_bbox, second_bbox):
    if first_bbox is None or second_bbox is None:
        return 0.0

    inter_x1 = max(first_bbox.x1, second_bbox.x1)
    inter_y1 = max(first_bbox.y1, second_bbox.y1)
    inter_x2 = min(first_bbox.x2, second_bbox.x2)
    inter_y2 = min(first_bbox.y2, second_bbox.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    union = first_bbox.area + second_bbox.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class TargetTracker:
    """Assign stable target IDs across repeated scans from the same detect pose."""

    def __init__(self, max_match_distance_px=220, name_mismatch_penalty_px=60):
        self.max_match_distance_px = max_match_distance_px
        self.name_mismatch_penalty_px = name_mismatch_penalty_px
        self._next_target_index = 1
        self._previous_objects: list[DetectedObject] = []

    def assign_ids(self, objects: list[DetectedObject]) -> list[DetectedObject]:
        if not objects:
            self._previous_objects = []
            return []

        assigned_target_ids: dict[int, str] = {}
        remaining_current = set(range(len(objects)))
        remaining_previous = set(range(len(self._previous_objects)))
        candidate_matches: list[tuple[float, int, int]] = []

        for current_index, current_object in enumerate(objects):
            for previous_index, previous_object in enumerate(self._previous_objects):
                dist_px = math.hypot(current_object.px - previous_object.px, current_object.py - previous_object.py)
                if dist_px > self.max_match_distance_px:
                    continue
                cost = dist_px
                if _normalize_name(current_object.tracking_name) != _normalize_name(previous_object.tracking_name):
                    cost += self.name_mismatch_penalty_px
                cost -= _bbox_iou(current_object.bbox, previous_object.bbox) * 100.0
                candidate_matches.append((cost, current_index, previous_index))

        for _cost, current_index, previous_index in sorted(candidate_matches, key=lambda item: item[0]):
            if current_index not in remaining_current or previous_index not in remaining_previous:
                continue
            previous_object = self._previous_objects[previous_index]
            assigned_target_ids[current_index] = previous_object.display_id
            remaining_current.remove(current_index)
            remaining_previous.remove(previous_index)

        tracked_objects = []
        for current_index, current_object in enumerate(objects):
            target_id = assigned_target_ids.get(current_index)
            if target_id is None:
                target_id = f"T{self._next_target_index:03d}"
                self._next_target_index += 1
            tracked_objects.append(replace(current_object, target_id=target_id))

        self._previous_objects = tracked_objects
        return tracked_objects


def assess_targets(scan_result: ScanResult) -> list[TargetAssessment]:
    """Score all detected objects to prioritize grasp order in multi-object scenes."""
    objects = scan_result.objects
    if not objects:
        return []

    image_h, image_w = scan_result.images.color.shape[:2]
    img_cx = image_w / 2
    img_cy = image_h / 2
    max_center_dist = math.hypot(img_cx, img_cy)
    ideal_border_margin = min(image_w, image_h) * 0.20
    ideal_neighbor_dist = min(image_w, image_h) * 0.25

    assessments = []
    for detected_object in objects:
        confidence = max(0.0, min(detected_object.confidence, 1.0))
        confidence_score = confidence * 45.0

        if detected_object.graspable:
            graspability_score = 12.0
        else:
            graspability_score = -28.0

        border_margin = min(
            detected_object.px,
            detected_object.py,
            image_w - 1 - detected_object.px,
            image_h - 1 - detected_object.py,
        )
        border_score = max(0.0, min(border_margin / ideal_border_margin, 1.0)) * 20.0

        center_dist = math.hypot(detected_object.px - img_cx, detected_object.py - img_cy)
        center_score = max(0.0, 1.0 - center_dist / max_center_dist) * 15.0

        neighbor_distances = [
            math.hypot(detected_object.px - other.px, detected_object.py - other.py)
            for other in objects
            if other is not detected_object
        ]
        if neighbor_distances:
            nearest_neighbor = min(neighbor_distances)
            isolation_score = max(0.0, min(nearest_neighbor / ideal_neighbor_dist, 1.0)) * 10.0
        else:
            nearest_neighbor = ideal_neighbor_dist
            isolation_score = 10.0

        bbox = detected_object.bbox
        if bbox is None:
            bbox_score = -6.0
            bbox_area_ratio = 0.0
        else:
            bbox_area_ratio = bbox.area / float(image_w * image_h)
            if bbox_area_ratio < 0.002:
                bbox_score = -8.0
            elif bbox_area_ratio < 0.01:
                bbox_score = 3.0
            elif bbox_area_ratio < 0.18:
                bbox_score = 8.0
            else:
                bbox_score = -6.0

        depth_m = lookup_depth(scan_result.images.depth, detected_object.px, detected_object.py, radius=5)
        if depth_m is None:
            depth_score = -20.0
        else:
            depth_score = 10.0

        total_score = (
            confidence_score
            + graspability_score
            + border_score
            + center_score
            + isolation_score
            + depth_score
            + bbox_score
        )

        reasons = (
            f"category={detected_object.category}",
            f"confidence={detected_object.confidence:.2f}",
            f"graspable={'yes' if detected_object.graspable else 'no'}",
            f"bbox_area_ratio={bbox_area_ratio:.3f}",
            f"border_margin={border_margin:.0f}px",
            f"center_offset={center_dist:.0f}px",
            f"nearest_neighbor={nearest_neighbor:.0f}px",
            "depth=ok" if depth_m is not None else "depth=missing",
        )
        assessments.append(
            TargetAssessment(
                detected_object=detected_object,
                total_score=total_score,
                confidence_score=confidence_score,
                graspability_score=graspability_score,
                border_score=border_score,
                center_score=center_score,
                isolation_score=isolation_score,
                depth_score=depth_score,
                bbox_score=bbox_score,
                reasons=reasons,
            )
        )

    return sorted(assessments, key=lambda assessment: assessment.total_score, reverse=True)
