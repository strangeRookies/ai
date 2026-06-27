from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Detection: TypeAlias = dict[str, JsonValue]
FrameShape: TypeAlias = tuple[int, int] | tuple[int, int, int] | list[int]


@dataclass(frozen=True, slots=True)
class CheapFilterConfig:
    enabled: bool = False
    slope_ratio_threshold: float = 1.3
    min_avg_keypoint_confidence: float = 0.25
    min_bbox_area_ratio: float = 0.005
    min_center_drop_ratio: float = 0.03
    min_aspect_ratio_growth: float = 0.20
    min_risk_score: float = 1.0


@dataclass(frozen=True, slots=True)
class CheapFilterDecision:
    keep: bool
    risk_score: float
    reasons: tuple[str, ...]


def evaluate_sequence_candidate(sequence: dict[str, JsonValue], config: CheapFilterConfig) -> CheapFilterDecision:
    if not config.enabled:
        return CheapFilterDecision(keep=True, risk_score=0.0, reasons=("disabled",))

    detections = _as_detections(sequence.get("detections"))
    if not detections:
        return CheapFilterDecision(keep=True, risk_score=0.0, reasons=("no_detections_passthrough",))

    frame_shapes = _as_frame_shapes(sequence.get("frame_shapes"))
    reasons: list[str] = []
    score = 0.0

    if _has_reliable_keypoints(detections, config.min_avg_keypoint_confidence):
        score += 0.25
        reasons.append("keypoint_confidence")

    if _has_large_enough_bbox(detections, frame_shapes, config.min_bbox_area_ratio):
        score += 0.25
        reasons.append("bbox_size")

    slope_frames = _slope_risk_frames(detections, config.slope_ratio_threshold)
    if slope_frames > 0:
        score += 0.5
        reasons.append("slope_ratio_1.3")

    if slope_frames >= 2:
        score += 0.25
        reasons.append("risk_persistence")

    if _center_drops(detections, frame_shapes, config.min_center_drop_ratio):
        score += 0.25
        reasons.append("center_drop")

    if _aspect_ratio_grows(detections, config.min_aspect_ratio_growth):
        score += 0.25
        reasons.append("bbox_ratio_change")

    keep = score >= config.min_risk_score
    return CheapFilterDecision(keep=keep, risk_score=round(score, 6), reasons=tuple(reasons or ["low_risk"]))


def _has_reliable_keypoints(detections: list[Detection], threshold: float) -> bool:
    confidences = [
        float(point.get("confidence", 0.0))
        for detection in detections
        for point in _as_keypoints(detection.get("keypoints"))
    ]
    if not confidences:
        return True
    return sum(confidences) / len(confidences) >= float(threshold)


def _has_large_enough_bbox(detections: list[Detection], frame_shapes: list[FrameShape], threshold: float) -> bool:
    for index, detection in enumerate(detections):
        bbox = _as_float_list(detection.get("bbox"))
        if len(bbox) < 4:
            continue
        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)
        frame_area = _frame_area(frame_shapes[index] if index < len(frame_shapes) else None, bbox)
        if frame_area > 0.0 and (width * height) / frame_area >= float(threshold):
            return True
    return False


def _slope_risk_frames(detections: list[Detection], threshold: float) -> int:
    return sum(1 for detection in detections if _torso_ratio(detection) >= float(threshold))


def _torso_ratio(detection: Detection) -> float:
    keypoints = _as_keypoints(detection.get("keypoints"))
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if any(index >= len(keypoints) or keypoints[index] is None for index in required):
        return 0.0
    shoulder = _midpoint(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER])
    hip = _midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])
    dx = abs(shoulder[0] - hip[0])
    dy = abs(shoulder[1] - hip[1])
    return dx / max(dy, 1e-6)


def _center_drops(detections: list[Detection], frame_shapes: list[FrameShape], threshold: float) -> bool:
    first = _bbox_center_y_ratio(detections[0], frame_shapes[0] if frame_shapes else None)
    last_shape = frame_shapes[-1] if len(frame_shapes) >= len(detections) else None
    last = _bbox_center_y_ratio(detections[-1], last_shape)
    if first is None or last is None:
        return False
    return last - first >= float(threshold)


def _aspect_ratio_grows(detections: list[Detection], threshold: float) -> bool:
    first = _bbox_aspect_ratio(detections[0])
    last = _bbox_aspect_ratio(detections[-1])
    if first is None or last is None:
        return False
    return last - first >= float(threshold)


def _bbox_center_y_ratio(detection: Detection, frame_shape: FrameShape | None) -> float | None:
    bbox = _as_float_list(detection.get("bbox"))
    if len(bbox) < 4:
        return None
    height = _frame_height(frame_shape, bbox)
    if height <= 0.0:
        return None
    return ((float(bbox[1]) + float(bbox[3])) / 2.0) / height


def _bbox_aspect_ratio(detection: Detection) -> float | None:
    bbox = _as_float_list(detection.get("bbox"))
    if len(bbox) < 4:
        return None
    width = max(float(bbox[2]) - float(bbox[0]), 0.0)
    height = max(float(bbox[3]) - float(bbox[1]), 1e-6)
    return width / height


def _frame_area(frame_shape: FrameShape | None, bbox: list[float]) -> float:
    height = _frame_height(frame_shape, bbox)
    width = _frame_width(frame_shape, bbox)
    return width * height


def _frame_height(frame_shape: FrameShape | None, bbox: list[float]) -> float:
    if frame_shape is not None and len(frame_shape) >= 2:
        return float(frame_shape[0])
    return max(float(bbox[3]), 1.0)


def _frame_width(frame_shape: FrameShape | None, bbox: list[float]) -> float:
    if frame_shape is not None and len(frame_shape) >= 2:
        return float(frame_shape[1])
    return max(float(bbox[2]), 1.0)


def _midpoint(first: Detection, second: Detection) -> tuple[float, float]:
    return (
        (float(first.get("x", 0.0)) + float(second.get("x", 0.0))) / 2.0,
        (float(first.get("y", 0.0)) + float(second.get("y", 0.0))) / 2.0,
    )


def _as_detections(value: JsonValue | None) -> list[Detection]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_frame_shapes(value: JsonValue | None) -> list[FrameShape]:
    if not isinstance(value, list):
        return []
    shapes: list[FrameShape] = []
    for item in value:
        if isinstance(item, tuple) and len(item) in {2, 3}:
            shapes.append(item)
        elif isinstance(item, list) and len(item) in {2, 3} and all(isinstance(part, int) for part in item):
            shapes.append(item)
    return shapes


def _as_keypoints(value: JsonValue | None) -> list[Detection]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_float_list(value: JsonValue | None) -> list[float]:
    if not isinstance(value, list):
        return []
    numbers: list[float] = []
    for item in value:
        if isinstance(item, int | float):
            numbers.append(float(item))
    return numbers
