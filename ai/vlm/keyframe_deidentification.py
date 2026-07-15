"""Keyframe face de-identification for the VLM Gemini gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ai.events.clip_worker import (
    _face_box_from_keypoints,
    _upper_body_fallback_box,
)
from ai.vlm.deidentification_contracts import (
    DeidentificationFrameReport,
    DeidentificationOutcome,
)
from ai.vlm.keyframe_extractor import ExtractedKeyframe

_MAX_POSE_DISTANCE_SEC = 1.0
_MOSAIC_TILE = 15
_TOP_BAND_RATIO = 0.15


class KeyframeDeidentificationError(RuntimeError):
    """Raised when keyframe de-identification cannot complete safely."""


@dataclass(frozen=True, slots=True)
class _PoseObservation:
    frame_index: int | None
    timestamp_sec: float | None
    bbox: tuple[float, float, float, float]
    keypoints: list[dict[str, float]]


def deidentify_keyframes(
    frames: tuple[ExtractedKeyframe, ...],
    metadata: Mapping[str, object] | None = None,
) -> DeidentificationOutcome:
    if not frames:
        raise KeyframeDeidentificationError("keyframe batch must not be empty")

    observations = _load_pose_observations(metadata or {})
    processed_frames: list[ExtractedKeyframe] = []
    reports: list[DeidentificationFrameReport] = []

    for frame in frames:
        masked, detected_count, deidentified_count = _mask_single_frame(frame, observations)
        processed_frames.append(masked)
        reports.append(
            DeidentificationFrameReport(
                index=frame.index,
                status="PASS",
                detected_person_count=detected_count,
                deidentified_person_count=deidentified_count,
            )
        )

    return DeidentificationOutcome(
        frames=tuple(processed_frames),
        reports=tuple(reports),
    )


def build_default_deidentify_frames(
    metadata: Mapping[str, object],
) -> Callable[[tuple[ExtractedKeyframe, ...]], DeidentificationOutcome]:
    def _deidentify(frames: tuple[ExtractedKeyframe, ...]) -> DeidentificationOutcome:
        return deidentify_keyframes(frames, metadata)

    return _deidentify


def _mask_single_frame(
    frame: ExtractedKeyframe,
    observations: Sequence[_PoseObservation],
) -> tuple[ExtractedKeyframe, int, int]:
    image = _decode_jpeg(frame.jpeg_bytes)
    height, width = image.shape[:2]
    matches = _match_observations(frame, observations)
    detected_count = len(matches)
    deidentified_count = 0

    for observation in matches:
        if _apply_masks(image, observation, width, height):
            deidentified_count += 1

    if detected_count == 0:
        return frame, 0, 0

    payload = _encode_jpeg(image)
    return (
        ExtractedKeyframe(
            index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            frame_index=frame.frame_index,
            width=frame.width,
            height=frame.height,
            sha256=hashlib.sha256(payload).hexdigest(),
            jpeg_bytes=payload,
        ),
        detected_count,
        deidentified_count,
    )


def _apply_masks(
    image: np.ndarray,
    observation: _PoseObservation,
    width: int,
    height: int,
) -> bool:
    box = {
        "x1": observation.bbox[0],
        "y1": observation.bbox[1],
        "x2": observation.bbox[2],
        "y2": observation.bbox[3],
        "keypoints": observation.keypoints,
    }
    region = _face_box_from_keypoints(box, width, height)
    if region is not None:
        _apply_solid_mask(image, region)
        return True

    region = _upper_body_fallback_box(box, width, height, ratio=_TOP_BAND_RATIO)
    if region is not None:
        _apply_mosaic(image, region)
        return True

    haar_region = _detect_haar_face(image, observation.bbox)
    if haar_region is not None:
        _apply_solid_mask(image, haar_region)
        return True

    fallback = _top_band_region(observation.bbox, width, height)
    if fallback is not None:
        _apply_mosaic(image, fallback)
        return True
    return False


def _decode_jpeg(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise KeyframeDeidentificationError("failed to decode keyframe JPEG")
    return image


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok or encoded is None or encoded.size == 0:
        raise KeyframeDeidentificationError("failed to encode de-identified keyframe")
    return encoded.tobytes()


def _apply_solid_mask(image: np.ndarray, region: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = region
    image[y1:y2, x1:x2] = 0


def _apply_mosaic(image: np.ndarray, region: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small = cv2.resize(
        roi,
        (
            max(1, (x2 - x1) // _MOSAIC_TILE),
            max(1, (y2 - y1) // _MOSAIC_TILE),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    image[y1:y2, x1:x2] = cv2.resize(
        small,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST,
    )


def _top_band_region(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    band_height = max(y2 - y1, 1.0) * _TOP_BAND_RATIO
    x1_px = int(max(0, min(x1, width)))
    x2_px = int(max(0, min(x2, width)))
    y1_px = int(max(0, min(y1, height)))
    y2_px = int(max(0, min(y1 + band_height, height)))
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    return x1_px, y1_px, x2_px, y2_px


def _detect_haar_face(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1_px = int(max(0, math.floor(x1)))
    y1_px = int(max(0, math.floor(y1)))
    x2_px = int(min(image.shape[1], math.ceil(x2)))
    y2_px = int(min(image.shape[0], math.ceil(y2)))
    if x2_px <= x1_px or y2_px <= y1_px:
        return None

    roi = image[y1_px:y2_px, x1_px:x2_px]
    if roi.size == 0:
        return None

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda item: item[2] * item[3])
    return (
        x1_px + int(fx),
        y1_px + int(fy),
        x1_px + int(fx + fw),
        y1_px + int(fy + fh),
    )


def _match_observations(
    frame: ExtractedKeyframe,
    observations: Sequence[_PoseObservation],
) -> list[_PoseObservation]:
    if not observations:
        return []

    ranked: list[tuple[float, _PoseObservation]] = []
    for observation in observations:
        distance = _observation_distance(frame, observation)
        if distance is None or distance > _MAX_POSE_DISTANCE_SEC:
            continue
        ranked.append((distance, observation))

    if not ranked:
        return []

    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked]


def _observation_distance(
    frame: ExtractedKeyframe,
    observation: _PoseObservation,
) -> float | None:
    if observation.frame_index is not None:
        return abs(float(frame.frame_index - observation.frame_index)) / 30.0
    if observation.timestamp_sec is not None:
        return abs(frame.timestamp_sec - observation.timestamp_sec)
    return None


def _load_pose_observations(metadata: Mapping[str, object]) -> list[_PoseObservation]:
    observations: list[_PoseObservation] = []
    for source in (
        metadata.get("pose_history"),
        metadata.get("keypoint_history"),
        metadata.get("keypoint_data"),
        metadata.get("bounding_box_data"),
    ):
        observations.extend(_parse_pose_source(source))
    return observations


def _parse_pose_source(value: object) -> list[_PoseObservation]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_pose_source_message(stripped)
        return _parse_pose_source(parsed)
    if isinstance(value, Mapping):
        if "bbox" in value or "bbox_xyxy" in value or "keypoints" in value:
            single = _parse_pose_record(value)
            return [single] if single is not None else []
        nested = value.get("history") or value.get("frames") or value.get("observations")
        if isinstance(nested, list):
            return [item for record in nested if (item := _parse_pose_record(record)) is not None]
        return []
    if isinstance(value, list):
        return [item for record in value if (item := _parse_pose_record(record)) is not None]
    return []


def _parse_pose_source_message(message: str) -> list[_PoseObservation]:
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return []
    return _parse_pose_source(parsed)


def _parse_pose_record(record: object) -> _PoseObservation | None:
    if not isinstance(record, Mapping):
        return None

    bbox = _parse_bbox(record)
    if bbox is None:
        return None

    keypoints = _parse_keypoints(record.get("keypoints"))
    frame_index = _optional_int(record.get("frame_index"))
    timestamp_sec = _optional_float(record.get("timestamp_sec") or record.get("timestamp"))
    return _PoseObservation(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        bbox=bbox,
        keypoints=keypoints,
    )


def _parse_bbox(record: Mapping[str, object]) -> tuple[float, float, float, float] | None:
    raw = record.get("bbox_xyxy") or record.get("bbox")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return tuple(float(raw[index]) for index in range(4))
    parts = [record.get("x1"), record.get("y1"), record.get("x2"), record.get("y2")]
    if any(part is None for part in parts):
        return None
    return tuple(float(part) for part in parts)


def _parse_keypoints(value: object) -> list[dict[str, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    keypoints: list[dict[str, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            x = _optional_float(item.get("x"))
            y = _optional_float(item.get("y"))
            confidence = _optional_float(item.get("confidence") or item.get("conf")) or 0.0
            if x is None or y is None:
                continue
            keypoints.append({"x": x, "y": y, "confidence": confidence})
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            x = _optional_float(item[0])
            y = _optional_float(item[1])
            confidence = _optional_float(item[2]) if len(item) >= 3 else 0.0
            if x is None or y is None:
                continue
            keypoints.append({"x": x, "y": y, "confidence": confidence or 0.0})
    return keypoints


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number