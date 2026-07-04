from __future__ import annotations

from typing import Final


KEYPOINT_COUNT: Final = 17
DEFAULT_MIN_KEYPOINT_CONFIDENCE: Final = 0.25


def build_pose_diagnostic_record(
    camera_login_id: str,
    source_url: str,
    assigned_video_path: str | None,
    frame_id: int | None,
    timestamp_ms: int | None,
    raw_detections: list[dict],
    tracker_diagnostics: dict,
    sequence_ready_count: int,
    min_keypoint_confidence: float = DEFAULT_MIN_KEYPOINT_CONFIDENCE,
    sequence_diagnostics: dict | None = None,
) -> dict:
    detection_metrics = [_detection_metric(detection, min_keypoint_confidence) for detection in raw_detections]
    active_tracks = int(tracker_diagnostics.get("active_tracks", 0))
    avg_keypoint_confidence = _avg(
        metric["avg_keypoint_confidence"]
        for metric in detection_metrics
        if metric["avg_keypoint_confidence"] is not None
    )
    record = {
        "stage": "pose_frame",
        "cameraLoginId": camera_login_id,
        "sourceUrl": source_url,
        "assignedVideoPath": assigned_video_path,
        "frameId": frame_id,
        "timestampMs": timestamp_ms,
        "raw_detection_count": len(raw_detections),
        "detections": detection_metrics,
        "bbox": detection_metrics[0]["bbox"] if detection_metrics else None,
        "bbox_confidence": detection_metrics[0]["bbox_confidence"] if detection_metrics else None,
        "avg_bbox_confidence": _avg(metric["bbox_confidence"] for metric in detection_metrics),
        "avg_keypoint_confidence": avg_keypoint_confidence,
        "valid_keypoint_count": sum(int(metric["valid_keypoint_count"]) for metric in detection_metrics),
        "missing_keypoint_count": sum(int(metric["missing_keypoint_count"]) for metric in detection_metrics),
        "track_id": None,
        "track_ids": _track_ids(tracker_diagnostics),
        "active_tracks": active_tracks,
        "sequenceReadyCount": int(sequence_ready_count),
        "id_switch_like_events": int(tracker_diagnostics.get("id_switch_like_events", 0)),
        "relink_success_count": int((sequence_diagnostics or {}).get("relink_success_count", 0)),
        "relink_fail_count": int((sequence_diagnostics or {}).get("relink_fail_count", 0)),
    }
    record["diagnosis"] = _diagnose(record, min_keypoint_confidence)
    return record


def minimal_jsonl_record(record: dict) -> dict:
    keys = [
        "cameraLoginId",
        "sourceUrl",
        "assignedVideoPath",
        "frameId",
        "timestampMs",
        "stage",
        "raw_detection_count",
        "avg_bbox_confidence",
        "avg_keypoint_confidence",
        "valid_keypoint_count",
        "active_tracks",
        "track_ids",
        "sequenceReadyCount",
        "diagnosis",
    ]
    return {key: record.get(key) for key in keys}


def bbox_for_diagnostic_image(detection: dict) -> list[float] | None:
    return _bbox(detection)


def _track_ids(tracker_diagnostics: dict) -> list[int]:
    explicit_ids = tracker_diagnostics.get("track_ids") or tracker_diagnostics.get("active_track_ids")
    if explicit_ids is not None:
        return sorted({int(track_id) for track_id in explicit_ids if track_id is not None})
    tracks = tracker_diagnostics.get("tracks") or {}
    ids: set[int] = set()
    if isinstance(tracks, dict):
        for key, value in tracks.items():
            track_id = value.get("track_id") if isinstance(value, dict) else key
            if track_id is not None:
                ids.add(int(track_id))
    return sorted(ids)


def _detection_metric(detection: dict, min_keypoint_confidence: float) -> dict:
    bbox = _bbox(detection)
    keypoint_confidences = [
        float(point.get("confidence", 0.0))
        for point in detection.get("keypoints") or []
        if point.get("confidence") is not None
    ]
    valid_keypoints = [value for value in keypoint_confidences if value >= min_keypoint_confidence]
    return {
        "bbox": bbox,
        "bbox_confidence": round(float(detection.get("confidence", 0.0)), 4),
        "avg_keypoint_confidence": _avg(keypoint_confidences),
        "valid_keypoint_count": len(valid_keypoints),
        "missing_keypoint_count": max(KEYPOINT_COUNT - len(valid_keypoints), 0),
        "bbox_width": _bbox_width(bbox),
        "bbox_height": _bbox_height(bbox),
        "track_id": detection.get("track_id"),
    }


def _diagnose(record: dict, min_keypoint_confidence: float) -> str:
    raw_count = int(record["raw_detection_count"])
    active_tracks = int(record["active_tracks"])
    avg_keypoint = record.get("avg_keypoint_confidence")
    if raw_count == 0 and active_tracks == 0:
        return "detector_person_missing"
    if raw_count > 0 and (avg_keypoint is None or float(avg_keypoint) < min_keypoint_confidence):
        return "pose_quality_low"
    if raw_count > 0 and active_tracks == 0:
        return "tracking_association_problem"
    if raw_count > 0 and active_tracks > 0 and int(record.get("sequenceReadyCount", 0)) == 0:
        return "sequence_buffer_not_ready"
    return "pose_tracking_ok"


def _bbox(detection: dict) -> list[float] | None:
    bbox = detection.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    return [round(float(value), 2) for value in bbox[:4]]


def _bbox_width(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return round(max(bbox[2] - bbox[0], 0.0), 2)


def _bbox_height(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return round(max(bbox[3] - bbox[1], 0.0), 2)


def _avg(values) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return round(sum(collected) / len(collected), 4)

