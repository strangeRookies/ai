from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence


def tracking_debug_enabled() -> bool:
    return os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}


def build_tracker_startup_record(
    camera_login_id: str,
    tracker_backend: str,
    postprocessor,
    args,
) -> dict:
    diagnostics = _safe_diagnostics(postprocessor)
    constructor = diagnostics.get("bytetrack_constructor") or {}
    return {
        "stage": "tracker_startup",
        "cameraLoginId": camera_login_id,
        "trackerBackend": tracker_backend,
        "trackerClass": postprocessor.__class__.__name__,
        "trackerObjectId": id(postprocessor),
        "trackBuffer": int(getattr(args, "track_buffer", 0)),
        "lostTrackBuffer": int(getattr(args, "track_buffer", 0)),
        "matchThreshold": float(getattr(args, "match_thresh", 0.0)),
        "highTrackThreshold": float(getattr(args, "track_thresh", 0.0)),
        "lowTrackThreshold": float(getattr(args, "track_thresh", 0.0)),
        "newTrackThreshold": float(getattr(args, "track_thresh", 0.0)),
        "assumedFps": int(getattr(args, "frame_rate", 0)),
        "detectorConfidence": float(getattr(args, "detector_conf", 0.0)),
        "detectorBackend": _detector_backend(str(getattr(args, "yolo_model", ""))),
        "bytetrackConstructorUsed": dict(constructor.get("used") or {}),
        "bytetrackConstructorIgnored": dict(constructor.get("ignored") or {}),
        "bytetrackConstructorSupportedParameters": list(constructor.get("supported_parameters") or []),
    }


def log_tracker_startup(camera_login_id: str, tracker_backend: str, postprocessor, args) -> None:
    record = build_tracker_startup_record(camera_login_id, tracker_backend, postprocessor, args)
    print(f"[tracker-startup] {json.dumps(record, ensure_ascii=False)}", flush=True)


def build_frame_tracking_record(
    camera_login_id: str,
    frame_id: int | None,
    captured_at_ms: int | None,
    processed_at_ms: int | None,
    frame_gap: int | None,
    dropped_frame_count: int,
    raw_detections: Sequence[Mapping[str, object]],
    tracked_detections: Sequence[Mapping[str, object]],
    tracker_diagnostics: Mapping[str, object],
    tracker_object_id: int,
) -> dict:
    return {
        "stage": "tracking",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "capturedAtMs": captured_at_ms,
        "processedAtMs": processed_at_ms,
        "frameGap": frame_gap,
        "droppedFrameCount": int(dropped_frame_count),
        "rawDetectionCount": len(raw_detections),
        "detectionCountAfterConfidenceFilter": len(tracked_detections),
        "avgDetectionConfidence": _average_field(tracked_detections, "confidence"),
        "avgKeypointConfidence": _average_field(tracked_detections, "keypoint_confidence"),
        "activeTrackIds": _track_ids(tracked_detections),
        "newTrackCount": int(tracker_diagnostics.get("new_tracks", 0) or 0),
        "lostTrackCount": int(tracker_diagnostics.get("lost_tracks", 0) or 0),
        "removedTrackIds": list(tracker_diagnostics.get("removed_track_ids") or []),
        "trackLifetimeFrames": _track_lifetimes(tracker_diagnostics),
        "trackerObjectId": int(tracker_object_id),
    }


def log_frame_tracking_debug(record: Mapping[str, object]) -> None:
    if not tracking_debug_enabled():
        return
    print(f"[tracking-debug] {json.dumps(dict(record), ensure_ascii=False)}", flush=True)


def build_tracker_reset_record(
    camera_login_id: str,
    frame_id: int | None,
    frame_gap: int | None,
    tracker_object_id: int,
    reset: bool,
    reason: str,
    stream_run_id: str | None = None,
) -> dict:
    return {
        "stage": "tracker_reset_decision",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "frameGap": frame_gap,
        "trackerObjectId": int(tracker_object_id),
        "reset": bool(reset),
        "reason": reason,
        "streamRunId": stream_run_id,
    }


def log_tracker_reset_decision(record: Mapping[str, object]) -> None:
    print(f"[tracker-reset] {json.dumps(dict(record), ensure_ascii=False)}", flush=True)


def _safe_diagnostics(postprocessor) -> Mapping[str, object]:
    diagnostics = getattr(postprocessor, "diagnostics", None)
    if diagnostics is None:
        return {}
    return diagnostics()


def _detector_backend(model_path: str) -> str:
    return "tensorrt" if model_path.lower().endswith(".engine") else "torch"


def _average_field(detections: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [
        float(item[field])
        for item in detections
        if item.get(field) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _track_ids(detections: Sequence[Mapping[str, object]]) -> list[int]:
    return sorted(
        int(item["track_id"])
        for item in detections
        if item.get("track_id") is not None
    )


def _track_lifetimes(tracker_diagnostics: Mapping[str, object]) -> dict[str, int]:
    tracks = tracker_diagnostics.get("tracks")
    if not isinstance(tracks, Mapping):
        return {}
    output: dict[str, int] = {}
    for track_id, value in tracks.items():
        if not isinstance(value, Mapping):
            continue
        age = value.get("track_age", value.get("age", 0))
        output[str(track_id)] = int(age or 0)
    return output


def log_sequence_stage(
    camera_login_id: str,
    frame_id: int | None,
    active_track_ids: Sequence[int],
    buffer_lengths: Mapping[int, int],
    sequences_generated: int,
    sequences_generated_by_track: Mapping[int, int],
    latest_faint_prob: float | None,
    sequence_diagnostics: Mapping[int, Mapping[str, object]] | None = None,
    checkpoint_input_size: int | None = None,
    runtime_feature_dim: int | None = None,
    tensor_shape: Sequence[int] | None = None,
    sequence_length: int | None = None,
) -> None:
    if not tracking_debug_enabled():
        return
    diagnostics = {
        str(track_id): dict(value)
        for track_id, value in (sequence_diagnostics or {}).items()
    }
    reasons = {str(value.get("reason", "")) for value in diagnostics.values()}
    if sequences_generated > 0:
        diagnosis = "sequence_ready"
    elif "feature_dim_mismatch" in reasons:
        diagnosis = "feature_dim_mismatch"
    elif "insufficient_keypoints" in reasons:
        diagnosis = "insufficient_keypoints"
    elif "selected_track_missing" in reasons:
        diagnosis = "selected_track_missing"
    elif "buffer_not_full" in reasons:
        diagnosis = "sequence_buffer_not_full"
    elif active_track_ids:
        diagnosis = "sequence_buffer_no_reason"
    else:
        diagnosis = "no_active_tracks"
    record = {
        "stage": "sequence",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "activeTrackIds": list(active_track_ids),
        "bufferLengths": {str(track_id): int(length) for track_id, length in buffer_lengths.items()},
        "sequencesGenerated": int(sequences_generated),
        "lstmSequenceGenerated": sequences_generated > 0,
        "sequencesGeneratedByTrack": {
            str(track_id): int(count) for track_id, count in sequences_generated_by_track.items()
        },
        "latestFaintProbability": latest_faint_prob,
        "trackDiagnostics": diagnostics,
        "diagnosis": diagnosis,
        "checkpointInputSize": checkpoint_input_size,
        "runtimeFeatureDim": runtime_feature_dim,
        "tensorShape": list(tensor_shape) if tensor_shape is not None else None,
        "sequenceLength": sequence_length,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)
