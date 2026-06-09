import json
import re
from pathlib import Path

from ai.action.faint_post_processing import (
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    faint_probability,
)


def normalize_detections(detections):
    boxes = []
    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        boxes.append(
            {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "score": float(detection.get("confidence", 0.0)),
                "class_name": "person",
                "keypoints": detection.get("keypoints"),
                "track_id": detection.get("track_id"),
                "raw_bbox": detection.get("raw_bbox") or bbox,
                "smoothed_bbox": detection.get("smoothed_bbox") or bbox,
                "track_age": detection.get("track_age"),
                "missing_frames": detection.get("missing_frames", 0),
                "track_confidence": detection.get("track_confidence", detection.get("confidence", 0.0)),
            }
        )
    return boxes


def mock_keypoints_for_bbox(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    points = []
    for idx in range(17):
        col = idx % 5
        row = idx // 5
        points.append(
            {
                "x": round(x1 + width * (0.2 + col * 0.15), 2),
                "y": round(y1 + height * (0.1 + row * 0.2), 2),
                "confidence": 0.9,
            }
        )
    return points


def ensure_mock_keypoints(detections):
    for detection in detections:
        if detection.get("keypoints"):
            continue
        bbox = detection.get("bbox")
        if bbox:
            detection["keypoints"] = mock_keypoints_for_bbox(bbox)
    return detections


def maybe_log_debug(packet, boxes, summary, prediction, args, prefix="[rtsp-inference-debug]"):
    every_n = max(0, int(getattr(args, "debug_every_n", 30)))
    missing_detection = len(boxes) == 0
    should_log = missing_detection or (every_n > 0 and summary["frames_processed"] % every_n == 0)
    if not should_log:
        return
    faint_prob = faint_probability(prediction)
    faint_text = "None" if faint_prob is None else f"{faint_prob:.4f}"
    print(
        f"{prefix} "
        f"frame={packet.frame_idx} "
        f"bbox={len(boxes)} "
        f"keypoints={summary.get('latest_frame_keypoints', 0)} "
        f"active_tracks={summary.get('active_tracks', 0)} "
        f"seq={summary['generated_sequences']} "
        f"pred={summary['lstm_predictions']} "
        f"latest_faint_prob={faint_text}",
        flush=True,
    )


def build_inference_event_payload(args, packet, prediction, boxes, sequence):
    bbox = sequence.get("bbox") if sequence else None
    track_id = sequence.get("track_id") if sequence else None
    payload = {
        "camera_id": args.camera_id,
        "timestamp": float(packet.timestamp),
        "event_type": prediction["label"],
        "severity": getattr(args, "event_severity", "HIGH"),
        "confidence": float(prediction["score"]),
        "bbox": bbox,
        "track_id": track_id,
    }
    clip_path = sequence.get("clip_path") if sequence else None
    clip_url = sequence.get("clip_url") if sequence else None
    # TODO: Event clip writing is asynchronous, so run_rtsp_inference cannot
    # attach the final saved MP4 path at trigger time without a larger callback
    # flow. Include clip_path/clip_url only when an upstream sequence already
    # provides one.
    if clip_path:
        payload["clip_path"] = clip_path
    if clip_url:
        payload["clip_url"] = clip_url
    return payload


def build_inference_event_log(args, packet, prediction, boxes, sequence):
    bbox = sequence.get("bbox") if sequence else None
    track_id = sequence.get("track_id") if sequence else None
    return {
        "camera_id": args.camera_id,
        "frame_idx": int(packet.frame_idx),
        "timestamp": float(packet.timestamp),
        "event_type": prediction["label"],
        "confidence": float(prediction["score"]),
        "bbox": bbox,
        "track_id": track_id,
        "sequence_window": {"start": sequence["start_frame"], "end": sequence["end_frame"]} if sequence else None,
        "probabilities": prediction.get("probabilities", {}),
        "threshold": getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD),
        "post_processing": {
            "min_consecutive_faint": getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
            "camera_cooldown_seconds": getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
        },
    }


def save_inference_event_log(event_log_dir, event_log):
    output_dir = Path(event_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_id = _safe_filename_part(event_log.get("camera_id", "camera"))
    event_type = _safe_filename_part(event_log.get("event_type", "event"))
    timestamp = _safe_filename_part(str(event_log.get("timestamp", "0")))
    track_id = _safe_filename_part(str(event_log.get("track_id", "none")))
    frame_idx = _safe_filename_part(str(event_log.get("frame_idx", "0")))
    output_path = output_dir / f"{camera_id}_{event_type}_{timestamp}_track-{track_id}_frame-{frame_idx}.json"
    output_path.write_text(json.dumps(event_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def update_prediction_counts(summary, prediction):
    label = prediction.get("label")
    if label == "Faint":
        summary["faint_predictions"] += 1
    elif label == "Normal":
        summary["normal_predictions"] += 1


def update_tracking_summary(summary, tracker_diagnostics):
    summary["active_tracks"] = int(tracker_diagnostics.get("active_tracks", 0))
    summary["max_active_tracks"] = max(summary.get("max_active_tracks", 0), summary["active_tracks"])
    summary["new_tracks"] += int(tracker_diagnostics.get("new_tracks", 0))
    summary["lost_tracks"] += int(tracker_diagnostics.get("lost_tracks", 0))
    summary["id_switch_like_events"] += int(tracker_diagnostics.get("id_switch_like_events", 0))
    summary["track_diagnostics"] = tracker_diagnostics.get("tracks", {})
