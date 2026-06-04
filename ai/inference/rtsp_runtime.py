from ai.action.faint_post_processing import (
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    faint_probability,
)
from ai.publishers.event_publisher import build_event_payload


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
    payload = build_event_payload(
        camera_id=args.camera_id,
        frame_idx=packet.frame_idx,
        timestamp=packet.timestamp,
        event_type=prediction["label"],
        score=prediction["score"],
        boxes=boxes,
        snapshot_path=None,
    )
    bbox = sequence.get("bbox") if sequence else None
    track_id = sequence.get("track_id") if sequence else None
    payload["bbox"] = bbox
    payload["confidence"] = prediction["score"]
    payload["threshold"] = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    payload["track_id"] = track_id
    payload["severity"] = getattr(args, "event_severity", "HIGH")
    payload["sequence_window"] = {"start": sequence["start_frame"], "end": sequence["end_frame"]} if sequence else None
    payload["probabilities"] = prediction.get("probabilities", {})
    payload["post_processing"] = {
        "min_consecutive_faint": getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        "camera_cooldown_seconds": getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
    }
    return payload


def update_prediction_counts(summary, prediction):
    label = prediction.get("label")
    if label == "Faint":
        summary["faint_predictions"] += 1
    elif label == "Normal":
        summary["normal_predictions"] += 1
