import time

from ai.action.faint_post_processing import DEFAULT_FAINT_THRESHOLD, faint_probability


def initial_overlay_summary():
    return {
        "_started_at": time.perf_counter(),
        "frames_processed": 0,
        "bbox_detections": 0,
        "keypoints_extracted": 0,
        "generated_sequences": 0,
        "lstm_predictions": 0,
        "events_generated": 0,
        "active_tracks": 0,
        "max_active_tracks": 0,
        "per_track_sequences_generated": {},
        "faint_predictions": 0,
        "normal_predictions": 0,
        "events_generated_by_track": {},
        "latest_faint_probability": None,
        "latest_prediction_label": None,
        "latest_consecutive_faint": 0,
        "latest_frame_bbox": 0,
        "latest_frame_keypoints": 0,
        "runtime_seconds": 0.0,
        "effective_fps": 0.0,
        "sample_event": None,
    }


def update_overlay_runtime(summary):
    started_at = float(summary.get("_started_at", time.perf_counter()))
    runtime_seconds = max(0.0, time.perf_counter() - started_at)
    frames_processed = int(summary.get("frames_processed", 0))
    summary["runtime_seconds"] = round(runtime_seconds, 3)
    summary["effective_fps"] = round(frames_processed / runtime_seconds, 2) if runtime_seconds > 0 else 0.0


def format_action_overlay_text(prediction, threshold, consecutive_faint=0, event_triggered=False):
    faint_prob = faint_probability(prediction)
    if faint_prob is None:
        return f"pred: waiting | th: {float(threshold):.2f}"
    label = prediction.get("label", "unknown")
    if event_triggered:
        return f"ALERT Faint: {faint_prob:.2f}"
    return f"Faint: {faint_prob:.2f} | pred: {label} | th: {float(threshold):.2f} | seq: {int(consecutive_faint)}"


def bbox_visual_state(box, threshold=DEFAULT_FAINT_THRESHOLD):
    if box.get("event_triggered"):
        return "alert"
    faint_prob = box.get("faint_probability")
    if faint_prob is not None and float(faint_prob) >= float(threshold):
        return "warning"
    return "normal"


def format_bbox_label(box, threshold=DEFAULT_FAINT_THRESHOLD):
    track_id = box.get("track_id")
    track_text = f"ID {int(track_id)}" if track_id is not None else "ID ?"
    faint_prob = box.get("faint_probability")
    state = bbox_visual_state(box, threshold)
    if state == "alert":
        prob_text = "n/a" if faint_prob is None else f"{float(faint_prob):.2f}"
        return f"ALERT | {track_text} | Faint {prob_text}"
    if state == "warning":
        return f"{track_text} | Faint {float(faint_prob):.2f}"
    return track_text


def annotate_boxes_with_action(boxes, prediction, args, consecutive_faint=0, event_triggered=False):
    text = format_action_overlay_text(prediction, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD), consecutive_faint, event_triggered)
    faint_prob = faint_probability(prediction)
    for box in boxes:
        box["action_overlay"] = text
        box["event_triggered"] = bool(event_triggered)
        box["faint_probability"] = faint_prob
        box["action_threshold"] = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    return boxes


def annotate_boxes_with_track_actions(boxes, predictions_by_track, consecutive_by_track, triggered_track_ids, args):
    threshold = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    for box in boxes:
        track_id = box.get("track_id")
        box["action_threshold"] = threshold
        if track_id is None:
            box["action_overlay"] = format_action_overlay_text(None, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD))
            continue
        track_id = int(track_id)
        prediction = predictions_by_track.get(track_id)
        box["faint_probability"] = faint_probability(prediction)
        event_triggered = track_id in triggered_track_ids
        box["action_overlay"] = format_action_overlay_text(
            prediction,
            getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD),
            consecutive_by_track.get(track_id, 0),
            event_triggered,
        )
        box["event_triggered"] = bool(event_triggered)
    return boxes


def draw_metrics_panel(frame, summary, args, prediction):
    import cv2

    lines = [
        f"camera: {args.camera_id}  fps: {float(summary.get('effective_fps', 0.0)):.2f}",
        f"detector: {args.detector_mode}",
        f"frames: {summary['frames_processed']}",
        f"frame bbox: {summary.get('latest_frame_bbox', 0)}  keypoints: {summary.get('latest_frame_keypoints', 0)}  tracks: {summary.get('active_tracks', 0)}",
        f"total bbox: {summary['bbox_detections']}  total keypoints: {summary['keypoints_extracted']}",
        f"seq: {summary['generated_sequences']}  pred: {summary['lstm_predictions']}  events: {summary['events_generated']}",
        f"threshold: {getattr(args, 'action_threshold', DEFAULT_FAINT_THRESHOLD):.2f}  consecutive_faint: {summary.get('latest_consecutive_faint', 0)}",
    ]
    if prediction:
        faint_prob = faint_probability(prediction)
        faint_text = "n/a" if faint_prob is None else f"{faint_prob:.2f}"
        lines.append(f"label: {prediction['label']}  confidence: {prediction['score']:.2f}  faint: {faint_text}")

    x, y = 12, 62
    width = 560
    height = 24 + 24 * len(lines)
    cv2.rectangle(frame, (x - 8, y - 22), (x + width, y - 22 + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 8, y - 22), (x + width, y - 22 + height), (80, 220, 120), 1)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)


def make_placeholder(message):
    import cv2
    import numpy as np

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (18, 22, 30)
    cv2.putText(image, "AI OVERLAY", (28, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, message, (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 185, 205), 2, cv2.LINE_AA)
    return image
