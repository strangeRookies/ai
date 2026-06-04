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
        "new_tracks": 0,
        "lost_tracks": 0,
        "id_switch_like_events": 0,
        "track_diagnostics": {},
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
    raw_id = box.get("track_id")
    display_id = box.get("display_id")
    # Prefer display_id for the operator-facing label; fall back to raw
    shown_id = display_id if display_id is not None else (int(raw_id) if raw_id is not None else None)
    track_text = f"ID {shown_id}" if shown_id is not None else "ID ?"
    faint_prob = box.get("faint_probability")
    state = bbox_visual_state(box, threshold)
    if state == "alert":
        prob_text = "n/a" if faint_prob is None else f"{float(faint_prob):.2f}"
        return append_track_debug(f"ALERT | {track_text} | Faint {prob_text}", box)
    if state == "warning":
        return append_track_debug(f"{track_text} | Faint {float(faint_prob):.2f}", box)
    return append_track_debug(track_text, box)


def append_track_debug(text, box):
    if not box.get("overlay_debug_tracks"):
        return text
    age = box.get("track_age")
    missing = box.get("missing_frames", 0)
    conf = box.get("track_confidence")
    conf_text = "n/a" if conf is None else f"{float(conf):.2f}"
    age_text = "?" if age is None else str(int(age))
    # Show raw track_id alongside display_id in debug mode
    raw_id = box.get("track_id")
    display_id = box.get("display_id")
    if raw_id is not None and display_id is not None and int(raw_id) != int(display_id):
        raw_suffix = f" / raw {int(raw_id)}"
    else:
        raw_suffix = ""
    return f"{text}{raw_suffix} | age {age_text} | miss {int(missing)} | conf {conf_text}"


def annotate_boxes_with_action(boxes, prediction, args, consecutive_faint=0, event_triggered=False):
    text = format_action_overlay_text(prediction, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD), consecutive_faint, event_triggered)
    faint_prob = faint_probability(prediction)
    for box in boxes:
        box["action_overlay"] = text
        box["event_triggered"] = bool(event_triggered)
        box["faint_probability"] = faint_prob
        box["action_threshold"] = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
        box["overlay_debug_tracks"] = bool(getattr(args, "overlay_debug_tracks", False))
    return boxes


def annotate_boxes_with_track_actions(boxes, predictions_by_track, consecutive_by_track, triggered_track_ids, args):
    threshold = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    for box in boxes:
        track_id = box.get("track_id")
        box["action_threshold"] = threshold
        box["overlay_debug_tracks"] = bool(getattr(args, "overlay_debug_tracks", False))
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

    fps = float(summary.get("effective_fps", 0.0))
    active_tracks = int(summary.get("active_tracks", 0))
    latest_bbox = int(summary.get("latest_frame_bbox", 0))
    
    # Line 1: System info & detections
    line1 = f"AI ACTIVE | CAM: {args.camera_id} | FPS: {fps:.1f} | TRACKS: {active_tracks} | BBOX: {latest_bbox}"
    
    # Line 2: Inference & sequence rule status
    faint_prob = faint_probability(prediction) if prediction else summary.get("latest_faint_probability")
    faint_text = "N/A" if faint_prob is None else f"{float(faint_prob):.2f}"
    label = prediction.get("label", "Normal") if prediction else "Normal"
    th = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    seq = summary.get("latest_consecutive_faint", 0)
    
    line2 = f"PRED: {label} | FAINT: {faint_text} (TH: {th:.2f}) | SEQ: {int(seq)}"

    x, y = 10, 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    
    # Compute size of the box based on text length
    text_w1, _ = cv2.getTextSize(line1, font, scale, thickness)[0]
    text_w2, _ = cv2.getTextSize(line2, font, scale, thickness)[0]
    width = max(text_w1, text_w2) + 20
    height = 38
    
    # Draw background with overlay blend for semi-transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 14), (x + width, y + height), (15, 23, 42), -1)
    cv2.rectangle(overlay, (x - 6, y - 14), (x + width, y + height), (74, 222, 128), 1)
    
    # Blend overlay with original frame (alpha=0.75)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw text
    cv2.putText(frame, line1, (x, y), font, scale, (241, 245, 249), thickness, cv2.LINE_AA)
    cv2.putText(frame, line2, (x, y + 20), font, scale, (241, 245, 249), thickness, cv2.LINE_AA)


def make_placeholder(message):
    import cv2
    import numpy as np

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (18, 22, 30)
    cv2.putText(image, "AI OVERLAY", (28, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, message, (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 185, 205), 2, cv2.LINE_AA)
    return image
