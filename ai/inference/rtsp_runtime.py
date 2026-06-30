import json
import os
import re
import time as _time
from pathlib import Path

from ai.action.classifier import LSTMActionClassifier, MockActionClassifier
from ai.action.cheap_filter import CheapFilterConfig
from ai.action.faint_post_processing import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    faint_probability,
)
from ai.postprocess.supervision_postprocessor import SupervisionPostProcessor
from ai.publishers.mqtt_payloads import build_confirmed_event_payload, frame_size_from_shape
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector
from tracking.simple_tracker import SimpleTrackAssigner


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


def create_detector(mode, model, device, imgsz=640, conf=0.25):
    if mode == "mock":
        return MockDetector(model_name="mock-pose-detector")
    return YoloPoseDetector(model, device=device, imgsz=imgsz, conf=conf)


def create_classifier(action_model=DEFAULT_ACTION_MODEL, device="auto", action_threshold=DEFAULT_FAINT_THRESHOLD):
    if action_model:
        return LSTMActionClassifier(action_model, device=device, faint_threshold=action_threshold), "lstm_checkpoint"
    return MockActionClassifier(default_label="Faint", score=0.80), "mock_lstm"


def cheap_filter_config_from_args(args):
    return CheapFilterConfig(
        enabled=bool(getattr(args, "cheap_filter_enabled", True)),
        slope_ratio_threshold=float(getattr(args, "cheap_filter_slope_ratio", 1.3)),
        min_avg_keypoint_confidence=float(getattr(args, "cheap_filter_min_keypoint_conf", 0.25)),
        min_bbox_area_ratio=float(getattr(args, "cheap_filter_min_bbox_area_ratio", 0.005)),
        min_center_drop_ratio=float(getattr(args, "cheap_filter_min_center_drop_ratio", 0.03)),
        min_aspect_ratio_growth=float(getattr(args, "cheap_filter_min_aspect_ratio_growth", 0.20)),
        min_risk_score=float(getattr(args, "cheap_filter_min_risk_score", 1.0)),
    )


def env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_detection_postprocessor(args):
    tracking_mode = str(getattr(args, "tracking_mode", "auto") or "auto").strip().lower()
    if tracking_mode == "supervision" or (
        tracking_mode == "auto" and env_flag("ENABLE_SUPERVISION_POSTPROCESSING", False)
    ):
        from ai.postprocess.supervision_postprocessor import SupervisionPostProcessorConfig
        config = SupervisionPostProcessorConfig(
            track_thresh=getattr(args, "track_thresh", 0.10),
            track_buffer=getattr(args, "track_buffer", 90),
            match_thresh=getattr(args, "match_thresh", 0.20),
            frame_rate=getattr(args, "frame_rate", 30),
            bbox_smoothing_alpha=getattr(args, "bbox_smoothing_alpha", 1.0),
        )
        return SupervisionPostProcessor(config=config), "supervision"
    return SimpleTrackAssigner(
        track_thresh=getattr(args, "track_thresh", 0.10),
        match_thresh=getattr(args, "match_thresh", 0.20),
        track_buffer=getattr(args, "track_buffer", 90),
        min_box_area=getattr(args, "min_box_area", 100.0),
        bbox_smoothing_alpha=getattr(args, "bbox_smoothing_alpha", 0.60),
        max_missing_seconds=getattr(args, "track_max_missing_seconds", 4.0),
        center_match_ratio=getattr(args, "center_match_ratio", 0.70),
    ), "simple_tracker"


def update_detections_with_postprocessor(postprocessor, detections, frame, timestamp):
    if isinstance(postprocessor, SupervisionPostProcessor):
        return postprocessor.process(detections, frame)
    return postprocessor.update(detections, now=timestamp)


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
        f"frame_id={summary.get('latest_frame_id', '')} "
        f"captured_at_ms={summary.get('latest_captured_at_ms', '')} "
        f"ai_latency_ms={summary.get('latest_ai_latency_ms', '')} "
        f"publish_latency_ms={summary.get('latest_publish_latency_ms', '')} "
        f"bbox={len(boxes)} "
        f"keypoints={summary.get('latest_frame_keypoints', 0)} "
        f"active_tracks={summary.get('active_tracks', 0)} "
        f"seq={summary['generated_sequences']} "
        f"pred={summary['lstm_predictions']} "
        f"latest_faint_prob={faint_text}",
        flush=True,
    )


def build_inference_event_payload(
    args,
    packet,
    prediction,
    boxes,
    sequence,
    frame_metadata=None,
    published_at_ms=None,
    dropped_frame_count=None,
    snapshot_path=None,
    clip_path=None,
):
    camera_login_id = getattr(args, "camera_login_id", None) or args.camera_id
    frame = getattr(packet, "frame", None)
    frame_width, frame_height = frame_size_from_shape(frame.shape) if frame is not None else (0, 0)
    return build_confirmed_event_payload(
        stream_id=camera_login_id,
        frame_width=frame_width,
        frame_height=frame_height,
        prediction=prediction,
        sequence=sequence,
        boxes=boxes,
        timestamp_ms=published_at_ms or int(_time.time() * 1000),
        frame_id=getattr(frame_metadata, "frame_id", None),
        captured_at_ms=getattr(frame_metadata, "captured_at_ms", None),
        processed_at_ms=getattr(frame_metadata, "processed_at_ms", None),
        published_at_ms=published_at_ms,
        sequence_metadata=sequence_metadata(sequence, args),
        dropped_frame_count=dropped_frame_count,
        snapshot_path=snapshot_path,
        clip_path=clip_path,
    )


def sequence_metadata(sequence, args):
    if not sequence:
        return None
    return {
        "sequenceLength": int(getattr(args, "sequence_length", 0)),
        "sequenceStride": int(getattr(args, "sequence_stride", 0)),
        "sequenceStartFrameId": int(sequence.get("sequence_start_frame_id", sequence.get("start_frame", 0))),
        "sequenceEndFrameId": int(sequence.get("sequence_end_frame_id", sequence.get("end_frame", 0))),
        "sequenceStartAtMs": sequence.get("sequence_start_at_ms"),
        "sequenceEndAtMs": sequence.get("sequence_end_at_ms"),
    }


def build_inference_event_log(args, packet, prediction, boxes, sequence):
    bbox = sequence.get("bbox") if sequence else None
    track_id = sequence.get("track_id") if sequence else None
    faint_prob = faint_probability(prediction)
    return {
        "camera_id": args.camera_id,
        "frame_idx": int(packet.frame_idx),
        "timestamp": float(packet.timestamp),
        "event_type": prediction["label"],
        "confidence": float(prediction["score"]),
        "faint_prob": faint_prob,
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
