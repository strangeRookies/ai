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
from ai.inference.pose_diagnostics import PoseDiagnosticsReporter, config_from_args as pose_diagnostics_config_from_args
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


def classifier_contract_summary(classifier):
    return {
        "checkpoint_path": getattr(classifier, "checkpoint_path", None),
        "checkpoint_input_size": getattr(classifier, "input_size", None),
        "feature_schema_version": getattr(classifier, "feature_schema", None),
        "feature_names_count": len(getattr(classifier, "feature_names", []) or []),
        "sequence_length": getattr(classifier, "checkpoint_sequence_length", None),
        "sequence_stride": getattr(classifier, "checkpoint_sequence_stride", None),
    }


def log_classifier_contract(prefix, camera_login_id, classifier):
    summary = classifier_contract_summary(classifier)
    print(
        f"{prefix} "
        f"cameraLoginId={camera_login_id} "
        f"checkpoint_path={summary['checkpoint_path']} "
        f"checkpoint_input_size={summary['checkpoint_input_size']} "
        f"feature_schema_version={summary['feature_schema_version']} "
        f"feature_names_count={summary['feature_names_count']} "
        f"sequence_length={summary['sequence_length']} "
        f"sequence_stride={summary['sequence_stride']}",
        flush=True,
    )


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
    """tracking backend를 선택하고 YOLO detection 후처리기를 만든다.

    기본 파이프라인은 Supervision ByteTrack이다. `tracking_mode=auto`에서는
    `ENABLE_SUPERVISION_POSTPROCESSING`이 켜진 경우만 supervision을 쓰고, 그렇지
    않으면 lightweight simple tracker로 간다. 이 함수에서 만든 객체가 프레임마다
    detection에 `track_id`를 붙이는 책임을 가진다.
    """

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
            stability_fallback=bool(getattr(args, "tracking_stability_fallback", False)),
            fallback_max_missing_seconds=getattr(args, "track_max_missing_seconds", 4.0),
            fallback_center_match_ratio=getattr(args, "center_match_ratio", 0.70),
            session_reconnect=bool(getattr(args, "person_session_reconnect", False)),
            session_reconnect_max_missing_seconds=getattr(args, "person_session_reconnect_max_missing_seconds", 3.0),
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
    """선택된 tracker 인터페이스 차이를 숨기고 tracked detections를 반환한다.

    SupervisionPostProcessor는 frame을 받는 `process()` API이고, SimpleTrackAssigner는
    timestamp 기반 `update()` API다. 상위 frame loop는 이 차이를 몰라도 되게 여기서
    한 번만 분기한다.
    """

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
    frame_id = getattr(frame_metadata, "frame_id", None)
    if frame_id is None:
        frame_id = getattr(packet, "frame_id", None)
    captured_at_ms = getattr(frame_metadata, "captured_at_ms", None)
    if captured_at_ms is None:
        captured_at_ms = getattr(packet, "captured_at_ms", None)
    processed_at_ms = getattr(frame_metadata, "processed_at_ms", None)
    return build_confirmed_event_payload(
        stream_id=camera_login_id,
        frame_width=frame_width,
        frame_height=frame_height,
        prediction=prediction,
        sequence=sequence,
        boxes=boxes,
        timestamp_ms=published_at_ms or int(_time.time() * 1000),
        frame_id=frame_id,
        captured_at_ms=captured_at_ms,
        processed_at_ms=processed_at_ms,
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


def create_pose_diagnostics_reporter(args) -> PoseDiagnosticsReporter:
    return PoseDiagnosticsReporter(pose_diagnostics_config_from_args(args))


def build_pose_tracking_config_dump(args, tracker_diagnostics: dict | None = None) -> dict:
    """worker 시작 시 실제 적용된 pose/tracking 설정을 JSON-serializable dict로 만든다.

    ByteTrack 생성자에서 사용된/무시된 parameter도 함께 남긴다. supervision 버전이
    바뀌면 `track_thresh` 같은 기존 이름이 무시될 수 있으므로, 이 dump가 현장 로그에서
    가장 먼저 확인해야 하는 설정 증거다.
    """

    bytetrack_constructor = (tracker_diagnostics or {}).get("bytetrack_constructor", {})
    return {
        "stage": "pose_tracking_config",
        "TRACK_THRESH": float(getattr(args, "track_thresh", 0.10)),
        "TRACK_IOU_THRESHOLD": float(getattr(args, "match_thresh", 0.20)),
        "TRACK_BUFFER": int(getattr(args, "track_buffer", 90)),
        "TRACK_FRAME_RATE": int(getattr(args, "frame_rate", 30)),
        "TRACKING_GRACE_PERIOD_SECONDS": float(getattr(args, "tracking_grace_period_seconds", 4.0)),
        "TRACKING_RELINK_IOU_THRESHOLD": float(getattr(args, "tracking_relink_iou_threshold", 0.30)),
        "TRACKING_RELINK_CENTER_RATIO": float(getattr(args, "tracking_relink_center_ratio", 0.70)),
        "TRACKING_RELINK_MAX_TIME_GAP_SECONDS": float(
            getattr(args, "tracking_relink_max_time_gap_seconds", 2.0)
        ),
        "POSE_MIN_KEYPOINT_CONFIDENCE": float(getattr(args, "pose_min_keypoint_confidence", 0.25)),
        "POSE_DEBUG_SUMMARY_EVERY_N": int(getattr(args, "pose_debug_summary_every_n", 60)),
        "POSE_DEBUG_SAVE_IMAGES": bool(getattr(args, "pose_debug_save_images", False)),
        "POSE_TRACKING_DIAG_JSONL": bool(getattr(args, "pose_tracking_diag_jsonl", False)),
        "POSE_TRACKING_DIAG_JSONL_PATH": str(
            getattr(args, "pose_tracking_diag_jsonl_path", "runs/diagnostics/pose_tracking_diag.jsonl")
        ),
        "bytetrack_constructor_used": dict(bytetrack_constructor.get("used") or {}),
        "bytetrack_constructor_ignored": dict(bytetrack_constructor.get("ignored") or {}),
        "bytetrack_constructor_supported_parameters": list(
            bytetrack_constructor.get("supported_parameters") or []
        ),
    }


def log_pose_tracking_config(args, detection_postprocessor) -> dict:
    diagnostics = detection_postprocessor.diagnostics() if hasattr(detection_postprocessor, "diagnostics") else {}
    record = build_pose_tracking_config_dump(args, diagnostics)
    print(f"[pose-tracking-config] {json.dumps(record, ensure_ascii=False)}", flush=True)
    return record


# ---------------------------------------------------------------------------
# 4단계 정량 검증 로그 함수 (TRACKING_DEBUG=true 일 때만 출력)
# Stage 1: Detection, Stage 2: Tracking, Stage 3: Classification, Stage 4: Payload
# ---------------------------------------------------------------------------

def _tracking_debug_enabled() -> bool:
    return os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}


def log_detection_stage(
    camera_login_id: str,
    frame_id,
    timestamp_ms,
    raw_detections: list,
) -> None:
    """Stage 1: YOLO detection 직후 — raw bbox와 confidence를 frameId별로 기록."""
    if not _tracking_debug_enabled():
        return
    entries = []
    for d in raw_detections:
        bbox = d.get("bbox") or []
        entries.append({
            "bbox": [round(float(v), 2) for v in bbox[:4]] if bbox else [],
            "confidence": round(float(d.get("confidence", 0.0)), 4),
            "hasKeypoints": bool(d.get("keypoints")),
        })
    record = {
        "stage": "detection",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "timestampMs": timestamp_ms,
        "detectionCount": len(raw_detections),
        "detections": entries,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)


def log_tracking_stage(
    camera_login_id: str,
    frame_id,
    pre_detections: list,
    post_detections: list,
    diagnostics: dict,
) -> None:
    """Stage 2: ByteTrack 적용 직후 — trackId 부여 여부, new/lost tracks 기록."""
    if not _tracking_debug_enabled():
        return
    tracked_count = sum(1 for d in post_detections if d.get("track_id") is not None)
    missing_track_count = len(post_detections) - tracked_count
    active_ids = sorted({int(d["track_id"]) for d in post_detections if d.get("track_id") is not None})
    keypoint_confidences = [
        float(point.get("confidence", 0.0))
        for detection in post_detections
        for point in detection.get("keypoints") or []
        if point.get("confidence") is not None
    ]
    avg_keypoint_confidence = (
        round(sum(keypoint_confidences) / len(keypoint_confidences), 4)
        if keypoint_confidences
        else None
    )
    track_details = []
    for d in post_detections:
        bbox = d.get("bbox") or []
        per_track_keypoint_confidences = [
            float(point.get("confidence", 0.0))
            for point in d.get("keypoints") or []
            if point.get("confidence") is not None
        ]
        per_track_avg_confidence = (
            round(sum(per_track_keypoint_confidences) / len(per_track_keypoint_confidences), 4)
            if per_track_keypoint_confidences
            else None
        )
        track_details.append({
            "trackId": d.get("track_id"),
            "displayId": d.get("display_id"),
            "bbox": [round(float(v), 2) for v in bbox[:4]] if bbox else [],
            "confidence": round(float(d.get("confidence", 0.0)), 4),
            "keypointCount": len(d.get("keypoints") or []),
            "avgKeypointConfidence": per_track_avg_confidence,
            "fallbackRisk": d.get("track_id") is None,
        })
    if len(pre_detections) > 0 and tracked_count == 0:
        diagnosis = "tracker_gating"
    elif tracked_count > 0 and int(diagnostics.get("active_tracks", tracked_count)) == 0:
        diagnosis = "active_track_filtering"
    else:
        diagnosis = "tracking_ok"
    record = {
        "stage": "tracking",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "detectionCount": len(pre_detections),
        "trackerInputCount": len(pre_detections),
        "trackedCount": tracked_count,
        "missingTrackCount": missing_track_count,
        "avgKeypointConfidence": avg_keypoint_confidence,
        "activeTrackIds": active_ids,
        "newTracks": diagnostics.get("new_tracks", 0),
        "lostTracks": diagnostics.get("lost_tracks", 0),
        "idSwitchLike": diagnostics.get("id_switch_like_events", 0),
        "diagnosis": diagnosis,
        "trackDetails": track_details,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)


def log_classification_stage(
    camera_login_id: str,
    frame_id,
    track_id,
    prediction: dict,
    faint_threshold: float,
    consecutive_count: int = 0,
    event_triggered: bool = False,
    checkpoint_input_size=None,
    runtime_feature_dim=None,
    tensor_shape=None,
    feature_schema=None,
    checkpoint_path=None,
) -> None:
    """Stage 3: LSTM 분류 직후 — trackId별 faintProbability와 threshold 통과 여부 기록."""
    if not _tracking_debug_enabled():
        return
    fp = faint_probability(prediction) if prediction else None
    record = {
        "stage": "classification",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "trackId": track_id,
        "predictionLabel": prediction.get("label") if prediction else None,
        "faintProbability": round(float(fp), 4) if fp is not None else None,
        "faintThreshold": faint_threshold,
        "thresholdPassed": (fp is not None and fp >= faint_threshold),
        "consecutiveCount": consecutive_count,
        "isFaintEvent": event_triggered,
        "checkpointInputSize": checkpoint_input_size,
        "runtimeFeatureDim": runtime_feature_dim,
        "tensorShape": list(tensor_shape) if tensor_shape is not None else None,
        "featureSchema": feature_schema,
        "checkpointPath": checkpoint_path,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)


def log_payload_stage(
    camera_login_id: str,
    frame_id,
    overlay_payload: dict,
) -> None:
    """Stage 4: build_overlay_payload 직후 — 실제 publish될 이벤트 목록 기록."""
    if not _tracking_debug_enabled():
        return
    events = overlay_payload.get("events", [])
    event_summaries = []
    for e in events:
        bbox = e.get("bbox") or e.get("boundingBox") or {}
        event_summaries.append({
            "trackId": e.get("trackId") or e.get("track_id") or e.get("trackingId"),
            "type": e.get("type"),
            "confidence": e.get("confidence"),
            "eventTriggered": e.get("eventTriggered", False),
            "bbox": bbox,
        })
    record = {
        "stage": "payload",
        "cameraLoginId": camera_login_id,
        "frameId": frame_id,
        "timestampMs": overlay_payload.get("timestampMs"),
        "capturedAtMs": overlay_payload.get("capturedAtMs"),
        "processedAtMs": overlay_payload.get("processedAtMs"),
        "publishedAtMs": overlay_payload.get("publishedAtMs"),
        "droppedFrameCount": overlay_payload.get("droppedFrameCount"),
        "eventCount": len(events),
        "events": event_summaries,
    }
    print(f"[stage-log] {json.dumps(record, ensure_ascii=False)}", flush=True)


def initial_quantitative_summary() -> dict:
    """서버 시작 시 정량 지표 summary 항목 초기화."""
    return {
        "total_frames": 0,
        "detected_frames": 0,
        "tracked_frames": 0,
        "missing_bbox_frames": 0,
        "fallback_track_id_frames": 0,
        "track_switch_like_events": 0,
    }


def update_quantitative_summary(summary: dict, boxes: list, tracker_diagnostics: dict | None = None) -> None:
    """매 프레임마다 정량 지표를 갱신한다."""
    summary["total_frames"] = summary.get("total_frames", 0) + 1
    det_count = len(boxes)
    tracked_count = sum(1 for b in boxes if b.get("track_id") is not None)
    missing_track_count = det_count - tracked_count

    if det_count > 0:
        summary["detected_frames"] = summary.get("detected_frames", 0) + 1
    else:
        summary["missing_bbox_frames"] = summary.get("missing_bbox_frames", 0) + 1

    if tracked_count > 0:
        summary["tracked_frames"] = summary.get("tracked_frames", 0) + 1

    if missing_track_count > 0:
        summary["fallback_track_id_frames"] = summary.get("fallback_track_id_frames", 0) + 1

    if tracker_diagnostics:
        summary["track_switch_like_events"] = (
            summary.get("track_switch_like_events", 0)
            + int(tracker_diagnostics.get("id_switch_like_events", 0))
        )

