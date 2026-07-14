import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# sys.path 등록이 먼저 수행되어야 하위 ai 패키지 로드 가능 (작업 디렉토리와 무관하도록 insert 사용)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from ai.events.event_clip import EventClipBuffer
from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip

from ai.events.event_clip import EventClipBuffer
from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip
from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.action.lstm_contract import DEFAULT_KEYPOINT_INPUT_SIZE, DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE, log_lstm_config
from ai.evidence import evidence_id
from ai.frame_sync import FrameMetadataBuffer, FramePacket, CameraFrameQueue
from ai.inference.rtsp_runtime import (
    build_inference_event_payload,
    cheap_filter_config_from_args,
    create_detection_postprocessor,
    ensure_mock_keypoints,
    log_pose_tracking_config,
    apply_tracker_timebase,
    tracker_configured_fps,
)
from ai.inference.rtsp_runtime import (
    log_classifier_contract,
    maybe_log_debug,
    normalize_detections,
    update_detections_with_postprocessor,
    update_prediction_counts,
    update_tracking_summary,
    lifecycle_payload_kwargs,
)
from ai.inference.rtsp_runtime import (
    log_detection_stage,
    log_tracking_stage,
    log_classification_stage,
    log_payload_stage,
    update_quantitative_summary,
)
from ai.inference.tracking_debug import (
    log_compact_tracking_debug,
    log_sequence_stage,
    log_track_lifecycle_events,
)
from ai.inference.fps_audit import FpsAuditWindow, log_fps_audit
from ai.inference.tracker_timebase import TrackerUpdateFpsEstimator
from ai.inference.session_boundary import session_reset_reason
from ai.inference.pose_diagnostics import PoseDiagnosticsReporter, config_from_args as pose_diagnostics_config_from_args
from ai.overlay_http import OverlayState, create_overlay_server
from ai.roi import apply_roi_mask, combine_roi_masks, find_boxes_in_exit_zone, find_boxes_in_hazard_zone
from ai.streams.video_reader import VideoReader
from ai.visualization.action_overlay import annotate_boxes_with_action, annotate_boxes_with_track_actions, draw_metrics_panel, faint_probability
from ai.visualization.action_overlay import format_action_overlay_text, initial_overlay_summary, update_overlay_runtime
from ai.visualization.draw import draw_overlay
from scripts.run_rtsp_inference import DEFAULT_ACTION_MODEL, DEFAULT_CAMERA_COOLDOWN_SECONDS, DEFAULT_FAINT_THRESHOLD, DEFAULT_MIN_CONSECUTIVE_FAINT
from scripts.run_rtsp_inference import FaintEventPostProcessor, create_classifier, create_detector
from ai.action.fall_lifecycle_config import build_faint_post_processor_from_args, resolved_faint_threshold
from ai.inference.tensorrt_runtime import log_periodic_inference_metrics, log_worker_backend_startup
from ai.runtime_metrics import RuntimeMetrics
from ai.action.faint_post_processing import ExitEventPostProcessor, DEFAULT_EXIT_MIN_CONSECUTIVE, DEFAULT_EXIT_COOLDOWN_SECONDS, HazardEventPostProcessor, DEFAULT_HAZARD_MIN_CONSECUTIVE, DEFAULT_HAZARD_COOLDOWN_SECONDS
from stream.rtsp_reader import redact_url
from tracking.display_id_mapper import DisplayIdMapper
from ai.publishers.async_delivery import AsyncMqttDelivery
from ai.publishers.event_outbox import event_outbox_path
from ai.publishers.event_publisher import create_event_publisher, mqtt_topic_settings_from_args
from ai.publishers.camera_status_publisher import CameraStatusPublisher
from ai.publishers.mqtt_payloads import build_overlay_payload, current_timestamp_ms, frame_size_from_shape, build_frame_sync_payload
from ai.postprocess.incident_recovery import (
    IncidentRecoveryManager,
    make_detect_roi_fn_from_yolo_pose,
)
from ai.postprocess.track_state_migration import finalize_recovery_detections
from ai.action.fall_event_state import FallState


def initial_summary():
    return initial_overlay_summary()

#각 카메라별로 동작하는 실시간 오버레이 스크립트 
#RTSP 스트림을 캡처하여 AI 분석(YOLO Pose 및 LSTM)을 수행
#감지된 객체의 바운딩 박스(bbox), 트래킹 ID 및 상태 메타데이터를 MQTT camera 토픽으로 실시간 발행
class OverlayPublishState:
    """MQTT overlay payload에 넣을 track별 최신 행동 신호를 보존한다.

    LSTM sequence는 매 프레임 만들어지지 않는다. 그래서 어떤 프레임에서는 bbox는
    살아 있지만 faint_probability/event flag가 새로 계산되지 않을 수 있다. 이 상태
    객체는 같은 track_id가 유지되는 동안 마지막 행동 신호를 bbox에 다시 붙여 frontend
    overlay가 한두 프레임마다 깜빡이지 않도록 한다.
    """

    def __init__(self):
        self.signals_by_track = {}
        self.last_timestamp_ms = 0

    def migrate_track(self, old_track_id, new_track_id) -> bool:
        """Move cached overlay signals across recovery track-id migration."""
        old_id, new_id = int(old_track_id), int(new_track_id)
        if old_id == new_id or old_id not in self.signals_by_track:
            return False
        if new_id not in self.signals_by_track:
            self.signals_by_track[new_id] = self.signals_by_track.pop(old_id)
        else:
            self.signals_by_track.pop(old_id, None)
        return True

    def apply_latest_signals(self, boxes):
        active_track_ids = {_track_id(box.get("track_id")) for box in boxes if box.get("track_id") is not None}
        for track_id in list(self.signals_by_track):
            if track_id not in active_track_ids:
                del self.signals_by_track[track_id]

        for box in boxes:
            raw_track_id = box.get("track_id")
            if raw_track_id is None:
                continue
            track_id = _track_id(raw_track_id)
            faint_prob = box.get("faint_probability")
            event_triggered = bool(box.get("event_triggered"))
            if faint_prob is not None or event_triggered:
                self.signals_by_track[track_id] = {
                    "faint_probability": faint_prob,
                    "event_triggered": event_triggered,
                }
                continue

            latest = self.signals_by_track.get(track_id)
            if latest is None:
                continue
            box["faint_probability"] = latest.get("faint_probability")
            box["event_triggered"] = bool(latest.get("event_triggered"))

    def next_timestamp_ms(self):
        timestamp_ms = current_timestamp_ms()
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms


def _track_id(value):
    return int(float(str(value)))


def _record_publish_outcome(summary, prefix, result=None, *, attempted=False):
    if attempted:
        summary[f"{prefix}_attempted"] = int(summary.get(f"{prefix}_attempted", 0)) + 1
    if result is True:
        summary[f"{prefix}_succeeded"] = int(summary.get(f"{prefix}_succeeded", 0)) + 1
    elif attempted:
        summary[f"{prefix}_failed"] = int(summary.get(f"{prefix}_failed", 0)) + 1


def _publish_event(publisher, payload, topic):
    enqueue = getattr(publisher, "enqueue_event", None)
    if callable(enqueue):
        return enqueue(payload, topic)
    return publisher.publish(payload, topic=topic)

def mjpeg_server_enabled(args: argparse.Namespace) -> bool:
    """MJPEG 서버를 켜야 하는지 판단. mjpeg_enabled 또는 mjpeg_debug 중 하나라도 켜지면 서버를 시작한다."""
    return bool(getattr(args, "mjpeg_enabled", False) or getattr(args, "mjpeg_debug", False))


def mjpeg_debug_enabled(args: argparse.Namespace) -> bool:
    """하위 호환성 alias: 기존 코드가 mjpeg_debug_enabled를 참조하므로 mjpeg_server_enabled와 동일."""
    return mjpeg_server_enabled(args)


def mjpeg_overlay_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "mjpeg_enable_overlay", True))


# Per-worker canary metric window (process-local; one overlay process = one camera).
_CANARY_METRICS: dict = {
    "window_started": None,
    "new_tracks": 0,
    "lost_tracks": 0,
    "multi_det_extra": 0,
    "near_dup_suppress": 0,
    "duplicate_frames": 0,
    "max_active_tracks": 0,
    "frames": 0,
    "person_present_frames": 0,
    "dominant_hist": {},
}


def _bbox_iou_simple(a, b) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _emit_canary_tracking_signals(camera_login_id, frame_id, detections, tracker_diag, args) -> None:
    """Log near-dup suppress events and periodic [tracking-canary] windows when canary is on.

    Also emits lightweight windows when TRACKING_CANARY_METRICS=true for baseline collection.
    """
    canary_on = (os.getenv("TRACKING_CANARY") or "").lower() in {"1", "true", "yes", "on"}
    metrics_on = canary_on or (os.getenv("TRACKING_CANARY_METRICS") or "").lower() in {"1", "true", "yes", "on"}
    if not metrics_on:
        return
    now = time.time()
    state = _CANARY_METRICS
    if state["window_started"] is None:
        state["window_started"] = now
    diag = dict(tracker_diag or {})
    events = list(diag.get("lifecycle_events") or [])
    state["new_tracks"] += int(diag.get("new_tracks") or 0)
    state["lost_tracks"] += int(diag.get("lost_tracks") or 0)
    state["frames"] += 1
    active = int(diag.get("active_tracks") or 0)
    state["max_active_tracks"] = max(int(state["max_active_tracks"]), active)
    tracked = [d for d in (detections or []) if d.get("track_id") is not None]
    if tracked:
        state["person_present_frames"] += 1
        # dominant id hist
        best = max(tracked, key=lambda d: float(d.get("confidence") or 0.0))
        tid = int(best["track_id"])
        hist = state["dominant_hist"]
        hist[tid] = int(hist.get(tid, 0)) + 1
    # duplicate active boxes
    boxes = [t.get("bbox") for t in tracked if t.get("bbox")]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _bbox_iou_simple(boxes[i], boxes[j]) >= 0.7:
                state["duplicate_frames"] += 1
                break
        else:
            continue
        break
    for ev in events:
        if ev.get("event") == "new_track" and (ev.get("switchReason") == "MULTI_DET_EXTRA"):
            state["multi_det_extra"] += 1
        if ev.get("event") == "filter" and ev.get("reason") == "near_duplicate_suppress":
            state["near_dup_suppress"] += 1
            meta = ev.get("suppressMeta") or {}
            best = meta.get("best") or {}
            print(
                f"[near-dup-suppress]\n"
                f"cameraLoginId={camera_login_id}\n"
                f"frameId={frame_id}\n"
                f"claimedTrackId={best.get('trackId')}\n"
                f"iou={best.get('iou')}\n"
                f"normalizedCenterDistance={best.get('centerRatio')}\n"
                f"bboxAreaRatio={best.get('areaRatio')}\n"
                f"keypointDistance={best.get('keypointDist')}\n"
                f"reason=NEAR_DUPLICATE_OF_CLAIMED_DETECTION",
                flush=True,
            )
    window_sec = float(os.getenv("TRACKING_CANARY_WINDOW_SEC") or 60)
    elapsed = now - float(state["window_started"])
    if elapsed < window_sec:
        return
    minutes = max(elapsed / 60.0, 1e-9)
    hist = state["dominant_hist"] or {}
    dominant_frames = max(hist.values()) if hist else 0
    person_frames = int(state["person_present_frames"] or 0)
    retention = (dominant_frames / person_frames) if person_frames else 0.0
    # rough ghost: missing tracks with missing_frames > fps
    ghost_count = 0
    max_ghost = 0.0
    tracks = diag.get("tracks") or {}
    fps = float(getattr(args, "frame_rate", 30) or 30)
    for _tid, tr in tracks.items() if isinstance(tracks, dict) else []:
        miss = int((tr or {}).get("missing_frames") or 0)
        if miss > 0:
            sec = miss / max(fps, 1e-6)
            if sec > 1.0:
                ghost_count += 1
                max_ghost = max(max_ghost, sec)
    print(
        f"[tracking-canary]\n"
        f"cameraLoginId={camera_login_id}\n"
        f"windowSec={round(elapsed, 1)}\n"
        f"analysisFps={round(state['frames'] / max(elapsed, 1e-9), 3)}\n"
        f"newTracks={state['new_tracks']}\n"
        f"newTracksPerMin={round(state['new_tracks'] / minutes, 4)}\n"
        f"lostTracks={state['lost_tracks']}\n"
        f"lostTracksPerMin={round(state['lost_tracks'] / minutes, 4)}\n"
        f"multiDetExtra={state['multi_det_extra']}\n"
        f"duplicateFrames={state['duplicate_frames']}\n"
        f"ghostTrackCount={ghost_count}\n"
        f"maxGhostDurationSec={round(max_ghost, 3)}\n"
        f"maxActiveTracks={state['max_active_tracks']}\n"
        f"suppressedDetections={state['near_dup_suppress']}\n"
        f"idRetentionProxy={round(retention, 4)}\n"
        f"workerPid={os.getpid()}\n"
        f"canary={str(canary_on).lower()}",
        flush=True,
    )
    # reset window
    state["window_started"] = now
    state["new_tracks"] = 0
    state["lost_tracks"] = 0
    state["multi_det_extra"] = 0
    state["near_dup_suppress"] = 0
    state["duplicate_frames"] = 0
    state["max_active_tracks"] = 0
    state["frames"] = 0
    state["person_present_frames"] = 0
    state["dominant_hist"] = {}


def log_worker_startup_contract(args: argparse.Namespace) -> None:
    camera_login_id = getattr(args, "camera_login_id", None) or getattr(args, "camera_id", "")
    print(
        "[ai-worker-startup] "
        f"cameraLoginId={camera_login_id} "
        f"cameraId={getattr(args, 'camera_id', '')} "
        f"rtsp_url={redact_url(getattr(args, 'rtsp_url', '') or '')} "
        f"action_model={getattr(args, 'action_model', None)} "
        f"classifier_input={getattr(args, 'classifier_input', None)} "
        f"selected_track_mode={getattr(args, 'selected_track_mode', None)} "
        f"selected_track_id={getattr(args, 'selected_track_id', None)}",
        flush=True,
    )
    near_dup = (os.getenv("NEAR_DUP_SUPPRESS_MODE") or "hybrid_kp").strip().lower()
    new_track_thresh = os.getenv("SIMPLE_TRACK_NEW_TRACK_THRESH") or "0.30"
    canary = (os.getenv("TRACKING_CANARY") or "false").strip().lower() in {"1", "true", "yes", "on"}
    config_source = "canary-override" if canary else "production-default"
    print(
        f"[tracking-config]\n"
        f"cameraLoginId={camera_login_id}\n"
        f"nearDupSuppressMode={near_dup}\n"
        f"newTrackThresh={new_track_thresh}\n"
        f"canary={str(canary).lower()}\n"
        f"configSource={config_source}",
        flush=True,
    )
    if getattr(args, "selected_track_mode", "strict") == "strict" and getattr(args, "selected_track_id", None) is None:
        print(
            "[selected-track-warning] "
            "strict mode set but selected_track_id is None; selected filtering inactive.",
            flush=True,
        )


def env_optional_int(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return int(value)
    return None


def env_optional_str(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _maybe_register_incident_recovery(
    *,
    incident_recovery,
    camera_login_id,
    track_id,
    emit_decision,
    post_processor,
    track_detection,
    timestamp,
    frame_id,
):
    """Register ROI recovery context when Fall/Faint suspected lifecycle is entered.

    Trigger on internal FALL_CANDIDATE / FALL_CONFIRMED / POST_FALL_LYING (and emit
    new_fall/unrecovered as backup). Does not require MQTT publish success.
    """
    if incident_recovery is None or track_id is None:
        return
    bb = None
    if track_detection is not None:
        bb = track_detection.get("bbox") or track_detection.get("smoothed_bbox")
        if bb is None and all(k in (track_detection or {}) for k in ("x1", "y1", "x2", "y2")):
            bb = [
                track_detection["x1"],
                track_detection["y1"],
                track_detection["x2"],
                track_detection["y2"],
            ]
    if not bb or len(bb) < 4:
        return

    lifecycle_state = None
    incident_id = None
    if emit_decision is not None:
        if getattr(emit_decision, "lifecycle", None) is not None:
            st = emit_decision.lifecycle.state
            lifecycle_state = st.value if hasattr(st, "value") else str(st)
        incident_id = getattr(emit_decision, "event_id", None) or getattr(emit_decision, "original_event_id", None)
        if getattr(emit_decision, "state", None):
            lifecycle_state = lifecycle_state or emit_decision.state

    if lifecycle_state is None and post_processor is not None:
        sm = getattr(post_processor, "_state_machine", None)
        if sm is not None and hasattr(sm, "get_state"):
            try:
                st = sm.get_state(camera_login_id, track_id)
                lifecycle_state = st.value if hasattr(st, "value") else str(st)
            except Exception:
                lifecycle_state = None

    suspected_states = {
        FallState.FALL_CANDIDATE.value,
        FallState.FALL_CONFIRMED.value,
        FallState.POST_FALL_LYING.value,
        "FALL_SUSPECTED",
        "FAINT_SUSPECTED",
    }
    should_register = False
    if lifecycle_state in suspected_states:
        should_register = True
    if emit_decision is not None and (emit_decision.is_new_fall or emit_decision.is_unrecovered):
        should_register = True
    if not should_register:
        return

    incident_recovery.note_fall_faint_suspected(
        camera_login_id=camera_login_id,
        track_id=int(track_id),
        bbox=bb,
        timestamp=float(timestamp),
        frame_id=int(frame_id),
        incident_id=incident_id,
    )


def process_frame(
    frame_packet,
    detector,
    classifier,
    sequence_buffer,
    summary,
    args,
    post_processor=None,
    tracker=None,
    state=None,
    display_id_mapper=None,
    publisher=None,
    overlay_publish_state=None,
    frame_buffer=None,
    dropped_frame_count=None,
    roi_mask=None,
    exit_roi_mask=None,
    exit_post_processor=None,
    hazard_roi_mask=None,
    hazard_post_processor=None,
    track_selector=None,
    sync_sink=None,
    pose_reporter=None,
    incident_recovery=None,
    recovery_detect_fn=None,
):
    """프레임 처리 중 예외가 나면 stage 정보를 붙여 worker log에 남긴다.

    실제 AI 처리 단계는 `_process_frame_impl()`에 있고, 이 wrapper는 장시간 worker가
    실패했을 때 YOLO/Tracking/LSTM/Payload 중 어느 구간에서 터졌는지 찾기 위한
    진입점이다.
    """

    try:
        return _process_frame_impl(
            frame_packet,
            detector,
            classifier,
            sequence_buffer,
            summary,
            args,
            post_processor=post_processor,
            tracker=tracker,
            state=state,
            display_id_mapper=display_id_mapper,
            publisher=publisher,
            overlay_publish_state=overlay_publish_state,
            frame_buffer=frame_buffer,
            dropped_frame_count=dropped_frame_count,
            roi_mask=roi_mask,
            exit_roi_mask=exit_roi_mask,
            exit_post_processor=exit_post_processor,
            hazard_roi_mask=hazard_roi_mask,
            hazard_post_processor=hazard_post_processor,
            track_selector=track_selector,
            sync_sink=sync_sink,
            pose_reporter=pose_reporter,
            incident_recovery=incident_recovery,
            recovery_detect_fn=recovery_detect_fn,
        )
    except Exception as exc:
        stage = getattr(exc, "stage", "yolo_inference")
        import traceback
        print(f"[ai-worker-error] stage={stage} error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise


def _process_frame_impl(
    frame_packet,
    detector,
    classifier,
    sequence_buffer,
    summary,
    args,
    post_processor=None,
    tracker=None,
    state=None,
    display_id_mapper=None,
    publisher=None,
    overlay_publish_state=None,
    frame_buffer=None,
    dropped_frame_count=None,
    roi_mask=None,
    exit_roi_mask=None,
    exit_post_processor=None,
    hazard_roi_mask=None,
    hazard_post_processor=None,
    track_selector=None,
    sync_sink=None,
    pose_reporter=None,
    incident_recovery=None,
    recovery_detect_fn=None,
):
    """RTSP frame 하나를 YOLO Pose -> tracking -> LSTM -> MQTT payload로 처리한다.

    핵심 순서:
    1. frame metadata를 기록해 frameId/timestampMs를 payload와 diagnostics에 맞춘다.
    2. ROI mask 적용 후 YOLO Pose raw detections를 얻고, mock 모드면 keypoint를 보강한다.
    3. ByteTrack/Simple tracker가 detection에 track_id를 붙인다.
    3b. Fall/Faint ROI recovery (전역 conf 변경 없이 crop-only) + track state 이관.
    4. per-track sequence buffer가 충분히 찬 track만 LSTM classifier로 보낸다.
    5. bbox/keypoint/tracking/LSTM 결과를 overlay payload와 진단 로그로 발행한다.
    """

    stream_id = getattr(args, "camera_login_id", None) or args.camera_id
    frame_metadata = None
    if frame_buffer is not None:
        packet_frame_id = getattr(frame_packet, "frame_id", None)
        if packet_frame_id is None:
            frame_metadata = frame_buffer.record_capture(stream_id, frame_packet, frame_packet.frame.shape)
        else:
            frame_metadata = frame_buffer.get_by_frame_id(stream_id, packet_frame_id)
        if frame_metadata is not None:
            summary["latest_frame_id"] = frame_metadata.frame_id
            summary["latest_captured_at_ms"] = frame_metadata.captured_at_ms
            summary["frame_sync_buffer_size"] = frame_buffer.size(stream_id)
    inference_frame = apply_roi_mask(frame_packet.frame, roi_mask)
    detections = detector.detect(inference_frame)
    if args.detector_mode == "mock":
        detections = ensure_mock_keypoints(detections)
    raw_detections = [dict(item) for item in detections]

    # Stage 1: detector 품질 확인용 raw detection 로그. ByteTrack 문제로 보기 전에
    # YOLO가 사람 bbox/keypoint를 안정적으로 잡는지 먼저 확인한다.
    _log_frame_id = frame_metadata.frame_id if frame_metadata is not None else getattr(frame_packet, "frame_idx", 0)
    _log_ts = frame_metadata.captured_at_ms if frame_metadata is not None else None
    log_detection_stage(stream_id, _log_frame_id, _log_ts, detections)

    _pre_track_detections = detections
    if tracker is not None:
        detections = update_detections_with_postprocessor(tracker, detections, frame_packet.frame, frame_packet.timestamp)

    # Stage 2b: Fall/Faint ROI recovery after tracking (no global conf/imgsz change).
    if incident_recovery is not None:
        frame_shape = getattr(frame_packet.frame, "shape", None)
        if frame_shape is not None and len(frame_shape) >= 2:
            detections = incident_recovery.on_tracked_frame(
                camera_login_id=stream_id,
                tracked=detections,
                timestamp=float(frame_packet.timestamp),
                frame_id=int(_log_frame_id),
                frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
                frame_bgr=frame_packet.frame,
                detect_roi_fn=recovery_detect_fn,
            )
            detections, recovery_migrations = finalize_recovery_detections(
                detections,
                camera_login_id=stream_id,
                incident_recovery=incident_recovery,
                tracker=tracker,
                now=float(frame_packet.timestamp),
                sequence_buffer=sequence_buffer,
                post_processor=post_processor,
                display_id_mapper=display_id_mapper,
                overlay_publish_state=overlay_publish_state,
            )
            if recovery_migrations:
                summary["incident_recovery_migrations"] = int(summary.get("incident_recovery_migrations", 0)) + len(
                    recovery_migrations
                )
                summary["last_incident_recovery_migration"] = recovery_migrations[-1]
            summary["incident_recovery"] = incident_recovery.diagnostics(stream_id)

    # Stage 2: tracking association 로그. raw detection은 있는데 active track이 0이면
    # detector가 아니라 tracker threshold/association 문제로 분류할 수 있다.
    _tracker_diag = tracker.diagnostics() if tracker is not None else {}
    log_tracking_stage(stream_id, _log_frame_id, _pre_track_detections, detections, _tracker_diag)
    log_track_lifecycle_events(stream_id, _log_frame_id, _tracker_diag)
    _emit_canary_tracking_signals(stream_id, _log_frame_id, detections, _tracker_diag, args)
    log_compact_tracking_debug(
        stream_id,
        _log_frame_id,
        _log_ts,
        _pre_track_detections,
        detections,
        _tracker_diag,
        worker_pid=os.getpid(),
        stream_run_id=str(getattr(args, "stream_run_id", "") or ""),
        every_n=15,
    )
    if os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}:
        frame_id = frame_metadata.frame_id if frame_metadata is not None else getattr(frame_packet, "frame_idx", 0)
        det_cnt = len(detections)
        tracked_cnt = sum(1 for item in detections if item.get("track_id") is not None)
        active_tids = sorted(list({int(item["track_id"]) for item in detections if item.get("track_id") is not None}))
        confs = [float(item.get("confidence", 0.0)) for item in detections]
        min_conf = min(confs) if confs else 0.0
        max_conf = max(confs) if confs else 0.0
        conf_range = f"{min_conf:.2f}-{max_conf:.2f}"
        matched_details = [f"det_{idx}->tid_{item.get('track_id')}" for idx, item in enumerate(detections)]
        keypoint_details = [
            f"tid_{item.get('track_id')}:kp_{len(item.get('keypoints') or [])}"
            for item in detections
        ]
        if len(_pre_track_detections) > 0 and tracked_cnt == 0:
            tracker_diagnosis = "tracker_gating"
        elif tracked_cnt > 0 and int(_tracker_diag.get("active_tracks", tracked_cnt)) == 0:
            tracker_diagnosis = "active_track_filtering"
        else:
            tracker_diagnosis = "tracking_ok"
        diag = _tracker_diag
        new_cnt = diag.get("new_tracks", 0)
        lost_cnt = diag.get("lost_tracks", 0)
        print(
            f"[Tracking Debug] camera: {stream_id} | frameId: {frame_id} | "
            f"detections: {det_cnt} | tracker_input_count: {len(_pre_track_detections)} | tracked: {tracked_cnt} | "
            f"new_tracks: {new_cnt} | lost_tracks: {lost_cnt} | "
            f"active_ids: {active_tids} | conf_range: {conf_range} | "
            f"keypoints: {keypoint_details} | diagnosis: {tracker_diagnosis} | mapping: {matched_details}",
            flush=True
        )


    from ai.inference.track_selection import deduplicate_tracked_detections
    detections, duplicate_skips = deduplicate_tracked_detections(detections, iou_threshold=0.85)

    selected_skips = []
    diag_info = {}
    if track_selector is not None:
        detections, selected_skips, diag_info = track_selector.filter(detections)
        summary["selected_track_id"] = track_selector.selected_track_id
        if selected_skips:
            summary["selected_track_skipped"] = summary.get("selected_track_skipped", 0) + len(selected_skips)
        if diag_info.get("fallback_active"):
            summary["selected_track_fallbacks"] = summary.get("selected_track_fallbacks", 0) + 1

        if os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}:
            print(
                f"[Selected Track Debug] camera: {stream_id} | "
                f"fallback_active: {diag_info.get('fallback_active')} | "
                f"fallback_track_id: {diag_info.get('fallback_track_id')} | "
                f"skipped_reason: {diag_info.get('skipped_reason')} | "
                f"missing_frames_count: {diag_info.get('missing_frames_count')}",
                flush=True
            )
    elif getattr(args, "selected_track_id", None) is not None:
        from ai.inference.track_selection import filter_selected_track
        detections, selected_skips = filter_selected_track(detections, args.selected_track_id)
        summary["selected_track_id"] = args.selected_track_id
        if selected_skips:
            summary["selected_track_skipped"] = summary.get("selected_track_skipped", 0) + len(selected_skips)


    boxes = normalize_detections(detections)
    frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
    active_tracks = len({int(item["track_id"]) for item in detections if item.get("track_id") is not None})
    summary["frames_processed"] += 1
    summary["bbox_detections"] += len(boxes)
    summary["keypoints_extracted"] += frame_keypoint_count
    summary["latest_frame_bbox"] = len(boxes)
    summary["latest_frame_keypoints"] = frame_keypoint_count
    summary["active_tracks"] = active_tracks
    summary["active_track_ids"] = sorted(list({int(item["track_id"]) for item in detections if item.get("track_id") is not None}))
    summary["max_active_tracks"] = max(summary.get("max_active_tracks", 0), active_tracks)
    if tracker is not None:
        update_tracking_summary(summary, tracker.diagnostics())

    # EXIT 이탈 감지: EXIT ROI = 안전 구역. 안전 구역에 들어갔다가 밖으로 나간 track만 알림.
    # ROI 안에 한 번도 없었던 사람을 계속 EXIT로 표시하지 않는다.
    if exit_roi_mask is not None and exit_post_processor is not None:
        all_track_ids = {int(float(str(b["track_id"]))) for b in boxes if b.get("track_id") is not None}
        in_safe_zone = find_boxes_in_exit_zone(boxes, exit_roi_mask)
        overlay_frame_id = frame_metadata.frame_id if frame_metadata is not None else getattr(frame_packet, "frame_idx", None)
        for track_id in all_track_ids:
            inside = track_id in in_safe_zone
            debug_before = {}
            if hasattr(exit_post_processor, "debug_state"):
                debug_before = exit_post_processor.debug_state(args.camera_id, track_id)
            should_fire = False
            if hasattr(exit_post_processor, "observe"):
                should_fire = exit_post_processor.observe(
                    args.camera_id,
                    track_id,
                    inside_safe_zone=inside,
                    timestamp=frame_packet.timestamp,
                )
            elif inside:
                exit_post_processor.reset_track(args.camera_id, track_id)
            else:
                should_fire = exit_post_processor.should_trigger(
                    args.camera_id, track_id, frame_packet.timestamp
                )
            exit_boxes = [
                b for b in boxes
                if b.get("track_id") is not None and int(float(str(b["track_id"]))) == track_id
            ]
            bbox_center = None
            if exit_boxes:
                box = exit_boxes[0]
                bbox_center = [
                    round((float(box.get("x1", 0)) + float(box.get("x2", 0))) / 2, 1),
                    round((float(box.get("y1", 0)) + float(box.get("y2", 0))) / 2, 1),
                ]
            if os.getenv("EXIT_DEBUG", "false").lower() in {"1", "true", "yes", "on"} or should_fire:
                print(
                    f"[exit-debug] cameraLoginId={stream_id} eventTrackId={track_id} "
                    f"overlayTrackId={track_id} eventFrameId={overlay_frame_id} "
                    f"overlayFrameId={overlay_frame_id} insideRoi={inside} "
                    f"previousInsideRoi={debug_before.get('wasInside')} "
                    f"exitStateAgeFrames={debug_before.get('outsideAgeFrames')} "
                    f"bboxCenter={bbox_center} fire={should_fire}",
                    flush=True,
                )
            if should_fire:
                exit_payload = build_inference_event_payload(
                    args, frame_packet,
                    {"label": "exit", "score": 1.0, "probabilities": {"exit": 1.0}},
                    exit_boxes, None,
                    frame_metadata=frame_metadata,
                    published_at_ms=None,
                    dropped_frame_count=dropped_frame_count,
                )
                topic_settings_exit = mqtt_topic_settings_from_args(args)
                if publisher is not None:
                    publisher.publish(exit_payload, topic=topic_settings_exit["event_topic"])
                print(f"[exit-event] {stream_id} track_id={track_id} frameId={overlay_frame_id}", flush=True)

    # HAZARD 위험구역 감지: 트래킹된 박스 center가 위험구역(HAZARD ROI) 안에 있으면 알림
    if hazard_roi_mask is not None and hazard_post_processor is not None:
        all_track_ids = {int(float(str(b["track_id"]))) for b in boxes if b.get("track_id") is not None}
        in_hazard_zone = find_boxes_in_hazard_zone(boxes, hazard_roi_mask)
        for track_id in all_track_ids - in_hazard_zone:
            hazard_post_processor.reset_track(args.camera_id, track_id)
        for track_id in in_hazard_zone:
            if hazard_post_processor.should_trigger(args.camera_id, track_id, frame_packet.timestamp):
                hazard_boxes = [b for b in boxes if b.get("track_id") is not None and int(float(str(b["track_id"]))) == track_id]
                hazard_payload = build_inference_event_payload(
                    args, frame_packet,
                    {"label": "hazard", "score": 1.0, "probabilities": {"hazard": 1.0}},
                    hazard_boxes, None,
                    frame_metadata=frame_metadata,
                    published_at_ms=None,
                    dropped_frame_count=dropped_frame_count,
                )
                topic_settings_hazard = mqtt_topic_settings_from_args(args)
                if publisher is not None:
                    publisher.publish(hazard_payload, topic=topic_settings_hazard["event_topic"])
                print(f"[hazard-event] {stream_id} track_id={track_id}", flush=True)

    # Update display ID mapping so operator labels stay compact (1, 2, 3…)
    if display_id_mapper is not None:
        active_raw_ids = {int(b["track_id"]) for b in boxes if b.get("track_id") is not None}
        display_id_mapper.update(active_raw_ids)
        for box in boxes:
            raw_id = box.get("track_id")
            if raw_id is not None:
                box["display_id"] = display_id_mapper.display_id(int(raw_id))
            if frame_metadata is not None:
                box["frameId"] = frame_metadata.frame_id
        summary["display_id_map"] = display_id_mapper.mapping_snapshot()
    elif frame_metadata is not None:
        for box in boxes:
            box["frameId"] = frame_metadata.frame_id

    classifier_input = getattr(args, "classifier_input", None)
    frame_id = frame_metadata.frame_id if frame_metadata is not None else None
    captured_at_ms = frame_metadata.captured_at_ms if frame_metadata is not None else None
    if classifier_input == "crops":
        sequences = sequence_buffer.add(
            frame_packet.frame_idx,
            frame_packet.frame,
            boxes,
            now=frame_packet.timestamp,
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
        )
    else:
        sequences = sequence_buffer.add(
            frame_packet.frame_idx,
            detections,
            frame_packet.frame.shape,
            now=frame_packet.timestamp,
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
        )
    buffer_lengths = sequence_buffer.buffer_lengths() if hasattr(sequence_buffer, "buffer_lengths") else {}
    prediction = None
    predictions_by_track = {}
    sequences_by_track = {}
    triggered_track_ids = set()
    consecutive_by_track = {}
    for sequence in sequences:
        sequence["camera_login_id"] = getattr(args, "camera_login_id", None) or args.camera_id
        prediction = classifier.predict(sequence)
        track_id = sequence.get("track_id")
        if track_id is not None:
            track_id = int(track_id)
            predictions_by_track[track_id] = prediction
            sequences_by_track[track_id] = sequence
        summary["generated_sequences"] += 1
        summary["lstm_predictions"] += 1
        summary["latest_prediction_label"] = prediction.get("label")
        summary["latest_faint_probability"] = faint_probability(prediction)
        summary["latest_runtime_feature_dim"] = getattr(classifier, "last_runtime_feature_dim", None)
        summary["latest_tensor_shape"] = getattr(classifier, "last_tensor_shape", None)
        summary["feature_schema"] = getattr(classifier, "feature_schema", None)
        update_prediction_counts(summary, prediction)
    summary["per_track_sequences_generated"] = {
        str(track_id): count for track_id, count in sequence_buffer.sequences_generated_by_track.items()
    }
    if hasattr(sequence_buffer, "sequence_completion_summary"):
        summary["sequence_completion"] = sequence_buffer.sequence_completion_summary()
    if pose_reporter is not None:
        summary["pose_diagnostics"] = pose_reporter.observe(
            camera_login_id=stream_id,
            source_url=str(getattr(args, "rtsp_url", "")),
            assigned_video_path=str(getattr(args, "assigned_video_path", "") or ""),
            frame_id=_log_frame_id,
            timestamp_ms=_log_ts,
            raw_detections=raw_detections,
            tracker_diagnostics=_tracker_diag,
            sequence_ready_count=len(sequences),
            sequence_diagnostics={
                "relink_success_count": sequence_buffer.relink_success_count,
                "relink_fail_count": sequence_buffer.relink_failure_count,
            },
            frame=frame_packet.frame,
        )
    log_sequence_stage(
        stream_id,
        _log_frame_id,
        active_track_ids=sequence_buffer.active_track_ids(),
        buffer_lengths=buffer_lengths,
        sequences_generated=len(sequences),
        sequences_generated_by_track=sequence_buffer.sequences_generated_by_track,
        latest_faint_prob=summary.get("latest_faint_probability"),
        sequence_diagnostics=sequence_buffer.sequence_diagnostics() if hasattr(sequence_buffer, "sequence_diagnostics") else {},
        checkpoint_input_size=getattr(classifier, "input_size", None),
        runtime_feature_dim=getattr(classifier, "last_runtime_feature_dim", None),
        tensor_shape=getattr(classifier, "last_tensor_shape", None),
        sequence_length=getattr(args, "sequence_length", None),
    )
    from ai.action.posture_estimator import detection_for_track

    emit_decisions_by_track = {}
    for track_id, track_prediction in predictions_by_track.items():
        event_triggered = False
        if post_processor is not None:
            track_detection = detection_for_track(detections, track_id) or detection_for_track(boxes, track_id)
            emit_decision = post_processor.evaluate(
                args.camera_id,
                track_prediction,
                frame_packet.timestamp,
                track_id=track_id,
                detection=track_detection,
            )
            emit_decisions_by_track[track_id] = emit_decision
            event_triggered = bool(emit_decision.emit)
            consecutive_by_track[track_id] = post_processor.consecutive_count(args.camera_id, track_id=track_id)
            # Register recovery context on internal FALL/FAINT suspected entry
            # (not only MQTT NEW_FALL / UNRECOVERED emit).
            if incident_recovery is not None:
                _maybe_register_incident_recovery(
                    incident_recovery=incident_recovery,
                    camera_login_id=stream_id,
                    track_id=track_id,
                    emit_decision=emit_decision,
                    post_processor=post_processor,
                    track_detection=track_detection,
                    timestamp=float(frame_packet.timestamp),
                    frame_id=int(_log_frame_id),
                )
        elif track_prediction and track_prediction.get("label") != "Normal":
            event_triggered = True
            consecutive_by_track[track_id] = 1
        if event_triggered:
            triggered_track_ids.add(track_id)
            track_key = str(track_id)
            summary["events_generated_by_track"][track_key] = summary["events_generated_by_track"].get(track_key, 0) + 1
        # Stage 3: Classification log per track
        log_classification_stage(
            stream_id,
            _log_frame_id,
            track_id,
            track_prediction,
            faint_threshold=getattr(args, "action_threshold", 0.5),
            consecutive_count=consecutive_by_track.get(track_id, 0),
            event_triggered=event_triggered,
            checkpoint_input_size=getattr(classifier, "input_size", None),
            runtime_feature_dim=getattr(classifier, "last_runtime_feature_dim", None),
            tensor_shape=getattr(classifier, "last_tensor_shape", None),
            feature_schema=getattr(classifier, "feature_schema", None),
            checkpoint_path=getattr(classifier, "checkpoint_path", None),
        )

    summary["latest_consecutive_faint"] = max(consecutive_by_track.values(), default=0)
    annotate_boxes_with_track_actions(boxes, predictions_by_track, consecutive_by_track, triggered_track_ids, args)
    topic_settings = mqtt_topic_settings_from_args(args)
    frame_width, frame_height = frame_size_from_shape(frame_packet.frame.shape)
    if frame_buffer is not None and frame_metadata is not None:
        frame_metadata = frame_buffer.mark_processed(stream_id, frame_metadata.frame_id)
        summary["latest_processed_at_ms"] = frame_metadata.processed_at_ms
        summary["latest_ai_latency_ms"] = frame_metadata.ai_latency_ms
    timestamp_ms = None
    published_at_ms = None
    if overlay_publish_state is not None:
        overlay_publish_state.apply_latest_signals(boxes)
        timestamp_ms = overlay_publish_state.next_timestamp_ms()
    if frame_buffer is not None and frame_metadata is not None:
        frame_metadata = frame_buffer.mark_published(stream_id, frame_metadata.frame_id)
        published_at_ms = frame_metadata.published_at_ms
        timestamp_ms = published_at_ms
        summary["latest_published_at_ms"] = published_at_ms
        summary["latest_publish_latency_ms"] = frame_metadata.publish_latency_ms
        evidence_key = evidence_id(stream_id, frame_metadata.frame_id, frame_metadata.captured_at_ms)
        log_frame_sync(args, stream_id, frame_metadata, frame_buffer, int(dropped_frame_count or 0))
        summary["latest_evidence_id"] = evidence_key
        summary["latest_trace_id"] = evidence_key
        summary["latest_dropped_frame_count"] = int(dropped_frame_count or 0)
        summary["latest_latency_order_valid"] = frame_metadata.latency_order_valid
    overlay_payload = build_overlay_payload(
        stream_id=stream_id,
        frame_width=frame_width,
        frame_height=frame_height,
        boxes=boxes,
        timestamp_ms=timestamp_ms,
        frame_id=getattr(frame_metadata, "frame_id", None),
        captured_at_ms=getattr(frame_metadata, "captured_at_ms", None),
        processed_at_ms=getattr(frame_metadata, "processed_at_ms", None),
        published_at_ms=published_at_ms,
        dropped_frame_count=dropped_frame_count,
    )
    # Stage 4: Payload log + quantitative metrics update
    _payload_frame_id = getattr(frame_metadata, "frame_id", None) or _log_frame_id
    log_payload_stage(stream_id, _payload_frame_id, overlay_payload)
    update_quantitative_summary(summary, boxes, _tracker_diag if tracker is not None else None)
    summary["latest_overlay_event_count"] = len(overlay_payload["events"])
    if sync_sink is not None:
        try:
            sync_sink.publish_frame(frame_packet.frame, overlay_payload)
        except Exception as exc:
            print(
                f"[ai-worker][error] failed to publish WebRTC sync frame for camera={stream_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if publisher is not None:
        try:
            overlay_publish_result = publisher.publish(overlay_payload, topic=topic_settings["camera_topic"])
            _record_publish_outcome(summary, "overlay_publish", overlay_publish_result, attempted=True)
        except Exception as exc:
            _record_publish_outcome(summary, "overlay_publish", attempted=True)
            print(f"[ai-worker][error] failed to publish overlay payload for camera={stream_id}: {exc}", file=sys.stderr, flush=True)
    for track_id in triggered_track_ids:
        track_prediction = predictions_by_track[track_id]
        sequence = sequences_by_track[track_id]
        emit_decision = emit_decisions_by_track.get(track_id)
        payload = build_inference_event_payload(
            args,
            frame_packet,
            track_prediction,
            boxes,
            sequence,
            frame_metadata=frame_metadata,
            published_at_ms=published_at_ms,
            dropped_frame_count=dropped_frame_count,
            **lifecycle_payload_kwargs(emit_decision),
        )
        log_lstm_event(args, stream_id, sequence, track_prediction)
        summary["events_generated"] += 1
        if emit_decision is not None and emit_decision.is_unrecovered:
            summary["unrecovered_events_generated"] = int(summary.get("unrecovered_events_generated", 0)) + 1
        if summary["sample_event"] is None:
            summary["sample_event"] = payload
        if args.print_events:
            print(f"[ai-overlay-event] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        if publisher is not None:
            try:
                event_publish_result = _publish_event(publisher, payload, topic_settings["event_topic"])
                _record_publish_outcome(summary, "events_publish", event_publish_result, attempted=True)
                # Single VLM snapshot assist hook (never blocks alert loop)
                try:
                    from ai.snapshot_assist_upload import submit_frame_snapshot_async
                    submit_frame_snapshot_async(
                            event_id=str(payload.get("eventId") or ""),
                            camera_login_id=str(stream_id),
                            frame=getattr(frame_packet, "frame", None),
                        )
                except Exception as snap_exc:
                    print(f"[snapshot-assist][warn] non-fatal upload schedule failed: {snap_exc}", flush=True)
            except Exception as exc:
                _record_publish_outcome(summary, "events_publish", attempted=True)
                print(f"[ai-worker][error] failed to publish event payload for camera={stream_id}: {exc}", file=sys.stderr, flush=True)
        # 낙상 감지 시 10초 스냅샷 버퍼 트리거 작동
        if state is not None and getattr(state, "clip_buffer", None) is not None:
            target_bbox = []
            for b in boxes:
                if b.get("track_id") is not None and int(b["track_id"]) == track_id:
                    x1, y1, x2, y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
                    if None not in (x1, y1, x2, y2):
                        target_bbox = [x1, y1, x2, y2]
                    break
            
            task_metadata = {
                "evidenceId": payload.get("eventId"),
                "event_timestamp": payload.get("timestamp"),
                "track_id": track_id,
                "bbox": target_bbox
            }
            
            clip_triggered = state.clip_buffer.trigger_event(
                event_type=payload.get("type", "fall_detected"),
                camera_id=stream_id,
                metadata=task_metadata,
            )
            if clip_triggered:
                print(f"[ai-overlay-event] triggered snapshot recording for camera={stream_id} eventId={payload.get('eventId')}", flush=True)
            else:
                print(f"[ai-overlay-event] triggered snapshot recording SKIPPED (max concurrent events reached or in cooldown) for camera={stream_id} eventId={payload.get('eventId')}", flush=True)
    maybe_log_debug(frame_packet, boxes, summary, prediction, args, prefix="[ai-overlay-debug]")

    update_overlay_runtime(summary)
    # MJPEG 서버가 켜져 있으면 항상 프레임을 반환해야 검은화면이 발생하지 않는다.
    if not mjpeg_server_enabled(args):
        return None
    # overlay drawing이 꺼져 있으면 raw frame을 그대로 반환한다.
    if not mjpeg_overlay_enabled(args):
        return frame_packet.frame.copy()
    # overlay가 켜져 있으면 bbox/keypoint/status text를 그린 annotated frame을 반환한다.
    try:
        overlay_frame_id = frame_metadata.frame_id if frame_metadata is not None else frame_packet.frame_idx
        overlay = draw_overlay(frame_packet.frame, boxes, prediction, overlay_frame_id)
        draw_metrics_panel(overlay, summary, args, prediction)
        return overlay
    except Exception as draw_exc:  # overlay 그리기 실패 시 원본 프레임 fallback (worker 중단 방지)
        print(
            f"[mjpeg-overlay][warn] overlay draw failed, falling back to raw frame: {draw_exc}",
            flush=True,
        )
        return frame_packet.frame.copy()


def log_frame_sync(args, stream_id, frame_metadata, frame_buffer, dropped_frame_count=0):
    every_n = max(0, int(getattr(args, "debug_every_n", 30)))
    warning_ms = max(0, int(getattr(args, "frame_sync_delay_warning_ms", 300)))
    publish_latency_ms = frame_metadata.publish_latency_ms
    if not frame_metadata.latency_order_valid:
        print(
            "[frame-sync] warning "
            f"{stream_id} "
            f"latency_order_invalid=true "
            f"latency_order_valid=false "
            f"frame_id={frame_metadata.frame_id} "
            f"captured_at_ms={frame_metadata.captured_at_ms} "
            f"processed_at_ms={frame_metadata.processed_at_ms} "
            f"published_at_ms={frame_metadata.published_at_ms}",
            flush=True,
        )
    if publish_latency_ms is not None and warning_ms > 0 and publish_latency_ms > warning_ms:
        print(
            "[frame-sync] warning "
            f"{stream_id} "
            f"overlay_delay_ms={publish_latency_ms} "
            f"frame_id={frame_metadata.frame_id}",
            flush=True,
        )
    if every_n <= 0 or frame_metadata.frame_id % every_n != 0:
        return
    print(
        "[frame-sync] "
        f"{stream_id} "
        f"frame_id={frame_metadata.frame_id} "
        f"captured_at_ms={frame_metadata.captured_at_ms} "
        f"ai_latency_ms={frame_metadata.ai_latency_ms} "
        f"publish_latency_ms={frame_metadata.publish_latency_ms} "
        f"dropped_frame_count={int(dropped_frame_count)} "
        f"evidence_id={stream_id}-{frame_metadata.frame_id}-{frame_metadata.captured_at_ms} "
        f"buffer_size={frame_buffer.size(stream_id)}",
        flush=True,
    )


def log_lstm_event(args, stream_id, sequence, prediction):
    probability = prediction.get("score")
    print(
        "[lstm-event] "
        f"{stream_id} "
        f"sequence={sequence.get('sequence_start_frame_id')}-{sequence.get('sequence_end_frame_id')} "
        f"length={getattr(args, 'sequence_length', '')} "
        f"stride={getattr(args, 'sequence_stride', '')} "
        f"event={prediction.get('label')} "
        f"prob={probability}",
        flush=True,
    )


class OverlayWorker:
    def __init__(self, args, state, sync_sink=None):
        self.args = args
        self.state = state
        self.sync_sink = sync_sink
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ai-overlay-worker", daemon=True)
        self.reader_thread = None
        self.camera_login_id = getattr(self.args, "camera_login_id", self.args.camera_id) or self.args.camera_id
        self.queue = CameraFrameQueue(self.camera_login_id, maxsize=getattr(self.args, "frame_queue_maxsize", 3))
        self.frame_buffer = FrameMetadataBuffer(maxlen=self.args.frame_sync_buffer_size)
        self.fps_audit = FpsAuditWindow(
            camera_login_id=self.camera_login_id,
            stream_run_id=str(getattr(self.args, "stream_run_id", "") or ""),
            window_sec=10.0,
            tracker_config_frame_rate=float(getattr(self.args, "frame_rate", 30) or 30),
            mjpeg_target_fps=float(getattr(self.args, "mjpeg_fps", 8) or 8),
        )
        self._resolution_audited = False

        # 10초 스냅샷 비디오 클립 버퍼 및 큐 초기화 (state에 공유하여 process_frame에서도 접근 가능케 함)
        self.state.clip_queue = queue.Queue(maxsize=10)
        self.state.clip_buffer = EventClipBuffer()
        self.clip_worker = None

    def start(self):
        self.reader_thread = threading.Thread(target=self._reader_run, name="ai-overlay-reader", daemon=True)
        self.reader_thread.start()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=3)
        self.thread.join(timeout=3)

        # 스냅샷 비디오 클립 워커 종료
        if hasattr(self, "clip_worker") and self.clip_worker is not None:
            self.clip_worker.stop()
            print(f"[ai-overlay-inference] stopped clip writer worker for camera={self.camera_login_id}", flush=True)

    def _reader_run(self):
        publisher = None
        status_publisher = None
        reconnect_count = 0
        try:
            publisher, publisher_mode = create_event_publisher(self.args, role="status")
            status_publisher = CameraStatusPublisher(
                mqtt_publisher=publisher,
                camera_login_id=self.camera_login_id,
                rtsp_url=self.args.rtsp_url,
                status_topic=getattr(self.args, "mqtt_status_topic", None),
            )
            while not self.stop_event.is_set():
                try:
                    with VideoReader(self.args.rtsp_url) as reader:
                        print(f"[ai-overlay-reader] connected: {redact_url(self.args.rtsp_url)}", flush=True)
                        status_publisher.notify_connected()
                        frame_count = 0
                        last_heartbeat_time = time.time()
                        while not self.stop_event.is_set():
                            packet = reader.read()
                            if packet is None:
                                status_publisher.notify_disconnected(reason="STREAM_ENDED")
                                break
                            
                            frame_metadata = self.frame_buffer.record_capture(
                                self.camera_login_id,
                                packet,
                                packet.frame.shape
                            )
                            packet_wrapped = FramePacket(
                                camera_login_id=self.camera_login_id,
                                frame_id=frame_metadata.frame_id,
                                captured_at_ms=frame_metadata.captured_at_ms,
                                frame=packet.frame,
                                width=frame_metadata.width,
                                height=frame_metadata.height,
                                frame_idx=packet.frame_idx,
                                timestamp=packet.timestamp,
                                fps=getattr(packet, "fps", 0.0)
                            )
                            dropped_before = int(getattr(self.queue, "dropped_frame_count", 0) or 0)
                            self.queue.put_latest(packet_wrapped)
                            dropped_after = int(getattr(self.queue, "dropped_frame_count", 0) or 0)
                            self.fps_audit.source_fps = float(getattr(packet, "fps", 0.0) or 0.0) or self.fps_audit.source_fps
                            self.fps_audit.note_capture(queue_dropped=max(0, dropped_after - dropped_before))
                            
                            frame_count += 1
                            now = time.time()
                            if now - last_heartbeat_time >= 1.0:
                                print(
                                    f"[heartbeat-reader] camera={self.camera_login_id} "
                                    f"read_count={frame_count} "
                                    f"fps={frame_count / (now - last_heartbeat_time):.1f} "
                                    f"queue_size={self.queue.size()}",
                                    flush=True
                                )
                                frame_count = 0
                                last_heartbeat_time = now

                            every_n = max(0, int(getattr(self.args, "debug_every_n", 30)))
                            if every_n > 0 and frame_metadata.frame_id % every_n == 0:
                                now_ms = time.time_ns() // 1_000_000
                                lag = now_ms - frame_metadata.captured_at_ms
                                print(
                                    f"[rtsp-buffer] {self.camera_login_id} "
                                    f"frame_id={frame_metadata.frame_id} "
                                    f"queue_lag_ms={lag} "
                                    f"dropped={self.queue.dropped_frame_count}",
                                    flush=True
                                )
                except Exception as exc:
                    message = f"Reader error {type(exc).__name__}: {exc}"
                    print(f"[ai-overlay-reader] {message}", file=sys.stderr, flush=True)
                    if status_publisher is not None:
                        status_publisher.notify_error(reason=type(exc).__name__)
                
                if not self.stop_event.is_set():
                    reconnect_count += 1
                    print(f"[ai-overlay-reader] camera={self.camera_login_id} reconnecting (count={reconnect_count}) after error/end", flush=True)
                    if status_publisher is not None:
                        status_publisher.notify_reconnecting()
                    time.sleep(self.args.reconnect_delay)
        finally:
            close = getattr(publisher, "close", None) if publisher is not None else None
            if close:
                close()

    def _run(self):
        action_threshold = resolved_faint_threshold(self.args)
        detector = create_detector(self.args.detector_mode, self.args.yolo_model, self.args.device, self.args.imgsz, conf=self.args.detector_conf)
        classifier, _classifier_mode = create_classifier(self.args.action_model, self.args.action_device, action_threshold)
        publisher, publisher_mode = create_event_publisher(self.args, role="inference")
        publisher = AsyncMqttDelivery(
            publisher,
            outbox_path=event_outbox_path(self.camera_login_id) if publisher_mode == "mqtt" else None,
        )
        print(f"[ai-overlay-inference] initialized event publisher: {publisher_mode}", flush=True)
        log_worker_backend_startup(
            camera_login_id=self.camera_login_id,
            requested_model=self.args.yolo_model,
            detector=detector,
            device=getattr(self.args, "device", None),
        )
        yolo_metrics = RuntimeMetrics()
        yolo_metrics.set_warmup_skip(int(getattr(self.args, "infer_warmup_frames", 20)))
        
        # S3 업로더 및 MQTT 발행 연동 스레드 시작
        self.clip_worker = ClipWriterWorker(
            self.state.clip_queue,
            publisher=publisher,
            mqtt_event_topic=self.args.mqtt_event_topic
        )
        self.clip_worker.start()
        print(f"[ai-overlay-inference] started clip writer worker for camera={self.camera_login_id}", flush=True)
        log_lstm_config(
            "[lstm-config]",
            self.args.sequence_length,
            self.args.sequence_stride,
            getattr(classifier, "input_size", DEFAULT_KEYPOINT_INPUT_SIZE),
            f"checkpoint/config/cli:{_classifier_mode}",
            getattr(classifier, "checkpoint_sequence_length", None),
            getattr(classifier, "checkpoint_sequence_stride", None),
        )
        log_worker_startup_contract(self.args)
        log_classifier_contract("[lstm-checkpoint]", self.camera_login_id, classifier)

        summary = initial_summary()
        post_processor = build_faint_post_processor_from_args(self.args)
        tracker, postprocessing_mode = create_detection_postprocessor(self.args)
        log_pose_tracking_config(self.args, tracker)
        cheap_filter_config = cheap_filter_config_from_args(self.args)
        pose_reporter = PoseDiagnosticsReporter(pose_diagnostics_config_from_args(self.args))
        print(f"[ai-overlay-inference] tracking postprocessor: {postprocessing_mode}", flush=True)
        display_id_mapper = DisplayIdMapper()
        overlay_publish_state = OverlayPublishState()
        incident_recovery = IncidentRecoveryManager()
        recovery_detect_fn = None
        if getattr(self.args, "detector_mode", "real") != "mock" and callable(getattr(detector, "detect", None)):
            try:
                recovery_detect_fn = make_detect_roi_fn_from_yolo_pose(detector)
            except Exception:
                recovery_detect_fn = None
        from ai.inference.track_selection import TrackSelector
        track_selector = TrackSelector(
            selected_track_id=getattr(self.args, "selected_track_id", None),
            selected_track_mode=getattr(self.args, "selected_track_mode", "strict"),
            missing_frames_threshold=getattr(self.args, "selected_track_missing_frames", 5),
        )
        camera_ids = [self.camera_login_id]
        camera_index = 0

        roi_configs = getattr(self.args, "roi_configs_parsed", [])
        cached_roi_mask = None
        cached_roi_frame_shape = None
        exit_roi_configs = getattr(self.args, "exit_roi_configs_parsed", [])
        cached_exit_mask = None
        cached_exit_frame_shape = None
        exit_post_processor = ExitEventPostProcessor(
            min_consecutive=getattr(self.args, "exit_min_consecutive", DEFAULT_EXIT_MIN_CONSECUTIVE),
            cooldown_seconds=getattr(self.args, "exit_cooldown_seconds", DEFAULT_EXIT_COOLDOWN_SECONDS),
        )

        hazard_roi_configs = getattr(self.args, "hazard_roi_configs_parsed", [])
        cached_hazard_mask = None
        cached_hazard_frame_shape = None
        hazard_post_processor = HazardEventPostProcessor(
            min_consecutive=getattr(self.args, "hazard_min_consecutive", DEFAULT_HAZARD_MIN_CONSECUTIVE),
            cooldown_seconds=getattr(self.args, "hazard_cooldown_seconds", DEFAULT_HAZARD_COOLDOWN_SECONDS),
        )

        sequence_buffers = {}
        for cid in camera_ids:
            if self.args.classifier_input == "crops":
                sequence_buffers[cid] = PerTrackCropSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    self.args.resize_size,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                )
            else:
                sequence_buffers[cid] = PerTrackKeypointSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                    cheap_filter_config=cheap_filter_config,
                    missing_track_grace_seconds=getattr(self.args, "tracking_grace_period_seconds", self.args.track_max_missing_seconds),
                    relink_iou_threshold=getattr(self.args, "tracking_relink_iou_threshold", 0.30),
                    relink_center_distance_ratio=getattr(self.args, "tracking_relink_center_ratio", 0.70),
                    relink_max_time_gap_seconds=getattr(self.args, "tracking_relink_max_time_gap_seconds", 2.0),
                )

        last_heartbeat_time = time.monotonic()
        inference_count = 0
        tracker_source_initialized = False
        tracker_timebase_estimator = None
        tracker_timebase_last_applied_at = 0.0
        mqtt_publish_count = 0
        last_frame_id = None
        last_captured_at_ms = None

        while not self.stop_event.is_set():
            if not camera_ids:
                time.sleep(0.01)
                continue
            current_cam_id = camera_ids[camera_index]
            camera_index = (camera_index + 1) % len(camera_ids)

            if current_cam_id != self.camera_login_id:
                continue

            frame_packet = self.queue.get_latest()
            if frame_packet is None:
                time.sleep(0.005)
                continue

            if not tracker_source_initialized:
                source_fps = getattr(frame_packet, "fps", None)
                tracker, postprocessing_mode = create_detection_postprocessor(self.args, source_fps=source_fps)
                tracker_source_initialized = True
                tracker_timebase_estimator = TrackerUpdateFpsEstimator(
                    source_fps,
                    getattr(self.args, "frame_rate", 30),
                    latest_frame_mode=True,
                )
                summary["tracker_effective_fps"] = tracker_configured_fps(tracker)
                summary["trackerFpsState"] = tracker_timebase_estimator.state
                summary["trackerFpsSource"] = tracker_timebase_estimator.source
                self.fps_audit.tracker_config_frame_rate = tracker_configured_fps(tracker)

            # RTSP reconnect / large frame gap tracker reset check
            frame_gap = None
            if last_frame_id is not None and frame_packet.frame_idx is not None:
                frame_gap = frame_packet.frame_idx - last_frame_id
            
            time_gap = None
            if last_captured_at_ms is not None:
                time_gap = frame_packet.captured_at_ms - last_captured_at_ms
            
            reset_reason = session_reset_reason(
                last_frame_id,
                frame_packet.frame_idx,
                last_captured_at_ms,
                frame_packet.captured_at_ms,
            )
            reset_decided = reset_reason is not None

            if reset_decided:
                source_fps = getattr(frame_packet, "fps", None)
                tracker, postprocessing_mode = create_detection_postprocessor(self.args, source_fps=source_fps)
                tracker_source_initialized = True
                tracker_timebase_estimator = TrackerUpdateFpsEstimator(
                    source_fps,
                    getattr(self.args, "frame_rate", 30),
                    latest_frame_mode=True,
                )
                tracker_timebase_last_applied_at = 0.0
                summary["tracker_effective_fps"] = tracker_configured_fps(tracker)
                if self.args.classifier_input == "crops":
                    sequence_buffers[current_cam_id] = PerTrackCropSequenceBuffers(
                        self.args.sequence_length,
                        self.args.sequence_stride,
                        self.args.resize_size,
                        max_track_age_seconds=self.args.track_max_missing_seconds,
                    )
                else:
                    sequence_buffers[current_cam_id] = PerTrackKeypointSequenceBuffers(
                        self.args.sequence_length,
                        self.args.sequence_stride,
                        max_track_age_seconds=self.args.track_max_missing_seconds,
                        cheap_filter_config=cheap_filter_config,
                        missing_track_grace_seconds=getattr(self.args, "tracking_grace_period_seconds", self.args.track_max_missing_seconds),
                        relink_iou_threshold=getattr(self.args, "tracking_relink_iou_threshold", 0.30),
                        relink_center_distance_ratio=getattr(self.args, "tracking_relink_center_ratio", 0.70),
                        relink_max_time_gap_seconds=getattr(self.args, "tracking_relink_max_time_gap_seconds", 2.0),
                    )
                post_processor = build_faint_post_processor_from_args(self.args)
                exit_post_processor = ExitEventPostProcessor(
                    min_consecutive=getattr(self.args, "exit_min_consecutive", DEFAULT_EXIT_MIN_CONSECUTIVE),
                    cooldown_seconds=getattr(self.args, "exit_cooldown_seconds", DEFAULT_EXIT_COOLDOWN_SECONDS),
                )
                hazard_post_processor = HazardEventPostProcessor(
                    min_consecutive=getattr(self.args, "hazard_min_consecutive", DEFAULT_HAZARD_MIN_CONSECUTIVE),
                    cooldown_seconds=getattr(self.args, "hazard_cooldown_seconds", DEFAULT_HAZARD_COOLDOWN_SECONDS),
                )
                display_id_mapper = DisplayIdMapper()
                overlay_publish_state = OverlayPublishState()
                # EOF / reconnect / video boundary: clear recovery + migration context.
                incident_recovery.reset_camera(self.camera_login_id)
                summary["video_state_resets"] = summary.get("video_state_resets", 0) + 1
                summary["last_video_reset_reason"] = reset_reason
                print(
                    f"[video-boundary] camera={self.camera_login_id} reset={reset_reason}",
                    flush=True,
                )
                from ai.inference.tracking_debug import log_tracker_reset_decision, build_tracker_reset_record
                record = build_tracker_reset_record(
                    camera_login_id=self.camera_login_id,
                    frame_id=frame_packet.frame_id,
                    frame_gap=frame_gap,
                    tracker_object_id=id(tracker),
                    reset=True,
                    reason=reset_reason,
                )
                log_tracker_reset_decision(record)
            elif os.getenv("TRACKING_DEBUG", "false").lower() in {"1", "true", "yes", "on"}:
                from ai.inference.tracking_debug import log_tracker_reset_decision, build_tracker_reset_record
                status_reason = "FRAME_GAP_OK" if frame_gap is not None else "NO_PREVIOUS_FRAME"
                record = build_tracker_reset_record(
                    camera_login_id=self.camera_login_id,
                    frame_id=frame_packet.frame_id,
                    frame_gap=frame_gap,
                    tracker_object_id=id(tracker),
                    reset=False,
                    reason=status_reason,
                )
                log_tracker_reset_decision(record)

            last_frame_id = frame_packet.frame_idx
            last_captured_at_ms = frame_packet.captured_at_ms

            # 매 프레임마다 스냅샷 클립 버퍼에 기록
            if self.state.clip_buffer is not None:
                for clip_task in self.state.clip_buffer.add_frame(frame_packet.frame):
                    enqueue_event_clip(self.state.clip_queue, clip_task)
            if roi_configs:
                h, w = frame_packet.frame.shape[:2]
                if cached_roi_mask is None or cached_roi_frame_shape != (h, w):
                    cached_roi_mask = combine_roi_masks(roi_configs, h, w)
                    cached_roi_frame_shape = (h, w)
                    print(
                        f"[ai-overlay][roi] mask built: {len(roi_configs)} region(s) frame={w}x{h}",
                        flush=True,
                    )

            if exit_roi_configs:
                h, w = frame_packet.frame.shape[:2]
                if cached_exit_mask is None or cached_exit_frame_shape != (h, w):
                    cached_exit_mask = combine_roi_masks(exit_roi_configs, h, w)
                    cached_exit_frame_shape = (h, w)
                    print(
                        f"[ai-overlay][exit-roi] mask built: {len(exit_roi_configs)} region(s) frame={w}x{h}",
                        flush=True,
                    )

            if hazard_roi_configs:
                h, w = frame_packet.frame.shape[:2]
                if cached_hazard_mask is None or cached_hazard_frame_shape != (h, w):
                    cached_hazard_mask = combine_roi_masks(hazard_roi_configs, h, w)
                    cached_hazard_frame_shape = (h, w)
                    print(
                        f"[ai-overlay][hazard-roi] mask built: {len(hazard_roi_configs)} region(s) frame={w}x{h}",
                        flush=True,
                    )

            inference_start = time.perf_counter()
            overlay = process_frame(
                frame_packet, detector, classifier, sequence_buffers[current_cam_id], summary, self.args,
                post_processor=post_processor, tracker=tracker,
                state=self.state, display_id_mapper=display_id_mapper,
                publisher=publisher,
                overlay_publish_state=overlay_publish_state,
                frame_buffer=self.frame_buffer,
                dropped_frame_count=self.queue.dropped_frame_count,
                roi_mask=cached_roi_mask,
                exit_roi_mask=cached_exit_mask,
                exit_post_processor=exit_post_processor,
                hazard_roi_mask=cached_hazard_mask,
                hazard_post_processor=hazard_post_processor,
                track_selector=track_selector,
                sync_sink=self.sync_sink,
                pose_reporter=pose_reporter,
                incident_recovery=incident_recovery,
                recovery_detect_fn=recovery_detect_fn,
            )
            # Analysis path counts: process_frame always runs detector+tracker for this frame.
            self.fps_audit.note_analysis()
            self.fps_audit.note_detector()
            self.fps_audit.note_tracker_update()
            if tracker_timebase_estimator is not None:
                tracker_timebase_estimator.observe_update(frame_packet.timestamp)
            if not self._resolution_audited and frame_packet.frame is not None:
                h, w = frame_packet.frame.shape[:2]
                print(
                    f"[resolution-audit]\n"
                    f"cameraLoginId={self.camera_login_id}\n"
                    f"frameId={frame_packet.frame_id}\n"
                    f"sourceShape={w}x{h}\n"
                    f"modelRequestedImgsz={getattr(self.args, 'imgsz', getattr(detector, 'imgsz', 640))}\n"
                    f"modelPath={getattr(detector, 'model_path', getattr(detector, 'model_name', ''))}\n"
                    f"coordinatePolicy=ultralytics_boxes.xyxy_and_keypoints.xy_are_source_space\n"
                    f"overlayCanvas={getattr(self.args, 'mjpeg_width', w)}x{getattr(self.args, 'mjpeg_height', h)}\n"
                    f"mjpegEncodedSize={getattr(self.args, 'mjpeg_width', w)}x{getattr(self.args, 'mjpeg_height', h)}\n"
                    f"mjpegTargetFps={getattr(self.args, 'mjpeg_fps', None)}\n"
                    f"trackerConfigFrameRate={getattr(self.args, 'frame_rate', None)}",
                    flush=True,
                )
                self._resolution_audited = True
            mjpeg_count = None
            try:
                mjpeg_count = int(self.state.status().get("mjpeg_frame_count") or 0)
            except Exception:
                mjpeg_count = None
            audit = self.fps_audit.maybe_emit(mjpeg_frame_count=mjpeg_count)
            if audit is not None:
                if tracker_timebase_estimator is not None:
                    tracker_timebase_estimator.record_window(audit.get("trackerUpdateFps"))
                    current_fps = tracker_configured_fps(tracker)
                    now_monotonic = time.monotonic()
                    if (
                        tracker_timebase_estimator.should_apply(current_fps)
                        and now_monotonic - tracker_timebase_last_applied_at >= 10.0
                        and apply_tracker_timebase(tracker, tracker_timebase_estimator.effective_fps)
                    ):
                        tracker_timebase_last_applied_at = now_monotonic
                        self.fps_audit.tracker_config_frame_rate = tracker_configured_fps(tracker)
                    summary["tracker_effective_fps"] = tracker_configured_fps(tracker)
                    summary["trackerFpsState"] = tracker_timebase_estimator.state
                    summary["trackerConfiguredFps"] = tracker_configured_fps(tracker)
                    summary["trackerMeasuredFps"] = tracker_timebase_estimator.measured_fps
                    summary["trackerFpsSource"] = tracker_timebase_estimator.source
                    summary["trackerFpsSampleCount"] = tracker_timebase_estimator.sample_count
                log_fps_audit(audit)

            now_ms = time.time_ns() // 1_000_000
            queue_lag_ms = now_ms - frame_packet.captured_at_ms
            published_at_ms = summary.get("latest_published_at_ms") or now_ms

            fs_payload = build_frame_sync_payload(
                camera_login_id=current_cam_id,
                frame_id=frame_packet.frame_id,
                captured_at_ms=frame_packet.captured_at_ms,
                published_at_ms=published_at_ms,
                queue_lag_ms=queue_lag_ms,
                dropped_frame_count=self.queue.dropped_frame_count,
                processed_at_ms=summary.get("latest_processed_at_ms"),
            )
            topic_settings = mqtt_topic_settings_from_args(self.args)
            if publisher is not None:
                try:
                    publisher.publish(fs_payload, topic=topic_settings["camera_topic"])
                    mqtt_publish_count += 1
                except Exception as exc:
                    print(
                        f"[ai-worker][error] failed to publish frame sync payload for camera={current_cam_id}: {exc}",
                        file=sys.stderr,
                        flush=True
                    )

            inference_count += 1
            now = time.monotonic()
            if now - last_heartbeat_time >= 1.0:
                mjpeg_summary = self.state.status()["summary"]
                latest_cap = mjpeg_summary.get("latest_captured_at_ms")
                last_frame_age_ms = int(time.time() * 1000 - latest_cap) if latest_cap else -1
                track_ids = mjpeg_summary.get("active_track_ids", [])
                print(
                    f"[heartbeat-inference] camera={self.camera_login_id} "
                    f"frame_id={frame_packet.frame_id} "
                    f"processed_frame_count={mjpeg_summary.get('frames_processed', 0)} "
                    f"detection_count={mjpeg_summary.get('bbox_detections', 0)} "
                    f"active_tracks={mjpeg_summary.get('active_tracks', 0)} "
                    f"track_ids={track_ids} "
                    f"dropped_frame_count={self.queue.dropped_frame_count} "
                    f"mjpeg_frame_count={mjpeg_summary.get('mjpeg_frame_count', 0)} "
                    f"last_processed_at={mjpeg_summary.get('latest_processed_at_ms', 0)} "
                    f"last_mjpeg_sent_at={mjpeg_summary.get('last_mjpeg_sent_at', 0.0):.3f} "
                    f"last_frame_age_ms={last_frame_age_ms} "
                    f"stream_clients={mjpeg_summary.get('mjpeg_client_count', 0)} "
                    f"overlay_enabled={str(self.args.mjpeg_enable_overlay).lower()}",
                    flush=True
                )
                inference_count = 0
                mqtt_publish_count = 0
                last_heartbeat_time = now

            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            yolo_metrics.add_yolo_ms(inference_ms)
            metrics_every = max(0, int(getattr(self.args, "infer_metrics_every_n", 30)))
            if metrics_every > 0 and summary.get("frames_processed", 0) % metrics_every == 0 and summary.get("frames_processed", 0) > 0:
                log_periodic_inference_metrics(
                    camera_login_id=self.camera_login_id,
                    backend=getattr(detector, "runtime", None),
                    metrics=yolo_metrics,
                    frames_processed=summary.get("frames_processed"),
                )
            every_n = max(0, int(getattr(self.args, "debug_every_n", 30)))
            if every_n > 0 and frame_packet.frame_id % every_n == 0:
                mjpeg_summary = self.state.status()["summary"]
                print(
                    f"[ai-worker] {current_cam_id} "
                    f"frame_id={frame_packet.frame_id} "
                    f"inference_ms={inference_ms:.1f} "
                    f"queue_lag_ms={queue_lag_ms} "
                    f"mjpeg_client_count={mjpeg_summary.get('mjpeg_client_count', 0)} "
                    f"mjpeg_queue_drop_count={self.queue.dropped_frame_count} "
                    f"mjpeg_encode_latency_ms={float(mjpeg_summary.get('mjpeg_encode_latency_ms', 0.0)):.1f}",
                    flush=True
                )

            if overlay is not None:
                if roi_configs:
                    from ai.visualization.draw import draw_roi_polygon
                    draw_roi_polygon(overlay, roi_configs, color=(0, 255, 255))
                if exit_roi_configs:
                    from ai.visualization.draw import draw_roi_polygon
                    draw_roi_polygon(overlay, exit_roi_configs, color=(0, 165, 255))
                if hazard_roi_configs:
                    from ai.visualization.draw import draw_roi_polygon
                    draw_roi_polygon(overlay, hazard_roi_configs, color=(0, 0, 255)) # Red in BGR
                self.state.update_frame(overlay, summary)
            if self.args.max_frames > 0 and summary["frames_processed"] >= self.args.max_frames:
                break

        # EOF / worker stop: clear recovery + migration context.
        try:
            incident_recovery.reset_camera(self.camera_login_id)
        except Exception:
            incident_recovery.reset_all()
        summary["incident_recovery"] = incident_recovery.diagnostics(self.camera_login_id)
        pose_reporter.log_final_summary()
        close = getattr(publisher, "close", None) if publisher is not None else None
        if close:
            close()


def main():
    parser = argparse.ArgumentParser(description="Publish AI metadata; optionally serve a debug MJPEG overlay stream.")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/cam_01"))
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--camera-login-id", default=None,
                        help="DB cameras.camera_login_id 와 일치하는 식별자. 미지정 시 --camera-id 값 사용")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mjpeg-fps", type=float, default=float(os.getenv("MJPEG_FPS", "8.0")))
    parser.add_argument("--mjpeg-width", type=int, default=int(os.getenv("MJPEG_WIDTH", "1280")))
    parser.add_argument("--mjpeg-height", type=int, default=int(os.getenv("MJPEG_HEIGHT", "720")))
    parser.add_argument("--mjpeg-jpeg-quality", type=int, default=int(os.getenv("MJPEG_JPEG_QUALITY", "80")))
    parser.add_argument("--mjpeg-base-path", default=os.getenv("MJPEG_BASE_PATH", "/mjpeg"))
    parser.add_argument(
        "--mjpeg-enable-overlay",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MJPEG_ENABLE_OVERLAY", "true").lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--mjpeg-debug-watermark",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MJPEG_DEBUG_WATERMARK", "false").lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--mjpeg-debug",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AI_MJPEG_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
        help="Expose annotated MJPEG only for local debugging.",
    )
    parser.add_argument(
        "--mjpeg-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MJPEG_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        help="Expose bounded MJPEG stream for demo/browser viewing.",
    )
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.15)
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_FAINT_THRESHOLD)
    parser.add_argument("--faint-threshold", type=float, default=float(os.getenv("FAINT_THRESHOLD", str(DEFAULT_FAINT_THRESHOLD))))
    parser.add_argument("--fall-threshold", type=float, default=float(os.getenv("FALL_THRESHOLD", str(DEFAULT_FAINT_THRESHOLD))))
    parser.add_argument("--min-consecutive-faint", type=int, default=DEFAULT_MIN_CONSECUTIVE_FAINT)
    parser.add_argument("--consecutive-required", type=int, default=int(os.getenv("CONSECUTIVE_REQUIRED", str(DEFAULT_MIN_CONSECUTIVE_FAINT))))
    parser.add_argument("--camera-cooldown-seconds", type=float, default=DEFAULT_CAMERA_COOLDOWN_SECONDS)
    parser.add_argument("--cooldown-sec", type=float, default=float(os.getenv("COOLDOWN_SEC", str(DEFAULT_CAMERA_COOLDOWN_SECONDS))))
    parser.add_argument("--use-fall-state-machine", action=argparse.BooleanOptionalAction, default=os.getenv("USE_FALL_STATE_MACHINE", "true").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--normal-recover-required", type=int, default=int(os.getenv("NORMAL_RECOVER_REQUIRED", "4")))
    parser.add_argument("--persistent-delay-sec", type=float, default=float(os.getenv("PERSISTENT_DELAY_SEC", "10")))
    parser.add_argument("--persistent-repeat-sec", type=float, default=float(os.getenv("PERSISTENT_REPEAT_SEC", "30")))
    parser.add_argument("--track-lost-grace-sec", type=float, default=float(os.getenv("TRACK_LOST_GRACE_SEC", "3")))
    parser.add_argument("--require-upright-to-lying", action=argparse.BooleanOptionalAction, default=os.getenv("REQUIRE_UPRIGHT_TO_LYING", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--lying-aspect-ratio", type=float, default=float(os.getenv("LYING_ASPECT_RATIO", "1.2")))
    parser.add_argument("--upright-aspect-ratio", type=float, default=float(os.getenv("UPRIGHT_ASPECT_RATIO", "1.3")))
    parser.add_argument("--min-keypoint-conf", type=float, default=float(os.getenv("MIN_KEYPOINT_CONF", "0.3")))
    parser.add_argument("--lying-frames-required", type=int, default=int(os.getenv("LYING_FRAMES_REQUIRED", "2")))
    parser.add_argument("--upright-frames-required", type=int, default=int(os.getenv("UPRIGHT_FRAMES_REQUIRED", "2")))
    parser.add_argument("--movement-low-threshold", type=float, default=float(os.getenv("MOVEMENT_LOW_THRESHOLD", "12.0")))
    parser.add_argument("--infer-metrics-every-n", type=int, default=int(os.getenv("INFER_METRICS_EVERY_N", "30")))
    parser.add_argument("--infer-warmup-frames", type=int, default=int(os.getenv("INFER_WARMUP_FRAMES", "20")))
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_LSTM_SEQUENCE_LENGTH)
    parser.add_argument("--sequence-stride", type=int, default=DEFAULT_LSTM_SEQUENCE_STRIDE)
    parser.add_argument("--cheap-filter-enabled", action=argparse.BooleanOptionalAction, default=os.getenv("CHEAP_FILTER_ENABLED", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--cheap-filter-slope-ratio", type=float, default=float(os.getenv("CHEAP_FILTER_SLOPE_RATIO", "1.3")))
    parser.add_argument("--cheap-filter-min-keypoint-conf", type=float, default=float(os.getenv("CHEAP_FILTER_MIN_KEYPOINT_CONF", "0.25")))
    parser.add_argument("--cheap-filter-min-bbox-area-ratio", type=float, default=float(os.getenv("CHEAP_FILTER_MIN_BBOX_AREA_RATIO", "0.005")))
    parser.add_argument("--cheap-filter-min-center-drop-ratio", type=float, default=float(os.getenv("CHEAP_FILTER_MIN_CENTER_DROP_RATIO", "0.03")))
    parser.add_argument("--cheap-filter-min-aspect-ratio-growth", type=float, default=float(os.getenv("CHEAP_FILTER_MIN_ASPECT_RATIO_GROWTH", "0.20")))
    parser.add_argument("--cheap-filter-min-risk-score", type=float, default=float(os.getenv("CHEAP_FILTER_MIN_RISK_SCORE", "1.0")))
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default="auto")
    parser.add_argument("--track-thresh", type=float, default=0.10)
    parser.add_argument("--match-thresh", "--tracker-iou-threshold", dest="match_thresh", type=float, default=0.20)
    parser.add_argument("--track-buffer", type=int, default=90)
    parser.add_argument("--frame-rate", type=int, default=int(os.getenv("TRACK_FRAME_RATE", os.getenv("FRAME_RATE", "30"))))
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--track-max-missing-seconds", type=float, default=4.0)
    parser.add_argument("--center-match-ratio", type=float, default=0.70)
    parser.add_argument("--tracking-grace-period-seconds", type=float, default=float(os.getenv("TRACKING_GRACE_PERIOD_SECONDS", os.getenv("TRACK_MAX_MISSING_SECONDS", "4.0"))))
    parser.add_argument("--tracking-relink-iou-threshold", type=float, default=float(os.getenv("TRACKING_RELINK_IOU_THRESHOLD", "0.30")))
    parser.add_argument("--tracking-relink-center-ratio", type=float, default=float(os.getenv("TRACKING_RELINK_CENTER_RATIO", "0.70")))
    parser.add_argument("--tracking-relink-max-time-gap-seconds", type=float, default=float(os.getenv("TRACKING_RELINK_MAX_TIME_GAP_SECONDS", "2.0")))
    parser.add_argument("--pose-debug", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_DEBUG", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-debug-summary-every-n", type=int, default=int(os.getenv("POSE_DEBUG_SUMMARY_EVERY_N", "60")))
    parser.add_argument("--pose-min-keypoint-confidence", type=float, default=float(os.getenv("POSE_MIN_KEYPOINT_CONFIDENCE", "0.25")))
    parser.add_argument("--pose-debug-save-images", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_DEBUG_SAVE_IMAGES", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-debug-image-dir", default=os.getenv("POSE_DEBUG_IMAGE_DIR", "runs/pose_debug"))
    parser.add_argument("--pose-debug-image-every-n", type=int, default=int(os.getenv("POSE_DEBUG_IMAGE_EVERY_N", "300")))
    parser.add_argument("--pose-tracking-diag-jsonl", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_TRACKING_DIAG_JSONL", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-tracking-diag-jsonl-path", default=os.getenv("POSE_TRACKING_DIAG_JSONL_PATH", "runs/diagnostics/pose_tracking_diag.jsonl"))
    parser.add_argument(
        "--tracking-stability-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("TRACKING_STABILITY_FALLBACK", "true").lower() in {"1", "true", "yes", "on"},
        help="Use a lightweight bbox continuity tracker after supervision to stabilize final track_id values.",
    )
    parser.add_argument("--overlay-debug-tracks", action="store_true")
    parser.add_argument("--frame-sync-debug", action=argparse.BooleanOptionalAction, default=os.getenv("FRAME_SYNC_DEBUG", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--frame-sync-buffer-size", type=int, default=int(os.getenv("FRAME_SYNC_BUFFER_SIZE", "60")))
    parser.add_argument("--frame-sync-delay-warning-ms", type=int, default=int(os.getenv("FRAME_SYNC_DELAY_WARNING_MS", "300")))
    parser.add_argument("--frame-queue-maxsize", type=int, default=int(os.getenv("FRAME_QUEUE_MAXSIZE", "3")))
    parser.add_argument(
        "--webrtc-sync-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AI_WEBRTC_SYNC_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        help="Expose an optional AI-origin WebRTC video track plus overlay-sync DataChannel.",
    )
    parser.add_argument("--webrtc-sync-host", default=os.getenv("AI_WEBRTC_SYNC_HOST", "0.0.0.0"))
    parser.add_argument("--webrtc-sync-port", type=int, default=int(os.getenv("AI_WEBRTC_SYNC_PORT", "8090")))
    parser.add_argument("--webrtc-sync-stream-id", default=os.getenv("AI_WEBRTC_SYNC_STREAM_ID"))
    parser.add_argument("--webrtc-sync-token", default=os.getenv("AI_WEBRTC_SYNC_TOKEN"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (do not publish events to MQTT)")
    parser.add_argument(
        "--roi-configs",
        default=None,
        help="JSON array of active ROI config objects (polygonPoints in 0~1 normalized coords)",
    )
    parser.add_argument(
        "--exit-roi-configs",
        default=None,
        help="JSON array of EXIT scenario ROI config objects",
    )
    parser.add_argument("--exit-min-consecutive", type=int, default=DEFAULT_EXIT_MIN_CONSECUTIVE)
    parser.add_argument("--exit-cooldown-seconds", type=float, default=DEFAULT_EXIT_COOLDOWN_SECONDS)
    parser.add_argument(
        "--hazard-roi-configs",
        default=None,
        help="JSON array of HAZARD scenario ROI config objects",
    )
    parser.add_argument("--hazard-min-consecutive", type=int, default=DEFAULT_HAZARD_MIN_CONSECUTIVE)
    parser.add_argument("--hazard-cooldown-seconds", type=float, default=DEFAULT_HAZARD_COOLDOWN_SECONDS)
    
    # MQTT Options
    parser.add_argument("--publisher", choices=["mqtt", "console"], help="Event publisher mode (default: from env or console if dry-run)")
    parser.add_argument("--mqtt-host", help="MQTT broker host (default: localhost)")
    parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (default: 1883)")
    parser.add_argument("--mqtt-topic", help="Legacy MQTT event topic alias")
    parser.add_argument("--mqtt-camera-topic", default=os.getenv("MQTT_CAMERA_TOPIC"), help="MQTT overlay topic (default: camera)")
    parser.add_argument("--mqtt-event-topic", default=os.getenv("MQTT_EVENT_TOPIC"), help="MQTT confirmed event topic (default: event or MQTT_TOPIC)")
    parser.add_argument("--mqtt-status-topic", default=os.getenv("MQTT_STATUS_TOPIC"), help="MQTT status topic (default: safety/cameras/status)")
    parser.add_argument("--mqtt-client-id", help="MQTT client ID")
    parser.add_argument("--mqtt-username", help="MQTT username")
    parser.add_argument("--mqtt-password", help="MQTT password")
    parser.add_argument(
        "--selected-track-id",
        "--preferred-track-id",
        dest="selected_track_id",
        type=int,
        default=env_optional_int(
            "AI_SELECTED_TRACK_ID",
            "SELECTED_TRACK_ID",
            "AI_PREFERRED_TRACK_ID",
            "PREFERRED_TRACK_ID",
        ),
        help="Selected track ID to filter overlay and predictions",
    )
    parser.add_argument(
        "--selected-track-mode",
        default=env_optional_str("AI_SELECTED_TRACK_MODE", "SELECTED_TRACK_MODE") or "strict",
        choices=["strict", "fallback"],
        help="Selected track filtering mode: strict or fallback",
    )
    parser.add_argument(
        "--selected-track-missing-frames",
        type=int,
        default=env_optional_int("AI_SELECTED_TRACK_MISSING_FRAMES", "SELECTED_TRACK_MISSING_FRAMES") or 5,
        help="Number of frames allowed for missing selected track in fallback mode",
    )
    
    args = parser.parse_args()
    args.camera_login_id = args.camera_login_id or args.camera_id

    args.roi_configs_parsed = []
    if args.roi_configs:
        try:
            parsed = json.loads(args.roi_configs)
            if isinstance(parsed, list):
                args.roi_configs_parsed = parsed
                print(f"[ai-overlay][roi] loaded {len(parsed)} ROI config(s)", flush=True)
        except json.JSONDecodeError:
            print("[ai-overlay][roi] warning: failed to parse --roi-configs JSON", flush=True)

    args.exit_roi_configs_parsed = []
    if args.exit_roi_configs:
        try:
            parsed = json.loads(args.exit_roi_configs)
            if isinstance(parsed, list):
                args.exit_roi_configs_parsed = parsed
                print(f"[ai-overlay][exit-roi] loaded {len(parsed)} EXIT ROI config(s)", flush=True)
        except json.JSONDecodeError:
            print("[ai-overlay][exit-roi] warning: failed to parse --exit-roi-configs JSON", flush=True)

    args.hazard_roi_configs_parsed = []
    if args.hazard_roi_configs:
        try:
            parsed = json.loads(args.hazard_roi_configs)
            if isinstance(parsed, list):
                args.hazard_roi_configs_parsed = parsed
                print(f"[ai-overlay][hazard-roi] loaded {len(parsed)} HAZARD ROI config(s)", flush=True)
        except json.JSONDecodeError:
            print("[ai-overlay][hazard-roi] warning: failed to parse --hazard-roi-configs JSON", flush=True)

    # Register worker to prevent duplicate starts for same cameraLoginId
    from ai.worker_registry import register_worker, unregister_worker
    try:
        register_worker(
            args.camera_login_id,
            os.getpid(),
            args.rtsp_url,
            f"http://{args.host}:{args.port}" if mjpeg_debug_enabled(args) else ""
        )
    except RuntimeError as exc:
        print(f"[ai-overlay][error] Duplicate worker detected: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    from ai.publishers.event_publisher import mqtt_topic_settings_from_args, mqtt_settings_from_env
    topic_settings = mqtt_topic_settings_from_args(args)
    settings = mqtt_settings_from_env()
    mqtt_host = getattr(args, "mqtt_host", None) or settings["host"]
    mqtt_port = getattr(args, "mqtt_port", None) or settings["port"]
    publisher_mode = getattr(args, "publisher", None) or ("console" if getattr(args, "dry_run", False) else "mqtt")
    status_topic = getattr(args, "mqtt_status_topic", None) or settings["status_topic"]

    print(
        f"[mqtt-topic-audit] cameraLoginId={args.camera_login_id} "
        f"streamId={args.camera_id} host={mqtt_host} port={mqtt_port} "
        f"camera_topic={topic_settings['camera_topic']} "
        f"event_topic={topic_settings['event_topic']} "
        f"status_topic={status_topic} publisher={publisher_mode}",
        flush=True,
    )
    print(
        "[mjpeg-config] "
        f"enabled={mjpeg_debug_enabled(args)} "
        f"host={args.host} "
        f"port={args.port} "
        f"base_path={args.mjpeg_base_path} "
        f"stream_path={args.mjpeg_base_path.rstrip('/')}/{args.camera_login_id} "
        f"fps={args.mjpeg_fps:g} "
        f"width={args.mjpeg_width} "
        f"height={args.mjpeg_height} "
        f"jpeg_quality={args.mjpeg_jpeg_quality} "
        f"enable_overlay={args.mjpeg_enable_overlay}",
        flush=True,
    )

    state = OverlayState()
    webrtc_sync_server = None
    if args.webrtc_sync_enabled:
        try:
            from ai.webrtc_sync import WebRtcSyncConfig, WebRtcSyncServer

            sync_stream_id = args.webrtc_sync_stream_id or f"{args.camera_login_id}_ai"
            webrtc_sync_server = WebRtcSyncServer(
                WebRtcSyncConfig(
                    host=args.webrtc_sync_host,
                    port=args.webrtc_sync_port,
                    stream_id=sync_stream_id,
                    token=args.webrtc_sync_token,
                )
            )
            webrtc_sync_server.start()
        except Exception as exc:
            print(
                f"[ai-webrtc-sync][error] failed to start; continuing with MQTT/STOMP overlay only: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            webrtc_sync_server = None
    worker = OverlayWorker(args, state, sync_sink=webrtc_sync_server)
    server = None

    def shutdown(_signum, _frame):
        print(f"[ai-overlay] Shutdown signal received for camera={args.camera_login_id}. Cleaning up.", flush=True)
        worker.stop()
        if webrtc_sync_server is not None:
            webrtc_sync_server.stop()
        if server is not None:
            server.shutdown()
        unregister_worker(args.camera_login_id)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        worker.start()
        print(f"[ai-overlay] input={redact_url(args.rtsp_url)} detector={args.detector_mode}", flush=True)
        if mjpeg_debug_enabled(args):
            server = create_overlay_server(
                args.host,
                args.port,
                state,
                args.camera_login_id,
                args.mjpeg_fps,
                base_path=args.mjpeg_base_path,
                jpeg_quality=args.mjpeg_jpeg_quality,
                width=args.mjpeg_width,
                height=args.mjpeg_height,
            )
            print(
                f"[ai-overlay] MJPEG serving http://{args.host}:{args.port}{args.mjpeg_base_path.rstrip('/')}/{args.camera_login_id}",
                flush=True,
            )
            server.serve_forever()
        else:
            if webrtc_sync_server is None:
                print("[ai-overlay] metadata-only mode; WebRTC stays on the MediaMTX stream", flush=True)
            else:
                print(
                    f"[ai-overlay] WebRTC sync mode enabled; offerUrl={webrtc_sync_server.url}",
                    flush=True,
                )
            while worker.thread.is_alive():
                worker.thread.join(timeout=1)
    finally:
        worker.stop()
        if webrtc_sync_server is not None:
            webrtc_sync_server.stop()
        if server is not None:
            server.server_close()
        unregister_worker(args.camera_login_id)


if __name__ == "__main__":
    main()
