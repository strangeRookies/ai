import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure the repo root is on sys.path before any ai.* imports,
# so serve_ai_overlay.py works regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.events.event_clip import EventClipBuffer
from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip

from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.action.lstm_contract import DEFAULT_KEYPOINT_INPUT_SIZE, DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE, log_lstm_config
from ai.evidence import evidence_id
from ai.frame_sync import FrameMetadataBuffer, FramePacket, CameraFrameQueue
from ai.inference.rtsp_runtime import build_inference_event_payload, cheap_filter_config_from_args, create_detection_postprocessor, ensure_mock_keypoints
from ai.inference.rtsp_runtime import maybe_log_debug, normalize_detections, update_detections_with_postprocessor, update_prediction_counts, update_tracking_summary
from ai.inference.rtsp_runtime import (
    log_detection_stage,
    log_tracking_stage,
    log_classification_stage,
    log_payload_stage,
    update_quantitative_summary,
)
from ai.inference.tracking_debug import log_sequence_stage
from ai.overlay_http import OverlayState, create_overlay_server
from ai.roi import apply_roi_mask, combine_roi_masks, find_boxes_in_exit_zone
from ai.streams.video_reader import VideoReader
from ai.visualization.action_overlay import annotate_boxes_with_action, annotate_boxes_with_track_actions, draw_metrics_panel, faint_probability
from ai.visualization.action_overlay import format_action_overlay_text, initial_overlay_summary, update_overlay_runtime
from ai.visualization.draw import draw_overlay
from scripts.run_rtsp_inference import DEFAULT_ACTION_MODEL, DEFAULT_CAMERA_COOLDOWN_SECONDS, DEFAULT_FAINT_THRESHOLD, DEFAULT_MIN_CONSECUTIVE_FAINT
from scripts.run_rtsp_inference import FaintEventPostProcessor, create_classifier, create_detector
from ai.action.faint_post_processing import ExitEventPostProcessor, DEFAULT_EXIT_MIN_CONSECUTIVE, DEFAULT_EXIT_COOLDOWN_SECONDS
from stream.rtsp_reader import redact_url
from tracking.display_id_mapper import DisplayIdMapper
from ai.publishers.event_publisher import create_event_publisher, mqtt_topic_settings_from_args
from ai.publishers.camera_status_publisher import CameraStatusPublisher
from ai.publishers.mqtt_payloads import build_overlay_payload, current_timestamp_ms, frame_size_from_shape, build_frame_sync_payload


def initial_summary():
    return initial_overlay_summary()

#각 카메라별로 동작하는 실시간 오버레이 스크립트 
#RTSP 스트림을 캡처하여 AI 분석(YOLO Pose 및 LSTM)을 수행
#감지된 객체의 바운딩 박스(bbox), 트래킹 ID 및 상태 메타데이터를 MQTT camera 토픽으로 실시간 발행
class OverlayPublishState:
    def __init__(self):
        self.signals_by_track = {}
        self.last_timestamp_ms = 0

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


def mjpeg_debug_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "mjpeg_debug", False))


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
    track_selector=None,
):
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
            track_selector=track_selector,
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
    track_selector=None,
):
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

    # Stage 1: Detection log
    _log_frame_id = frame_metadata.frame_id if frame_metadata is not None else getattr(frame_packet, "frame_idx", 0)
    _log_ts = frame_metadata.captured_at_ms if frame_metadata is not None else None
    log_detection_stage(stream_id, _log_frame_id, _log_ts, detections)

    _pre_track_detections = detections
    if tracker is not None:
        detections = update_detections_with_postprocessor(tracker, detections, frame_packet.frame, frame_packet.timestamp)

    # Stage 2: Tracking log
    _tracker_diag = tracker.diagnostics() if tracker is not None else {}
    log_tracking_stage(stream_id, _log_frame_id, _pre_track_detections, detections, _tracker_diag)
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
        diag = _tracker_diag
        new_cnt = diag.get("new_tracks", 0)
        lost_cnt = diag.get("lost_tracks", 0)
        print(
            f"[Tracking Debug] camera: {stream_id} | frameId: {frame_id} | "
            f"detections: {det_cnt} | tracked: {tracked_cnt} | "
            f"new_tracks: {new_cnt} | lost_tracks: {lost_cnt} | "
            f"active_ids: {active_tids} | conf_range: {conf_range} | "
            f"mapping: {matched_details}",
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
    summary["max_active_tracks"] = max(summary.get("max_active_tracks", 0), active_tracks)
    if tracker is not None:
        update_tracking_summary(summary, tracker.diagnostics())

    # EXIT 이탈 감지: 트래킹된 박스 center가 EXIT ROI 안에 있으면 알림
    if exit_roi_mask is not None and exit_post_processor is not None:
        all_track_ids = {int(float(str(b["track_id"]))) for b in boxes if b.get("track_id") is not None}
        in_exit_zone = find_boxes_in_exit_zone(boxes, exit_roi_mask)
        for track_id in all_track_ids - in_exit_zone:
            exit_post_processor.reset_track(args.camera_id, track_id)
        for track_id in in_exit_zone:
            if exit_post_processor.should_trigger(args.camera_id, track_id, frame_packet.timestamp):
                exit_boxes = [b for b in boxes if b.get("track_id") is not None and int(float(str(b["track_id"]))) == track_id]
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
                print(f"[exit-event] {stream_id} track_id={track_id}", flush=True)

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
        update_prediction_counts(summary, prediction)
    summary["per_track_sequences_generated"] = {
        str(track_id): count for track_id, count in sequence_buffer.sequences_generated_by_track.items()
    }
    log_sequence_stage(
        stream_id,
        _log_frame_id,
        active_track_ids=sequence_buffer.active_track_ids(),
        buffer_lengths=buffer_lengths,
        sequences_generated=len(sequences),
        sequences_generated_by_track=sequence_buffer.sequences_generated_by_track,
        latest_faint_prob=summary.get("latest_faint_probability"),
    )
    for track_id, track_prediction in predictions_by_track.items():
        event_triggered = False
        if post_processor is not None:
            event_triggered = post_processor.should_trigger(args.camera_id, track_prediction, frame_packet.timestamp, track_id=track_id)
            consecutive_by_track[track_id] = post_processor.consecutive_count(args.camera_id, track_id=track_id)
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
    if publisher is not None:
        try:
            publisher.publish(overlay_payload, topic=topic_settings["camera_topic"])
        except Exception as exc:
            print(f"[ai-worker][error] failed to publish overlay payload for camera={stream_id}: {exc}", file=sys.stderr, flush=True)
    for track_id in triggered_track_ids:
        track_prediction = predictions_by_track[track_id]
        sequence = sequences_by_track[track_id]
        payload = build_inference_event_payload(
            args,
            frame_packet,
            track_prediction,
            boxes,
            sequence,
            frame_metadata=frame_metadata,
            published_at_ms=published_at_ms,
            dropped_frame_count=dropped_frame_count,
        )
        log_lstm_event(args, stream_id, sequence, track_prediction)
        summary["events_generated"] += 1
        if summary["sample_event"] is None:
            summary["sample_event"] = payload
        if args.print_events:
            print(f"[ai-overlay-event] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        if publisher is not None:
            try:
                publisher.publish(payload, topic=topic_settings["event_topic"])
            except Exception as exc:
                print(f"[ai-worker][error] failed to publish event payload for camera={stream_id}: {exc}", file=sys.stderr, flush=True)

        # 낙상 감지 시 10초 스냅샷 버퍼 트리거 작동
        if state is not None and getattr(state, "clip_buffer", None) is not None:
            target_bbox = []
            for b in boxes:
                if b.get("track_id") is not None and int(b["track_id"]) == track_id:
                    target_bbox = b.get("box", [])
                    break
            
            task_metadata = {
                "evidenceId": payload.get("eventId"),
                "event_timestamp": payload.get("timestamp"),
                "track_id": track_id,
                "bbox": target_bbox
            }
            
            state.clip_buffer.trigger_event(
                event_type=payload.get("type", "fall_detected"),
                camera_id=stream_id,
                metadata=task_metadata,
                queue=state.clip_queue
            )
            print(f"[ai-overlay-event] triggered snapshot recording for camera={stream_id} eventId={payload.get('eventId')}", flush=True)
    maybe_log_debug(frame_packet, boxes, summary, prediction, args, prefix="[ai-overlay-debug]")

    update_overlay_runtime(summary)
    if not mjpeg_debug_enabled(args):
        return None
    overlay_frame_id = frame_metadata.frame_id if frame_metadata is not None else frame_packet.frame_idx
    overlay = draw_overlay(frame_packet.frame, boxes, prediction, overlay_frame_id)
    draw_metrics_panel(overlay, summary, args, prediction)
    return overlay


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
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ai-overlay-worker", daemon=True)
        self.reader_thread = None
        self.camera_login_id = getattr(self.args, "camera_login_id", self.args.camera_id) or self.args.camera_id
        self.queue = CameraFrameQueue(self.camera_login_id, maxsize=getattr(self.args, "frame_queue_maxsize", 3))
        self.frame_buffer = FrameMetadataBuffer(maxlen=self.args.frame_sync_buffer_size)

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
            publisher, publisher_mode = create_event_publisher(self.args)
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
                            self.queue.put_latest(packet_wrapped)
                            
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
        detector = create_detector(self.args.detector_mode, self.args.yolo_model, self.args.device, self.args.imgsz, conf=self.args.detector_conf)
        classifier, _classifier_mode = create_classifier(self.args.action_model, self.args.action_device, self.args.action_threshold)
        publisher, publisher_mode = create_event_publisher(self.args)
        print(f"[ai-overlay-inference] initialized event publisher: {publisher_mode}", flush=True)
        
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

        summary = initial_summary()
        post_processor = FaintEventPostProcessor(
            min_consecutive_faint=self.args.min_consecutive_faint,
            cooldown_seconds=self.args.camera_cooldown_seconds,
        )
        tracker, postprocessing_mode = create_detection_postprocessor(self.args)
        cheap_filter_config = cheap_filter_config_from_args(self.args)
        print(f"[ai-overlay-inference] tracking postprocessor: {postprocessing_mode}", flush=True)
        display_id_mapper = DisplayIdMapper()
        overlay_publish_state = OverlayPublishState()
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
                )

        last_heartbeat_time = time.monotonic()
        inference_count = 0
        mqtt_publish_count = 0

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

            # 매 프레임마다 스냅샷 클립 버퍼에 기록
            if self.state.clip_buffer is not None:
                self.state.clip_buffer.add_frame(frame_packet.frame)

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
                track_selector=track_selector,
            )

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
                print(
                    f"[heartbeat-inference] camera={self.camera_login_id} "
                    f"inference_count={inference_count} "
                    f"mqtt_publish_count={mqtt_publish_count} "
                    f"fps={inference_count / (now - last_heartbeat_time):.1f} "
                    f"dropped={self.queue.dropped_frame_count}",
                    flush=True
                )
                inference_count = 0
                mqtt_publish_count = 0
                last_heartbeat_time = now

            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            every_n = max(0, int(getattr(self.args, "debug_every_n", 30)))
            if every_n > 0 and frame_packet.frame_id % every_n == 0:
                print(
                    f"[ai-worker] {current_cam_id} "
                    f"frame_id={frame_packet.frame_id} "
                    f"inference_ms={inference_ms:.1f} "
                    f"queue_lag_ms={queue_lag_ms}",
                    flush=True
                )

            if overlay is not None:
                if roi_configs:
                    from ai.visualization.draw import draw_roi_polygon
                    draw_roi_polygon(overlay, roi_configs, color=(0, 255, 255))
                if exit_roi_configs:
                    from ai.visualization.draw import draw_roi_polygon
                    draw_roi_polygon(overlay, exit_roi_configs, color=(0, 165, 255))
                self.state.update_frame(overlay, summary)
            if self.args.max_frames > 0 and summary["frames_processed"] >= self.args.max_frames:
                break

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
    parser.add_argument("--mjpeg-fps", type=float, default=8.0)
    parser.add_argument(
        "--mjpeg-debug",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AI_MJPEG_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
        help="Expose annotated MJPEG only for local debugging.",
    )
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.15)
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_FAINT_THRESHOLD)
    parser.add_argument("--min-consecutive-faint", type=int, default=DEFAULT_MIN_CONSECUTIVE_FAINT)
    parser.add_argument("--camera-cooldown-seconds", type=float, default=DEFAULT_CAMERA_COOLDOWN_SECONDS)
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
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--track-max-missing-seconds", type=float, default=4.0)
    parser.add_argument("--center-match-ratio", type=float, default=0.70)
    parser.add_argument(
        "--tracking-stability-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("TRACKING_STABILITY_FALLBACK", "false").lower() in {"1", "true", "yes", "on"},
        help="Use a lightweight bbox continuity tracker after supervision to stabilize final track_id values.",
    )
    parser.add_argument("--overlay-debug-tracks", action="store_true")
    parser.add_argument("--frame-sync-debug", action=argparse.BooleanOptionalAction, default=os.getenv("FRAME_SYNC_DEBUG", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--frame-sync-buffer-size", type=int, default=int(os.getenv("FRAME_SYNC_BUFFER_SIZE", "60")))
    parser.add_argument("--frame-sync-delay-warning-ms", type=int, default=int(os.getenv("FRAME_SYNC_DELAY_WARNING_MS", "300")))
    parser.add_argument("--frame-queue-maxsize", type=int, default=int(os.getenv("FRAME_QUEUE_MAXSIZE", "3")))
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

    state = OverlayState()
    worker = OverlayWorker(args, state)
    server = None

    def shutdown(_signum, _frame):
        print(f"[ai-overlay] Shutdown signal received for camera={args.camera_login_id}. Cleaning up.", flush=True)
        worker.stop()
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
            server = create_overlay_server(args.host, args.port, state, args.camera_id, args.mjpeg_fps)
            print(f"[ai-overlay] debug MJPEG serving http://{args.host}:{args.port}/stream", flush=True)
            server.serve_forever()
        else:
            print("[ai-overlay] metadata-only mode; WebRTC stays on the MediaMTX stream", flush=True)
            while worker.thread.is_alive():
                worker.thread.join(timeout=1)
    finally:
        worker.stop()
        if server is not None:
            server.server_close()
        unregister_worker(args.camera_login_id)


if __name__ == "__main__":
    main()
