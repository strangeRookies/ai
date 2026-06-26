import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.inference.rtsp_runtime import build_inference_event_payload, cheap_filter_config_from_args, create_detection_postprocessor, ensure_mock_keypoints
from ai.inference.rtsp_runtime import maybe_log_debug, normalize_detections, update_detections_with_postprocessor, update_prediction_counts, update_tracking_summary
from ai.overlay_http import OverlayState, create_overlay_server
from ai.streams.video_reader import VideoReader
from ai.visualization.action_overlay import annotate_boxes_with_action, annotate_boxes_with_track_actions, draw_metrics_panel, faint_probability
from ai.visualization.action_overlay import format_action_overlay_text, initial_overlay_summary, update_overlay_runtime
from ai.visualization.draw import draw_overlay
from scripts.run_rtsp_inference import DEFAULT_ACTION_MODEL, DEFAULT_CAMERA_COOLDOWN_SECONDS, DEFAULT_FAINT_THRESHOLD, DEFAULT_MIN_CONSECUTIVE_FAINT
from scripts.run_rtsp_inference import FaintEventPostProcessor, create_classifier, create_detector
from stream.rtsp_reader import redact_url
from tracking.display_id_mapper import DisplayIdMapper
from ai.publishers.event_publisher import create_event_publisher, mqtt_topic_settings_from_args
from ai.publishers.camera_status_publisher import CameraStatusPublisher
from ai.publishers.mqtt_payloads import build_overlay_payload, current_timestamp_ms, frame_size_from_shape


def initial_summary():
    return initial_overlay_summary()


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


def process_frame(packet, detector, classifier, sequence_buffer, summary, args, post_processor=None, tracker=None, state=None, display_id_mapper=None, publisher=None, overlay_publish_state=None):
    detections = detector.detect(packet.frame)
    if args.detector_mode == "mock":
        detections = ensure_mock_keypoints(detections)
    if tracker is not None:
        detections = update_detections_with_postprocessor(tracker, detections, packet.frame, packet.timestamp)
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

    # Update display ID mapping so operator labels stay compact (1, 2, 3…)
    if display_id_mapper is not None:
        active_raw_ids = {int(b["track_id"]) for b in boxes if b.get("track_id") is not None}
        display_id_mapper.update(active_raw_ids)
        for box in boxes:
            raw_id = box.get("track_id")
            if raw_id is not None:
                box["display_id"] = display_id_mapper.display_id(int(raw_id))
        summary["display_id_map"] = display_id_mapper.mapping_snapshot()

    classifier_input = getattr(args, "classifier_input", None)
    if classifier_input == "crops":
        sequences = sequence_buffer.add(packet.frame_idx, packet.frame, boxes, now=packet.timestamp)
    else:
        sequences = sequence_buffer.add(packet.frame_idx, detections, packet.frame.shape, now=packet.timestamp)
    prediction = None
    predictions_by_track = {}
    sequences_by_track = {}
    triggered_track_ids = set()
    consecutive_by_track = {}
    for sequence in sequences:
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
    for track_id, track_prediction in predictions_by_track.items():
        event_triggered = False
        if post_processor is not None:
            event_triggered = post_processor.should_trigger(args.camera_id, track_prediction, packet.timestamp, track_id=track_id)
            consecutive_by_track[track_id] = post_processor.consecutive_count(args.camera_id, track_id=track_id)
        elif track_prediction and track_prediction.get("label") != "Normal":
            event_triggered = True
            consecutive_by_track[track_id] = 1
        if event_triggered:
            triggered_track_ids.add(track_id)
            track_key = str(track_id)
            summary["events_generated_by_track"][track_key] = summary["events_generated_by_track"].get(track_key, 0) + 1
    summary["latest_consecutive_faint"] = max(consecutive_by_track.values(), default=0)
    annotate_boxes_with_track_actions(boxes, predictions_by_track, consecutive_by_track, triggered_track_ids, args)
    topic_settings = mqtt_topic_settings_from_args(args)
    frame_width, frame_height = frame_size_from_shape(packet.frame.shape)
    stream_id = getattr(args, "camera_login_id", None) or args.camera_id
    timestamp_ms = None
    if overlay_publish_state is not None:
        overlay_publish_state.apply_latest_signals(boxes)
        timestamp_ms = overlay_publish_state.next_timestamp_ms()
    overlay_payload = build_overlay_payload(
        stream_id=stream_id,
        frame_width=frame_width,
        frame_height=frame_height,
        boxes=boxes,
        timestamp_ms=timestamp_ms,
    )
    summary["latest_overlay_event_count"] = len(overlay_payload["events"])
    if publisher is not None:
        publisher.publish(overlay_payload, topic=topic_settings["camera_topic"])
    for track_id in triggered_track_ids:
        track_prediction = predictions_by_track[track_id]
        sequence = sequences_by_track[track_id]
        payload = build_inference_event_payload(args, packet, track_prediction, boxes, sequence)
        summary["events_generated"] += 1
        if summary["sample_event"] is None:
            summary["sample_event"] = payload
        if args.print_events:
            print(f"[ai-overlay-event] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        if publisher is not None:
            publisher.publish(payload, topic=topic_settings["event_topic"])
    maybe_log_debug(packet, boxes, summary, prediction, args, prefix="[ai-overlay-debug]")

    update_overlay_runtime(summary)
    overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
    draw_metrics_panel(overlay, summary, args, prediction)
    return overlay


class OverlayWorker:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ai-overlay-worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _run(self):
        detector = create_detector(self.args.detector_mode, self.args.yolo_model, self.args.device, self.args.imgsz, conf=self.args.detector_conf)
        classifier, _classifier_mode = create_classifier(self.args.action_model, self.args.action_device, self.args.action_threshold)
        publisher, publisher_mode = create_event_publisher(self.args)
        print(f"[ai-overlay] initialized event publisher: {publisher_mode}", flush=True)

        # 카메라 연결 상태 퍼블리셔 (safety/cameras/status 토픽)
        camera_login_id = getattr(self.args, "camera_login_id", self.args.camera_id)
        status_publisher = CameraStatusPublisher(
            mqtt_publisher=publisher,
            camera_login_id=camera_login_id,
            rtsp_url=self.args.rtsp_url,
        )
        summary = initial_summary()
        post_processor = FaintEventPostProcessor(
            min_consecutive_faint=self.args.min_consecutive_faint,
            cooldown_seconds=self.args.camera_cooldown_seconds,
        )
        tracker, postprocessing_mode = create_detection_postprocessor(self.args)
        cheap_filter_config = cheap_filter_config_from_args(self.args)
        print(f"[ai-overlay] tracking postprocessor: {postprocessing_mode}", flush=True)
        display_id_mapper = DisplayIdMapper()
        overlay_publish_state = OverlayPublishState()
        while not self.stop_event.is_set():
            if self.args.classifier_input == "crops":
                sequence_buffer = PerTrackCropSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    self.args.resize_size,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                )
            else:
                sequence_buffer = PerTrackKeypointSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                    cheap_filter_config=cheap_filter_config,
                )
            # Reset display ID mapping on each camera reconnect so IDs restart from 1
            display_id_mapper.reset()
            try:
                with VideoReader(self.args.rtsp_url) as reader:
                    print(f"[ai-overlay] connected: {redact_url(self.args.rtsp_url)}", flush=True)
                    # RTSP 연결 성공 → MQTT 상태 이벤트 발행
                    status_publisher.notify_connected()
                    while not self.stop_event.is_set():
                        packet = reader.read()
                        if packet is None:
                            # 스트림 종료 (프레임 없음) → 연결 끊김으로 판단
                            status_publisher.notify_disconnected(reason="STREAM_ENDED")
                            break
                        overlay = process_frame(
                            packet, detector, classifier, sequence_buffer, summary, self.args,
                            post_processor=post_processor, tracker=tracker,
                            state=self.state, display_id_mapper=display_id_mapper,
                            publisher=publisher,
                            overlay_publish_state=overlay_publish_state
                        )
                        self.state.update_frame(overlay, summary)
                        if self.args.max_frames > 0 and summary["frames_processed"] >= self.args.max_frames:
                            return
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"[ai-overlay] {message}", file=sys.stderr, flush=True)
                self.state.update_error(message)
                # 연결 오류 → MQTT 상태 이벤트 발행
                status_publisher.notify_error(reason=type(exc).__name__)
            # 재연결 대기 → MQTT 상태 이벤트 발행
            if not self.stop_event.is_set():
                status_publisher.notify_reconnecting()
            time.sleep(self.args.reconnect_delay)


def main():
    parser = argparse.ArgumentParser(description="Serve a local MJPEG stream with AI bbox/keypoint/action overlays.")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/cam_01"))
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--camera-login-id", default=None,
                        help="DB cameras.camera_login_id 와 일치하는 식별자. 미지정 시 --camera-id 값 사용")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mjpeg-fps", type=float, default=8.0)
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.10)
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_FAINT_THRESHOLD)
    parser.add_argument("--min-consecutive-faint", type=int, default=DEFAULT_MIN_CONSECUTIVE_FAINT)
    parser.add_argument("--camera-cooldown-seconds", type=float, default=DEFAULT_CAMERA_COOLDOWN_SECONDS)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--cheap-filter-enabled", action=argparse.BooleanOptionalAction, default=os.getenv("CHEAP_FILTER_ENABLED", "true").lower() in {"1", "true", "yes", "on"})
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
    parser.add_argument("--overlay-debug-tracks", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--print-events", action="store_true")
    
    # MQTT Options
    parser.add_argument("--publisher", choices=["mqtt", "console"], help="Event publisher mode (default: from env or console if dry-run)")
    parser.add_argument("--mqtt-host", help="MQTT broker host (default: localhost)")
    parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (default: 1883)")
    parser.add_argument("--mqtt-topic", help="Legacy MQTT event topic alias")
    parser.add_argument("--mqtt-camera-topic", default=os.getenv("MQTT_CAMERA_TOPIC"), help="MQTT overlay topic (default: camera)")
    parser.add_argument("--mqtt-event-topic", default=os.getenv("MQTT_EVENT_TOPIC"), help="MQTT confirmed event topic (default: event or MQTT_TOPIC)")
    parser.add_argument("--mqtt-client-id", help="MQTT client ID")
    parser.add_argument("--mqtt-username", help="MQTT username")
    parser.add_argument("--mqtt-password", help="MQTT password")
    
    args = parser.parse_args()
    args.camera_login_id = args.camera_login_id or args.camera_id

    state = OverlayState()
    worker = OverlayWorker(args, state)
    server = create_overlay_server(args.host, args.port, state, args.camera_id, args.mjpeg_fps)

    worker.start()
    print(f"[ai-overlay] serving http://{args.host}:{args.port}/stream", flush=True)
    print(f"[ai-overlay] input={redact_url(args.rtsp_url)} detector={args.detector_mode}", flush=True)

    def shutdown(_signum, _frame):
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    finally:
        worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
