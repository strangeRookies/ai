import argparse
import sys
import time
from queue import Empty, Queue

from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip
from ai.events.event_clip import EventClipBuffer
from ai.action.lstm_contract import DEFAULT_KEYPOINT_INPUT_SIZE, log_lstm_config
from config import load_settings
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector
from messaging.event_schema import build_safety_event
from rules.fall_rule import FallRuleEngine
from rules.track_sequence import PerTrackSequenceBuffer
from stream.rtsp_reader import RtspFrameReader, redact_url
from tracking.simple_tracker import SimpleTrackAssigner


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Smart Safety Edge AI pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print events without requiring MQTT connectivity.")
    parser.add_argument("--once", action="store_true", help="Exit after the first emitted event.")
    return parser.parse_args()


def create_detector(settings):
    if settings.detector_mode == "mock":
        print("[edge-ai] using mock detector")
        return MockDetector()

    try:
        print(f"[edge-ai] loading YOLO pose model: {settings.yolo_model}")
        return YoloPoseDetector(settings.yolo_model, settings.yolo_device)
    except Exception as exc:
        if not settings.allow_mock_fallback:
            raise
        print(f"[edge-ai] YOLO detector unavailable, falling back to mock detector: {exc}", file=sys.stderr)
        return MockDetector()


def mock_frames(interval_seconds):
    while True:
        yield {"mock": True}
        time.sleep(interval_seconds)


def rtsp_frames(settings, metrics=None):
    reader = RtspFrameReader(
        settings.rtsp_url,
        queue_size=settings.frame_queue_size,
        reconnect_delay_seconds=settings.reconnect_delay_seconds,
    )
    reader.start()
    try:
        while True:
            try:
                frame = reader.read_latest(timeout=1)
                if metrics is not None:
                    metrics["frames_read"] = reader.frames_read
                    metrics["queue_drop_count"] = reader.drop_count
                yield frame
            except Empty:
                if settings.allow_mock_fallback:
                    print("[edge-ai] no RTSP frame available, using mock frame fallback", file=sys.stderr)
                    yield {"mock": True}
                else:
                    print(f"[edge-ai] waiting for RTSP frames: url={redact_url(settings.rtsp_url)}", file=sys.stderr)
    finally:
        reader.stop()


class DryRunPublisher:
    def connect(self):
        return None

    def publish_event(self, event):
        print(f"[edge-ai] dry-run event: {event}", flush=True)

    def close(self):
        return None


def build_fall_event(settings, detection, rule_score, pose_state):
    model_name = detection.get("model_name")
    return build_safety_event(
        event_type="fall_detected",
        camera_id=settings.camera_id,
        severity="HIGH",
        message="Fall-like safety event detected.",
        source="edge-ai",
        track_id=detection.get("track_id"),
        status=detection.get("event_status", "confirmed"),
        confidence=rule_score,
        bbox=detection.get("bbox"),
        model={
            "detector": model_name,
            "classifier": "rule-fusion-v1",
        },
        evidence={
            "snapshot_url": None,
            "clip_url": None,
            "pre_seconds": None,
            "post_seconds": None,
        },
        metadata={
            "detector_confidence": detection.get("confidence"),
            "rule_score": rule_score,
            "pose_state": pose_state,
            "model_name": model_name,
            "sequence_length": detection.get("sequence_length"),
            "sequence_ready": detection.get("sequence_ready"),
            "decision_window": detection.get("decision_window"),
            "decision_votes": detection.get("decision_votes"),
            "keypoint_confidence": detection.get("keypoint_confidence"),
            "keypoint_missing_rate": detection.get("keypoint_missing_rate"),
        },
    )


def main():
    args = parse_args()
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[edge-ai] configuration error: {exc}", file=sys.stderr)
        return 1

    detector = create_detector(settings)
    rule_engine = FallRuleEngine(
        min_duration_seconds=settings.fall_min_duration_seconds,
        debounce_seconds=settings.fall_debounce_seconds,
        candidate_threshold=settings.fall_candidate_threshold,
        decision_window=settings.fall_decision_window,
        decision_required=settings.fall_decision_required,
    )
    tracker = SimpleTrackAssigner(
        iou_threshold=settings.track_iou_threshold,
        max_missing_seconds=settings.track_max_missing_seconds,
    )
    sequence_buffer = PerTrackSequenceBuffer(
        sequence_length=settings.sequence_length,
        max_track_age_seconds=settings.sequence_max_track_age_seconds,
    )
    log_lstm_config("[lstm-config]", settings.sequence_length, settings.sequence_stride, DEFAULT_KEYPOINT_INPUT_SIZE, "config/env")

    if args.dry_run:
        publisher = DryRunPublisher()
    else:
        from messaging.mqtt_publisher import MqttPublisher

        publisher = MqttPublisher(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            topic=settings.mqtt_topic,
            client_id=settings.mqtt_client_id,
            username=settings.mqtt_username,
            password=settings.mqtt_password,
        )
        publisher.connect()

    clip_worker = None
    clip_buffer = None
    clip_queue = None
    if settings.event_clip_enabled:
        clip_queue = Queue(maxsize=max(1, settings.event_clip_queue_size))
        clip_buffer = EventClipBuffer(
            pre_event_frame_count=settings.event_clip_pre_frames,
            post_event_frame_count=settings.event_clip_post_frames,
            cooldown_seconds=settings.event_clip_cooldown_seconds,
            fps=settings.event_clip_fps,
            output_dir=settings.event_clip_output_dir,
        )
        clip_worker = ClipWriterWorker(clip_queue)
        clip_worker.start()

    rtsp_metrics = None
    frames = mock_frames(settings.mock_frame_interval_seconds)
    if settings.detector_mode != "mock":
        rtsp_metrics = {"frames_read": 0, "queue_drop_count": 0}
        frames = rtsp_frames(settings, rtsp_metrics)

    processed = 0
    started_at = time.perf_counter()
    inference_latency_ms = None
    try:
        for frame in frames:
            clip_task = clip_buffer.add_frame(frame) if clip_buffer else None
            if clip_task and clip_queue:
                enqueue_event_clip(clip_queue, clip_task)

            inference_started_at = time.perf_counter()
            detections = detector.detect(frame)
            detections = tracker.update(detections)
            detections = sequence_buffer.update(detections)
            for detection, rule_score, pose_state in rule_engine.evaluate(detections):
                event = build_fall_event(settings, detection, rule_score, pose_state)
                publisher.publish_event(event)
                if clip_buffer:
                    clip_buffer.trigger_event(
                        event_type=event["type"],
                        camera_id=event["camera_id"],
                        metadata={
                            "event_timestamp": event.get("timestamp"),
                            "track_id": detection.get("track_id"),
                            "rule_score": rule_score,
                            "pose_state": pose_state,
                        },
                    )
                if args.once:
                    return 0
            inference_latency_ms = (time.perf_counter() - inference_started_at) * 1000.0

            processed += 1
            if processed % 30 == 0:
                runtime_seconds = max(time.perf_counter() - started_at, 1e-9)
                effective_processing_fps = processed / runtime_seconds
                frames_read = rtsp_metrics["frames_read"] if rtsp_metrics is not None else "TODO(non-RTSP path)"
                queue_drop_count = rtsp_metrics["queue_drop_count"] if rtsp_metrics is not None else "TODO(non-RTSP path)"
                latency_text = "TODO(no completed inference)" if inference_latency_ms is None else f"{inference_latency_ms:.3f}"
                print(
                    "[edge-ai] rtsp processing status: "
                    f"frames_read={frames_read} "
                    f"frames_processed={processed} "
                    f"effective_processing_fps={effective_processing_fps:.3f} "
                    f"inference_latency_ms={latency_text} "
                    f"queue_drop_count={queue_drop_count}",
                    flush=True,
                )
            if settings.max_frames > 0 and processed >= settings.max_frames:
                print(f"[edge-ai] reached MAX_FRAMES={settings.max_frames}")
                return 0
    except KeyboardInterrupt:
        print("\n[edge-ai] stopped by user")
        return 0
    finally:
        if clip_worker:
            clip_worker.stop()
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
