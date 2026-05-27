import argparse
import sys
import time
from queue import Empty, Queue

from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip
from ai.events.event_clip import EventClipBuffer
from config import load_settings
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector
from messaging.event_schema import build_safety_event
from messaging.mqtt_publisher import MqttPublisher
from rules.fall_rule import FallRuleEngine
from stream.rtsp_reader import RtspFrameReader, redact_url


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


def rtsp_frames(settings):
    reader = RtspFrameReader(
        settings.rtsp_url,
        queue_size=settings.frame_queue_size,
        reconnect_delay_seconds=settings.reconnect_delay_seconds,
    )
    reader.start()
    try:
        while True:
            try:
                yield reader.read_latest(timeout=1)
            except Empty:
                if settings.allow_mock_fallback:
                    print("[edge-ai] no RTSP frame available, using mock frame fallback", file=sys.stderr)
                    yield {"mock": True}
                else:
                    print(f"[edge-ai] waiting for RTSP frames: url={redact_url(settings.rtsp_url)}", file=sys.stderr)
    finally:
        reader.stop()


def build_fall_event(settings, detection, rule_score, pose_state):
    return build_safety_event(
        event_type="fall_detected",
        camera_id=settings.camera_id,
        severity="HIGH",
        message="쓰러짐 의심 상황이 감지되었습니다.",
        source="edge-ai",
        track_id=detection.get("track_id"),
        metadata={
            "bbox": detection.get("bbox"),
            "confidence": detection.get("confidence"),
            "rule_score": rule_score,
            "pose_state": pose_state,
            "model_name": detection.get("model_name"),
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
    )

    publisher = MqttPublisher(
        host=settings.mqtt_host,
        port=settings.mqtt_port,
        topic=settings.mqtt_topic,
        client_id=settings.mqtt_client_id,
    )
    if not args.dry_run:
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

    frames = mock_frames(settings.mock_frame_interval_seconds)
    if settings.detector_mode != "mock":
        frames = rtsp_frames(settings)

    processed = 0
    try:
        for frame in frames:
            clip_task = clip_buffer.add_frame(frame) if clip_buffer else None
            if clip_task and clip_queue:
                enqueue_event_clip(clip_queue, clip_task)

            detections = detector.detect(frame)
            for detection, rule_score, pose_state in rule_engine.evaluate(detections):
                event = build_fall_event(settings, detection, rule_score, pose_state)
                if args.dry_run:
                    print(f"[edge-ai] dry-run event: {event}", flush=True)
                else:
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

            processed += 1
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
