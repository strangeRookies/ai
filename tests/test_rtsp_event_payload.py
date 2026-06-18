import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

import ai.publishers.event_publisher as publisher_module
import scripts.run_rtsp_inference as rtsp_module
from scripts.run_rtsp_inference import (
    build_inference_event_log,
    build_inference_event_payload,
    create_event_publisher,
    run,
    save_inference_event_log,
    write_run_summary,
)


class RtspEventPayloadTest(unittest.TestCase):
    def test_inference_event_payload_contains_only_operational_fields(self):
        args = Namespace(camera_id="legacy_cam", camera_login_id="cam_01", event_severity="HIGH", yolo_model="yolo26n-pose.pt")
        packet = Namespace(frame_idx=12, timestamp=123.5)
        prediction = {"label": "Faint", "score": 0.81, "probabilities": {"Normal": 0.19, "Faint": 0.81}}
        sequence = {"bbox": [1, 2, 3, 4], "track_id": 9, "start_frame": 4, "end_frame": 12}

        payload = build_inference_event_payload(args, packet, prediction, boxes=[], sequence=sequence)

        self.assertEqual(set(payload), {
            "type", "camera_id", "camera_login_id", "timestamp", "severity",
            "message", "source", "track_id", "metadata",
        })
        self.assertEqual(payload["camera_id"], "cam_01")
        self.assertEqual(payload["camera_login_id"], "cam_01")
        self.assertEqual(payload["type"], "fall_detected")
        self.assertEqual(payload["severity"], "HIGH")
        self.assertEqual(payload["message"], "쓰러짐 의심 상황이 감지되었습니다.")
        self.assertEqual(payload["source"], "edge-ai")
        self.assertEqual(payload["track_id"], 9)
        self.assertTrue(isinstance(payload["timestamp"], str) and payload["timestamp"].endswith("Z"))
        self.assertEqual(
            payload["metadata"],
            {
                "bbox": [1, 2, 3, 4],
                "confidence": 0.81,
                "rule_score": 0.81,
                "pose_state": "LYING",
                "model_name": "yolo26n-pose",
            },
        )
        self.assertNotIn("event_type", payload)
        self.assertNotIn("detected_at", payload)
        self.assertNotIn("probabilities", payload)
        self.assertNotIn("threshold", payload)
        self.assertNotIn("post_processing", payload)

    def test_inference_event_payload_omits_track_id_when_missing(self):
        args = Namespace(camera_id="cam_01", event_severity="HIGH")
        packet = Namespace(frame_idx=12, timestamp=123.5)
        prediction = {"label": "Faint", "score": 0.81}
        sequence = {"bbox": [1, 2, 3, 4], "start_frame": 4, "end_frame": 12}

        payload = build_inference_event_payload(args, packet, prediction, boxes=[], sequence=sequence)

        self.assertNotIn("track_id", payload)
        self.assertEqual(payload["metadata"]["bbox"], [1, 2, 3, 4])

    def test_inference_event_payload_keeps_clip_reference_out_of_mqtt_contract(self):
        args = Namespace(camera_id="cam_01", event_severity="HIGH")
        packet = Namespace(frame_idx=12, timestamp=123.5)
        prediction = {"label": "Faint", "score": 0.81}
        sequence = {
            "bbox": [1, 2, 3, 4],
            "track_id": 9,
            "start_frame": 4,
            "end_frame": 12,
            "clip_path": "clips/faint_cam_01.mp4",
            "clip_url": "https://example.invalid/clips/faint_cam_01.mp4",
        }

        payload = build_inference_event_payload(args, packet, prediction, boxes=[], sequence=sequence)

        self.assertNotIn("clip_path", payload)
        self.assertNotIn("clip_url", payload)

    def test_inference_event_log_contains_debug_fields(self):
        args = Namespace(camera_id="cam_01", action_threshold=0.3, min_consecutive_faint=3, camera_cooldown_seconds=10)
        packet = Namespace(frame_idx=12, timestamp=123.5)
        prediction = {"label": "Faint", "score": 0.81, "probabilities": {"Normal": 0.19, "Faint": 0.81}}
        sequence = {"bbox": [1, 2, 3, 4], "track_id": 9, "start_frame": 4, "end_frame": 12}

        event_log = build_inference_event_log(args, packet, prediction, boxes=[], sequence=sequence)

        self.assertEqual(event_log["frame_idx"], 12)
        self.assertEqual(event_log["sequence_window"], {"start": 4, "end": 12})
        self.assertEqual(event_log["probabilities"], {"Normal": 0.19, "Faint": 0.81})
        self.assertEqual(event_log["threshold"], 0.3)
        self.assertEqual(event_log["faint_prob"], 0.81)
        self.assertEqual(event_log["post_processing"]["min_consecutive_faint"], 3)

    def test_save_inference_event_log_writes_one_json_file(self):
        event_log = {"camera_id": "cam/01", "frame_idx": 12, "timestamp": 123.5, "event_type": "Faint", "track_id": 9}
        with tempfile.TemporaryDirectory() as tmp:
            path = save_inference_event_log(tmp, event_log)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["camera_id"], "cam/01")
        self.assertIn("cam_01_Faint_123.5_track-9_frame-12", path.name)

    def test_event_log_dir_writes_event_file_but_output_remains_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_log_dir = Path(tmp) / "events"
            args = fake_run_args(event_log_dir=str(event_log_dir))
            summary = run_with_fake_rtsp(args)
            event_logs = list(event_log_dir.glob("*.json"))
            event_log = json.loads(event_logs[0].read_text(encoding="utf-8"))

        self.assertEqual(summary["events_generated"], 1)
        self.assertIsNotNone(summary["sample_event"])
        self.assertEqual(len(event_logs), 1)
        self.assertIn("probabilities", event_log)
        self.assertNotIn("probabilities", summary["sample_event"])

    def test_event_log_file_is_not_written_without_event_log_dir(self):
        calls = []
        original_save = rtsp_module.save_inference_event_log
        try:
            rtsp_module.save_inference_event_log = lambda *args: calls.append(args)
            summary = run_with_fake_rtsp(fake_run_args(event_log_dir=None))
        finally:
            rtsp_module.save_inference_event_log = original_save

        self.assertEqual(calls, [])
        self.assertEqual(summary["events_generated"], 1)

    def test_write_run_summary_keeps_output_as_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "summary.json"
            summary = {"events_generated": 1, "sample_event": {"camera_id": "cam_01"}}

            write_run_summary(output_path, summary)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, summary)

    def test_dry_run_defaults_to_console_publisher(self):
        publisher, mode = create_event_publisher(fake_run_args(event_log_dir=None))

        self.assertEqual(mode, "console")
        self.assertTrue(hasattr(publisher, "publish"))

    def test_non_dry_run_uses_mqtt_publisher_without_changing_event_logic(self):
        published = []
        original_publisher = publisher_module.MqttEventPublisher
        try:
            publisher_module.MqttEventPublisher = lambda **unused_kwargs: FakeMqttPublisher(published)
            summary = run_with_fake_rtsp(fake_run_args(event_log_dir=None, dry_run=False))
        finally:
            publisher_module.MqttEventPublisher = original_publisher

        self.assertEqual(summary["alert_delivery_result"], "mqtt")
        self.assertEqual(summary["events_generated"], 1)
        self.assertEqual(len(published), 1)
        self.assertNotIn("probabilities", published[0])
        self.assertEqual(published[0]["type"], "fall_detected")
        self.assertIn("metadata", published[0])


def fake_run_args(event_log_dir, dry_run=True):
    return Namespace(
        rtsp_url="fake://cam1",
        camera_id="cam_01",
        camera_login_id=None,
        max_frames=4,
        detector_mode="mock",
        dry_run=dry_run,
        output=None,
        overlay_output=None,
        yolo_model="yolov8n-pose.pt",
        device="auto",
        action_model=None,
        action_device="auto",
        action_threshold=0.3,
        min_consecutive_faint=3,
        camera_cooldown_seconds=10,
        event_severity="HIGH",
        classifier_input="keypoints",
        sequence_length=2,
        sequence_stride=1,
        resize_size=32,
        tracker_iou_threshold=0.3,
        track_max_missing_seconds=2.0,
        event_log_dir=event_log_dir,
        publisher=None,
        mqtt_host=None,
        mqtt_port=None,
        mqtt_topic=None,
        mqtt_client_id=None,
        mqtt_username=None,
        mqtt_password=None,
    )


def run_with_fake_rtsp(args):
    original_video_reader = rtsp_module.VideoReader
    original_create_detector = rtsp_module.create_detector
    original_create_classifier = rtsp_module.create_classifier
    try:
        rtsp_module.VideoReader = FakeVideoReader
        rtsp_module.create_detector = lambda *unused_args, **unused_kwargs: FakeDetector()
        rtsp_module.create_classifier = lambda *unused_args, **unused_kwargs: (FakeClassifier(), "fake_lstm")
        return run(args)
    finally:
        rtsp_module.VideoReader = original_video_reader
        rtsp_module.create_detector = original_create_detector
        rtsp_module.create_classifier = original_create_classifier


class FakePacket:
    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
        self.timestamp = float(frame_idx)
        self.fps = 10.0
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)


class FakeVideoReader:
    def __init__(self, unused_url):
        self.frame_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        if self.frame_idx >= 4:
            return None
        packet = FakePacket(self.frame_idx)
        self.frame_idx += 1
        return packet


class FakeDetector:
    def detect(self, unused_frame):
        return [
            {
                "bbox": [1, 2, 20, 30],
                "confidence": 0.9,
                "keypoints": [{"x": 5, "y": 6, "confidence": 0.9} for _ in range(17)],
            }
        ]


class FakeClassifier:
    def predict(self, unused_sequence):
        return {"label": "Faint", "score": 0.8, "probabilities": {"Normal": 0.2, "Faint": 0.8}}


class FakeMqttPublisher:
    def __init__(self, published):
        self.published = published

    def connect(self):
        return True

    def publish(self, payload):
        self.published.append(payload)
        return True

    def close(self):
        return None


