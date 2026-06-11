import os
import unittest
from argparse import Namespace

import numpy as np

import scripts.run_rtsp_inference as rtsp_module
from scripts.run_rtsp_inference import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    parse_args,
    run,
)


class RtspInferenceConfigTest(unittest.TestCase):
    def test_defaults_use_error_augmented_lstm_and_three_consecutive_faint(self):
        self.assertEqual(
            DEFAULT_ACTION_MODEL,
            "benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt",
        )
        self.assertEqual(DEFAULT_MIN_CONSECUTIVE_FAINT, 3)

    def test_parse_args_reads_lstm_post_processing_from_env(self):
        original_env = {
            "ACTION_MODEL": os.environ.get("ACTION_MODEL"),
            "ACTION_THRESHOLD": os.environ.get("ACTION_THRESHOLD"),
            "MIN_CONSECUTIVE_FAINT": os.environ.get("MIN_CONSECUTIVE_FAINT"),
            "CAMERA_COOLDOWN_SECONDS": os.environ.get("CAMERA_COOLDOWN_SECONDS"),
        }
        try:
            os.environ["ACTION_MODEL"] = "custom/best.pt"
            os.environ["ACTION_THRESHOLD"] = "0.31"
            os.environ["MIN_CONSECUTIVE_FAINT"] = "4"
            os.environ["CAMERA_COOLDOWN_SECONDS"] = "11"

            args = parse_args([])
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(args.action_model, "custom/best.pt")
        self.assertEqual(args.action_threshold, 0.31)
        self.assertEqual(args.min_consecutive_faint, 4)
        self.assertEqual(args.camera_cooldown_seconds, 11.0)

    def test_preflight_reports_model_paths_and_post_processing_without_rtsp_or_mqtt(self):
        args = fake_run_args()
        args.preflight_only = True
        args.action_model = "custom/best.pt"
        args.yolo_model = "yolo26n-pose.pt"
        original_video_reader = rtsp_module.VideoReader
        original_create_detector = rtsp_module.create_detector
        original_create_classifier = rtsp_module.create_classifier
        original_create_event_publisher = rtsp_module.create_event_publisher
        try:
            rtsp_module.VideoReader = FailIfCalled
            rtsp_module.create_detector = lambda *unused_args, **unused_kwargs: FakeDetector()
            rtsp_module.create_classifier = lambda *unused_args, **unused_kwargs: (FakeClassifier(), "fake_lstm")
            rtsp_module.create_event_publisher = FailIfCalled

            summary = run(args)
        finally:
            rtsp_module.VideoReader = original_video_reader
            rtsp_module.create_detector = original_create_detector
            rtsp_module.create_classifier = original_create_classifier
            rtsp_module.create_event_publisher = original_create_event_publisher

        self.assertEqual(summary["alert_delivery_result"], "preflight")
        self.assertEqual(summary["action_model"], "custom/best.pt")
        self.assertEqual(summary["yolo_model"], "yolo26n-pose.pt")
        self.assertEqual(summary["action_threshold"], 0.3)
        self.assertEqual(summary["min_consecutive_faint"], 3)
        self.assertEqual(summary["camera_cooldown_seconds"], 10)
        self.assertEqual(summary["sequence_length"], 2)
        self.assertEqual(summary["sequence_stride"], 1)


def fake_run_args():
    return Namespace(
        rtsp_url="fake://cam1",
        camera_id="cam_01",
        max_frames=4,
        detector_mode="mock",
        dry_run=True,
        output=None,
        overlay_output=None,
        yolo_model="yolo26n-pose.pt",
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
        event_log_dir=None,
        publisher=None,
        mqtt_host=None,
        mqtt_port=None,
        mqtt_topic=None,
        mqtt_client_id=None,
        mqtt_username=None,
        mqtt_password=None,
    )


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


class FailIfCalled:
    def __init__(self, *unused_args, **unused_kwargs):
        raise AssertionError("preflight must not open RTSP or MQTT")
