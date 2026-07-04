import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai.inference.pose_diagnostics import (
    PoseDiagnosticsConfig,
    PoseDiagnosticsReporter,
    build_pose_diagnostic_record,
    pose_debug_enabled,
)


class PoseDiagnosticsTest(unittest.TestCase):
    def test_diagnoses_detector_miss_when_no_raw_detection_and_no_active_tracks(self):
        record = build_pose_diagnostic_record(
            camera_login_id="cam_04",
            source_url="rtsp://localhost:8554/cam_04",
            assigned_video_path="same.mp4",
            frame_id=12,
            timestamp_ms=1234000,
            raw_detections=[],
            tracker_diagnostics={"active_tracks": 0},
            sequence_ready_count=0,
        )

        self.assertEqual(record["diagnosis"], "detector_person_missing")
        self.assertEqual(record["raw_detection_count"], 0)
        self.assertEqual(record["active_tracks"], 0)

    def test_diagnoses_pose_quality_when_keypoint_confidence_is_low(self):
        record = build_pose_diagnostic_record(
            camera_login_id="cam_04",
            source_url="rtsp://localhost:8554/cam_04",
            assigned_video_path="same.mp4",
            frame_id=13,
            timestamp_ms=1235000,
            raw_detections=[
                {
                    "bbox": [10, 20, 110, 220],
                    "confidence": 0.82,
                    "keypoints": [{"x": 1, "y": 2, "confidence": 0.1} for _ in range(17)],
                }
            ],
            tracker_diagnostics={"active_tracks": 1},
            sequence_ready_count=0,
            min_keypoint_confidence=0.25,
        )

        self.assertEqual(record["diagnosis"], "pose_quality_low")
        self.assertEqual(record["avg_keypoint_confidence"], 0.1)
        self.assertEqual(record["valid_keypoint_count"], 0)
        self.assertEqual(record["missing_keypoint_count"], 17)

    def test_diagnoses_tracking_association_when_pose_is_good_but_tracker_empty(self):
        record = build_pose_diagnostic_record(
            camera_login_id="cam_05",
            source_url="rtsp://localhost:8554/cam_05",
            assigned_video_path="same.mp4",
            frame_id=14,
            timestamp_ms=1236000,
            raw_detections=[
                {
                    "bbox": [10, 20, 110, 220],
                    "confidence": 0.91,
                    "keypoints": [{"x": 1, "y": 2, "confidence": 0.8} for _ in range(17)],
                }
            ],
            tracker_diagnostics={"active_tracks": 0},
            sequence_ready_count=0,
        )

        self.assertEqual(record["diagnosis"], "tracking_association_problem")
        self.assertEqual(record["raw_detection_count"], 1)
        self.assertEqual(record["avg_bbox_confidence"], 0.91)
        self.assertEqual(record["avg_keypoint_confidence"], 0.8)

    def test_reporter_prints_dynamic_camera_summary_and_comparison_ready_fields(self):
        output = StringIO()
        reporter = PoseDiagnosticsReporter(
            PoseDiagnosticsConfig(
                enabled=True,
                summary_every_n=2,
                image_output_enabled=False,
            )
        )

        with redirect_stdout(output):
            reporter.observe(
                camera_login_id="cam_04",
                source_url="rtsp://localhost:8554/cam_04",
                assigned_video_path="same.mp4",
                frame_id=1,
                timestamp_ms=1000,
                raw_detections=[
                    {
                        "bbox": [0, 0, 100, 200],
                        "confidence": 0.9,
                        "keypoints": [{"x": 1, "y": 1, "confidence": 0.9} for _ in range(17)],
                    }
                ],
                tracker_diagnostics={"active_tracks": 1},
                sequence_ready_count=1,
            )
            reporter.observe(
                camera_login_id="cam_05",
                source_url="rtsp://localhost:8554/cam_05",
                assigned_video_path="same.mp4",
                frame_id=1,
                timestamp_ms=1000,
                raw_detections=[],
                tracker_diagnostics={"active_tracks": 0},
                sequence_ready_count=0,
            )

        lines = [json.loads(line.removeprefix("[pose-diagnostics] ")) for line in output.getvalue().splitlines()]
        frame_records = [line for line in lines if line["stage"] == "pose_frame"]
        summary_records = [line for line in lines if line["stage"] == "pose_summary"]

        self.assertEqual({item["cameraLoginId"] for item in frame_records}, {"cam_04", "cam_05"})
        self.assertEqual(frame_records[0]["assignedVideoPath"], "same.mp4")
        self.assertEqual(frame_records[0]["sequenceReadyCount"], 1)
        self.assertEqual(summary_records[-1]["cameras"]["cam_04"]["tracker_active_rate"], 1.0)
        self.assertEqual(summary_records[-1]["cameras"]["cam_05"]["frames_without_person"], 1)

    def test_sample_image_save_is_default_off_and_explicit_opt_in(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detections = [{"bbox": [4, 4, 20, 28], "keypoints": [{"x": 8, "y": 8, "confidence": 0.9}]}]
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = PoseDiagnosticsReporter(
                PoseDiagnosticsConfig(
                    enabled=True,
                    image_output_enabled=False,
                    image_output_dir=Path(temp_dir),
                )
            )
            self.assertIsNone(reporter.save_sample_image("cam_04", 1, 1000, frame, detections))
            self.assertEqual(list(Path(temp_dir).glob("*.jpg")), [])

            enabled_reporter = PoseDiagnosticsReporter(
                PoseDiagnosticsConfig(
                    enabled=True,
                    image_output_enabled=True,
                    image_output_dir=Path(temp_dir),
                )
            )
            path = enabled_reporter.save_sample_image("cam_04", 1, 1000, frame, detections)

        self.assertIsNotNone(path)

    def test_pose_debug_enabled_uses_pose_or_tracking_flags(self):
        with patch.dict(os.environ, {"POSE_DEBUG": "true"}, clear=True):
            self.assertTrue(pose_debug_enabled())
        with patch.dict(os.environ, {"TRACKING_DEBUG": "true"}, clear=True):
            self.assertTrue(pose_debug_enabled())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(pose_debug_enabled())


if __name__ == "__main__":
    unittest.main()
