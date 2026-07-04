import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ai.inference.pose_diagnostics import PoseDiagnosticsReporter, config_from_args


class PoseDiagnosticsJsonlTest(unittest.TestCase):
    def test_config_from_args_enables_jsonl_without_console_debug(self):
        args = Namespace(
            pose_debug=False,
            pose_debug_summary_every_n=60,
            pose_min_keypoint_confidence=0.25,
            pose_debug_save_images=False,
            pose_debug_image_dir="runs/pose_debug",
            pose_debug_image_every_n=300,
            pose_tracking_diag_jsonl=True,
            pose_tracking_diag_jsonl_path="runs/diagnostics/test.jsonl",
        )

        config = config_from_args(args)

        self.assertFalse(config.enabled)
        self.assertTrue(config.jsonl_enabled)
        self.assertEqual(config.jsonl_path, Path("runs/diagnostics/test.jsonl"))

    def test_config_from_args_enables_console_when_tracking_debug_env_is_set(self):
        args = Namespace(
            pose_debug=False,
            pose_debug_summary_every_n=60,
            pose_min_keypoint_confidence=0.25,
            pose_debug_save_images=False,
            pose_debug_image_dir="runs/pose_debug",
            pose_debug_image_every_n=300,
            pose_tracking_diag_jsonl=False,
            pose_tracking_diag_jsonl_path="runs/diagnostics/test.jsonl",
        )

        with patch.dict("os.environ", {"TRACKING_DEBUG": "true"}, clear=True):
            config = config_from_args(args)

        self.assertTrue(config.enabled)

    def test_reporter_writes_minimal_jsonl_fields_and_extended_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "diag.jsonl"
            args = Namespace(
                pose_debug=False,
                pose_debug_summary_every_n=2,
                pose_min_keypoint_confidence=0.25,
                pose_debug_save_images=False,
                pose_debug_image_dir="runs/pose_debug",
                pose_debug_image_every_n=300,
                pose_tracking_diag_jsonl=True,
                pose_tracking_diag_jsonl_path=str(jsonl_path),
            )
            reporter = PoseDiagnosticsReporter(config_from_args(args))

            reporter.observe(
                camera_login_id="cam_04",
                source_url="rtsp://localhost:8554/shared",
                assigned_video_path="shared.mp4",
                frame_id=10,
                timestamp_ms=10000,
                raw_detections=[
                    {
                        "bbox": [10, 20, 110, 220],
                        "confidence": 0.91,
                        "keypoints": [{"x": 1, "y": 2, "confidence": 0.8} for _ in range(17)],
                    }
                ],
                tracker_diagnostics={
                    "active_tracks": 1,
                    "tracks": {"7": {"track_id": 7}},
                    "id_switch_like_events": 1,
                },
                sequence_ready_count=1,
                sequence_diagnostics={"relink_success_count": 2, "relink_fail_count": 1},
            )
            reporter.observe(
                camera_login_id="cam_04",
                source_url="rtsp://localhost:8554/shared",
                assigned_video_path="shared.mp4",
                frame_id=11,
                timestamp_ms=10033,
                raw_detections=[],
                tracker_diagnostics={"active_tracks": 0, "tracks": {}},
                sequence_ready_count=0,
            )

            lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            summary = reporter.log_summary()["cameras"]["cam_04"]
            final_summary = reporter.log_final_summary()["cameras"]["cam_04"]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["stage"], "pose_frame")
        self.assertEqual(lines[0]["cameraLoginId"], "cam_04")
        self.assertEqual(lines[0]["sourceUrl"], "rtsp://localhost:8554/shared")
        self.assertEqual(lines[0]["assignedVideoPath"], "shared.mp4")
        self.assertEqual(lines[0]["frameId"], 10)
        self.assertEqual(lines[0]["timestampMs"], 10000)
        self.assertEqual(lines[0]["raw_detection_count"], 1)
        self.assertEqual(lines[0]["avg_bbox_confidence"], 0.91)
        self.assertEqual(lines[0]["avg_keypoint_confidence"], 0.8)
        self.assertEqual(lines[0]["valid_keypoint_count"], 17)
        self.assertEqual(lines[0]["active_tracks"], 1)
        self.assertEqual(lines[0]["track_ids"], [7])
        self.assertEqual(lines[0]["sequenceReadyCount"], 1)
        self.assertEqual(lines[0]["diagnosis"], "pose_tracking_ok")
        self.assertEqual(summary["active_tracks_zero_count"], 1)
        self.assertEqual(summary["relink_success_count"], 2)
        self.assertEqual(summary["relink_fail_count"], 1)
        self.assertEqual(final_summary["issue_counts"]["detector_missing"], 1)
        self.assertEqual(final_summary["issue_counts"]["normal"], 1)


if __name__ == "__main__":
    unittest.main()
