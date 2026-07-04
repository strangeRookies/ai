import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_pose_tracking_diag import summarize_jsonl_path


class PoseTrackingDiagSummaryTest(unittest.TestCase):
    def test_summarizes_camera_metrics_from_jsonl(self):
        records = [
            {
                "stage": "pose_frame",
                "cameraLoginId": "cam_04",
                "sourceUrl": "rtsp://shared",
                "assignedVideoPath": "shared.mp4",
                "frameId": 1,
                "raw_detection_count": 1,
                "avg_bbox_confidence": 0.9,
                "avg_keypoint_confidence": 0.8,
                "active_tracks": 1,
                "sequenceReadyCount": 1,
            },
            {
                "stage": "pose_frame",
                "cameraLoginId": "cam_05",
                "sourceUrl": "rtsp://shared",
                "assignedVideoPath": "shared.mp4",
                "frameId": 1,
                "raw_detection_count": 0,
                "avg_bbox_confidence": None,
                "avg_keypoint_confidence": None,
                "active_tracks": 0,
                "sequenceReadyCount": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diag.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            summary = summarize_jsonl_path(path)

        self.assertEqual(summary["cam_04"]["frames"], 1)
        self.assertEqual(summary["cam_04"]["avg_raw_detection_count"], 1.0)
        self.assertEqual(summary["cam_04"]["avg_bbox_confidence"], 0.9)
        self.assertEqual(summary["cam_04"]["avg_keypoint_confidence"], 0.8)
        self.assertEqual(summary["cam_04"]["tracker_active_rate"], 1.0)
        self.assertEqual(summary["cam_04"]["sequence_ready_count"], 1)
        self.assertEqual(summary["cam_05"]["tracker_active_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
