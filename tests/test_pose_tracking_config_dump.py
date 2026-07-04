import unittest
from argparse import Namespace

from ai.inference.rtsp_runtime import build_pose_tracking_config_dump


class PoseTrackingConfigDumpTest(unittest.TestCase):
    def test_config_dump_reports_effective_values_and_ignored_bytetrack_kwargs(self):
        args = Namespace(
            track_thresh=0.11,
            match_thresh=0.22,
            track_buffer=33,
            frame_rate=24,
            tracking_grace_period_seconds=3.5,
            tracking_relink_iou_threshold=0.44,
            tracking_relink_center_ratio=0.55,
            tracking_relink_max_time_gap_seconds=1.5,
            pose_min_keypoint_confidence=0.31,
            pose_debug_summary_every_n=9,
            pose_debug_save_images=True,
            pose_tracking_diag_jsonl=True,
            pose_tracking_diag_jsonl_path="runs/diagnostics/custom.jsonl",
        )
        tracker_diagnostics = {
            "bytetrack_constructor": {
                "used": {"track_activation_threshold": 0.11},
                "ignored": {"track_buffer": 33},
                "supported_parameters": ["track_activation_threshold"],
            }
        }

        record = build_pose_tracking_config_dump(args, tracker_diagnostics)

        self.assertEqual(record["stage"], "pose_tracking_config")
        self.assertEqual(record["TRACK_THRESH"], 0.11)
        self.assertEqual(record["TRACK_IOU_THRESHOLD"], 0.22)
        self.assertEqual(record["TRACK_BUFFER"], 33)
        self.assertEqual(record["TRACK_FRAME_RATE"], 24)
        self.assertEqual(record["TRACKING_GRACE_PERIOD_SECONDS"], 3.5)
        self.assertEqual(record["TRACKING_RELINK_IOU_THRESHOLD"], 0.44)
        self.assertEqual(record["TRACKING_RELINK_CENTER_RATIO"], 0.55)
        self.assertEqual(record["TRACKING_RELINK_MAX_TIME_GAP_SECONDS"], 1.5)
        self.assertEqual(record["POSE_MIN_KEYPOINT_CONFIDENCE"], 0.31)
        self.assertEqual(record["POSE_DEBUG_SUMMARY_EVERY_N"], 9)
        self.assertTrue(record["POSE_DEBUG_SAVE_IMAGES"])
        self.assertTrue(record["POSE_TRACKING_DIAG_JSONL"])
        self.assertEqual(record["POSE_TRACKING_DIAG_JSONL_PATH"], "runs/diagnostics/custom.jsonl")
        self.assertEqual(record["bytetrack_constructor_ignored"], {"track_buffer": 33})


if __name__ == "__main__":
    unittest.main()
