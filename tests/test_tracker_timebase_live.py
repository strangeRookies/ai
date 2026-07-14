import csv
import tempfile
import unittest
from pathlib import Path

from ai.inference.tracker_timebase import TrackerUpdateFpsEstimator
from ai.inference.rtsp_runtime import apply_tracker_timebase
from scripts.replay_tracking_from_cache import PRESETS, TrackerConfig, run_tracker_config, write_comparison
from tracking.simple_tracker import SimpleTrackAssigner, predicted_bbox


class TrackerTimebaseLiveTest(unittest.TestCase):
    def test_latest_cadence_updates_only_after_stable_windows(self):
        estimator = TrackerUpdateFpsEstimator(30.0, 30.0, True, min_samples=4, stable_windows_required=3)
        for index in range(10):
            estimator.observe_update(index / 15.0)
        estimator.record_window(15.0)
        estimator.record_window(15.0)
        self.assertFalse(estimator.should_apply(30.0))
        estimator.record_window(15.0)
        self.assertEqual(estimator.state, "measured_tracker_update_fps")
        self.assertAlmostEqual(estimator.effective_fps, 15.0, places=1)
        self.assertTrue(estimator.should_apply(30.0))

    def test_source_rate_cadence_does_not_reconfigure_tracker(self):
        estimator = TrackerUpdateFpsEstimator(30.0, 30.0, True, min_samples=4, stable_windows_required=3)
        for index in range(10):
            estimator.observe_update(index / 30.0)
        for _ in range(3):
            estimator.record_window(30.0)
        self.assertFalse(estimator.should_apply(30.0))

    def test_in_place_reconfiguration_preserves_tracker_object(self):
        tracker = SimpleTrackAssigner(track_buffer=90, assumed_fps=30.0)
        original_id = id(tracker)
        self.assertTrue(apply_tracker_timebase(tracker, 15.0))
        self.assertEqual(id(tracker), original_id)
        self.assertEqual(tracker.track_buffer, 45)

    def test_sequential_keeps_source_rate_and_reconnect_resets_samples(self):
        estimator = TrackerUpdateFpsEstimator(15.0, 30.0, False)
        for index in range(10):
            estimator.observe_update(index / 15.0)
        self.assertEqual(estimator.effective_fps, 15.0)
        self.assertEqual(estimator.state, "source_fps")
        live = TrackerUpdateFpsEstimator(30.0, 30.0, True, min_samples=2)
        for index in range(4):
            live.observe_update(index / 15.0)
        live.reset(15.0)
        self.assertEqual(live.sample_count, 0)
        self.assertEqual(live.effective_fps, 15.0)

    def test_transient_spike_does_not_authorize_reconfiguration(self):
        estimator = TrackerUpdateFpsEstimator(30.0, 30.0, True, min_samples=4, stable_windows_required=3)
        for index in range(10):
            estimator.observe_update(index / 30.0)
        estimator.record_window(8.0)
        estimator.record_window(29.0)
        self.assertFalse(estimator.should_apply(30.0))

    def test_prediction_fallback_uses_effective_track_fps(self):
        common = {"bbox": [0, 0, 10, 10], "velocity": [10, 0, 10, 0]}
        at_30 = predicted_bbox({**common, "missing_frames": 15, "assumed_fps": 30.0}, max_predict_seconds=1.0)
        at_15 = predicted_bbox({**common, "missing_frames": 8, "assumed_fps": 15.0}, max_predict_seconds=1.0)
        self.assertAlmostEqual(at_30[0], 5.0, places=2)
        self.assertAlmostEqual(at_15[0], at_30[0], delta=0.4)

    def test_buffer_seconds_are_preserved(self):
        tracker = SimpleTrackAssigner(track_buffer=90, assumed_fps=30.0, max_missing_seconds=3.0)
        self.assertTrue(tracker.set_assumed_fps(15.0))
        self.assertEqual(tracker.track_buffer, 45)
        self.assertEqual(tracker.max_missing_seconds, 3.0)

    def test_replay_uses_metadata_fps_independent_of_resolution(self):
        frames = [{"frame_id": 0, "timestamp_ms": 0, "detections": []}]
        for width, fps in ((1280, 15.0), (1280, 30.0), (1920, 15.0)):
            with tempfile.TemporaryDirectory() as directory:
                summary = run_tracker_config(frames, TrackerConfig(name=str(width)), Path(directory), fps)
            self.assertEqual(summary["source_fps"], fps)
            self.assertEqual(summary["tracker_assumed_fps"], fps)
            self.assertEqual(summary["tracker_fps_source"], "cache_metadata")
        with tempfile.TemporaryDirectory() as directory:
            summary = run_tracker_config(frames, TrackerConfig(name="fallback", assumed_fps=20.0), Path(directory), 0.0)
        self.assertEqual(summary["tracker_assumed_fps"], 20.0)
        self.assertEqual(summary["tracker_fps_source"], "configured_fallback")

    def test_threshold_preset_and_report_columns_exist(self):
        self.assertEqual(PRESETS["new-track-thresholds"], ("A_current", "B_new_thresh_020", "C_new_thresh_030"))
        frames = [
            {"frame_id": 0, "timestamp_ms": 0, "detections": [{"bbox_xyxy": [0, 0, 10, 20], "confidence": 0.9}]},
            {"frame_id": 1, "timestamp_ms": 33, "detections": [{"bbox_xyxy": [1, 0, 11, 20], "confidence": 0.9}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_tracker_config(frames, TrackerConfig(name="A_current"), root / "A_current", 30.0)
            write_comparison(root, [summary])
            with (root / "comparison_summary.csv").open(newline="", encoding="utf-8") as handle:
                columns = csv.DictReader(handle).fieldnames or []
        for column in ("total_id_switch_events", "switch_reason_iou_below_threshold", "fragmentation_count", "hard_match_rate", "sequence_completion_rate", "incomplete_insufficient_track_frames", "maximum_frame_gap"):
            self.assertIn(column, columns)


if __name__ == "__main__":
    unittest.main()
