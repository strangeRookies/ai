import tempfile
import unittest
from pathlib import Path

from scripts.eval_two_person_synthetic_gt import evaluate, make_frames
from scripts.replay_tracking_from_cache import CONFIGS, TrackerConfig, run_tracker_config


class TrackingPriorityMetricsTest(unittest.TestCase):
    def test_hybrid_kp_two_person_analysis_reports_identity_switches(self):
        frames = make_frames(n=40, fps=30.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hybrid"
            run_tracker_config(frames, CONFIGS["M_I_hybrid_kp_safe"], output, source_fps=30.0)
            result = evaluate(output / "frame_results.jsonl", frames)
        self.assertIn("person_identity_switch_count", result)
        self.assertIn("track_owner_switch_count", result)
        self.assertIn("person_identity_switches", result)
        self.assertEqual(set(result["person_identity_switches"]), {"P1", "P2"})

    def test_replay_separates_unique_ghost_tracks_from_ghost_frames(self):
        frames = [{"frame_id": 0, "timestamp_ms": 0, "detections": [{"bbox_xyxy": [0, 0, 20, 60], "confidence": 0.9}]}]
        frames.extend({"frame_id": index, "timestamp_ms": index * 33, "detections": []} for index in range(1, 50))
        config = TrackerConfig(name="ghost", track_buffer=90, max_missing_seconds=4.0)
        with tempfile.TemporaryDirectory() as directory:
            summary = run_tracker_config(frames, config, Path(directory), source_fps=30.0)
        self.assertEqual(summary["unique_ghost_tracks"], 1)
        self.assertGreater(summary["ghost_frame_count"], 0)
        self.assertEqual(summary["ghost_count"], summary["ghost_frame_count"])


if __name__ == "__main__":
    unittest.main()
