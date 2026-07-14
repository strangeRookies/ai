import tempfile
import unittest
from pathlib import Path

from scripts.replay_tracking_from_cache import TrackerConfig, run_tracker_config


class ReplayTimingTest(unittest.TestCase):
    def test_replay_uses_source_fps_for_tracker_timing(self):
        frames = [{"frame_id": 0, "timestamp_ms": 0, "detections": []}]
        with tempfile.TemporaryDirectory() as directory:
            summary = run_tracker_config(
                frames, TrackerConfig(name="fps"), Path(directory), source_fps=12.5
            )
        self.assertEqual(summary["tracker_assumed_fps"], 12.5)
