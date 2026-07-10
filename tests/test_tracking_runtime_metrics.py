"""Tracking runtime metrics collector isolation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai.tracking_runtime_metrics import TrackingRuntimeMetrics


class TrackingRuntimeMetricsTest(unittest.TestCase):
    def test_counters_and_average_duration(self):
        m = TrackingRuntimeMetrics()
        m.on_track_created(1, now=1.0)
        m.on_track_created(2, now=1.0)
        m.on_match("hard")
        m.on_match("soft")
        m.on_match("sole")
        m.on_match_rejected("low_iou")
        m.on_fragmentation_suspected(2)
        m.on_track_removed(1, now=3.0)
        summary = m.summary()
        self.assertEqual(summary["track_created_total"], 2)
        self.assertEqual(summary["track_removed_total"], 1)
        self.assertEqual(summary["match_hard_total"], 1)
        self.assertEqual(summary["match_soft_total"], 1)
        self.assertEqual(summary["match_sole_total"], 1)
        self.assertEqual(summary["match_rejected_total"], 1)
        self.assertEqual(summary["suspected_fragmentation_total"], 2)
        self.assertAlmostEqual(summary["average_track_duration"], 2.0)
        self.assertEqual(summary["rejection_reasons"]["low_iou"], 1)

    def test_ingest_tracker_events(self):
        m = TrackingRuntimeMetrics()
        m.ingest_tracker_events(
            [
                {"event": "new_track", "trackId": 7},
                {"event": "match", "reason": "soft_center"},
                {"event": "id_switch_like"},
                {"event": "reject", "reason": "filtered_small"},
                {"event": "lost", "trackId": 7},
            ],
            now=5.0,
        )
        s = m.summary()
        self.assertEqual(s["track_created_total"], 1)
        self.assertEqual(s["match_soft_total"], 1)
        self.assertEqual(s["suspected_fragmentation_total"], 1)
        self.assertEqual(s["match_rejected_total"], 1)
        self.assertEqual(s["track_removed_total"], 1)

    def test_ingest_simple_tracker_filter_events(self):
        """SimpleTrackAssigner emits event='filter', not 'reject'/'filtered'."""
        from tracking.simple_tracker import SimpleTrackAssigner

        tracker = SimpleTrackAssigner(track_thresh=0.5, min_box_area=100)
        tracker.update(
            [
                {"bbox": [0, 0, 50, 50], "confidence": 0.9},  # area 2500 ok
                {"bbox": [10, 10, 60, 120], "confidence": 0.1},  # low_confidence
                {"bbox": [1, 1, 2, 2], "confidence": 0.9},  # tiny_box
            ],
            now=1.0,
        )
        m = TrackingRuntimeMetrics()
        m.ingest_tracker_events(tracker.last_events, now=1.0)
        s = m.summary()
        self.assertGreaterEqual(s["match_rejected_total"], 2)
        self.assertIn("low_confidence", s["rejection_reasons"])
        self.assertIn("tiny_box", s["rejection_reasons"])

    def test_record_safe_never_raises(self):
        m = TrackingRuntimeMetrics()
        # missing method path
        m.record_safe("does_not_exist", 1)
        # broken callback via direct
        m.record_safe("on_track_created")  # missing track_id -> TypeError swallowed
        self.assertEqual(m.track_created_total, 0)

    def test_save_session_summary_json_csv(self):
        m = TrackingRuntimeMetrics()
        m.on_track_created(1, now=0.0)
        m.on_match("hard")
        with tempfile.TemporaryDirectory() as tmp:
            paths = m.save_session_summary(tmp, run_id="sess-1")
            self.assertIn("json", paths)
            self.assertIn("csv", paths)
            data = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(data["runId"], "sess-1")
            self.assertEqual(data["track_created_total"], 1)
            self.assertTrue(paths["csv"].is_file())

    def test_save_failure_is_warning_only(self):
        m = TrackingRuntimeMetrics()
        # invalid path characters on Windows / non-writable
        paths = m.save_session_summary("\x00invalid" if False else Path("/nonexistent_root_dir_xyz/nope"), run_id="x")
        # may return empty on failure
        self.assertIsInstance(paths, dict)


if __name__ == "__main__":
    unittest.main()
