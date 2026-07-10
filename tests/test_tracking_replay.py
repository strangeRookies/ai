"""Tracking replay harness + synthetic fixture tests (not real video performance)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai.tracking_replay import DISCLAIMER, ReplayFrame, load_detections_jsonl, run_tracking_replay
from tracking.simple_tracker import SimpleTrackAssigner


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "tracking" / "synthetic_scenarios.jsonl"


class TrackingReplayTest(unittest.TestCase):
    def test_load_and_replay_writes_artifacts(self):
        self.assertTrue(FIXTURE.is_file())
        frames = load_detections_jsonl(FIXTURE)
        self.assertGreaterEqual(len(frames), 10)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tracking_replay(frames, output_dir=Path(tmp) / "run1", run_id="test-run-1")
            out = Path(result["outputDir"])
            for name in (
                "manifest.json",
                "frame-results.jsonl",
                "track-lifecycle.csv",
                "match-decisions.csv",
                "metrics.json",
                "report.md",
            ):
                self.assertTrue((out / name).is_file(), name)
            metrics = result["metrics"]
            self.assertEqual(metrics["disclaimer"], DISCLAIMER)
            self.assertIn("synthetic_id_switch_count", metrics)
            self.assertIn("fragmentation_count", metrics)
            self.assertIn("match_hard_count", metrics)
            self.assertIn("unmatched_detection_count", metrics)
            # Real tracker path must surface filter→rejection metrics from fixture
            self.assertGreater(metrics["runtime_metrics"]["match_rejected_total"], 0)
            self.assertTrue(metrics["rejection_reason_distribution"])
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("실제 영상 성능이 아님", report)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runId"], "test-run-1")

    def test_video_boundary_resets_session(self):
        frames = load_detections_jsonl(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tracking_replay(frames, output_dir=Path(tmp) / "b", run_id="boundary")
            # begin_session already resets once; boundary adds another
            self.assertGreaterEqual(result["metrics"]["session_reset_count"], 2)
            rows = [
                json.loads(line)
                for line in (Path(result["outputDir"]) / "frame-results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stream_ids = {row["streamRunId"] for row in rows}
            self.assertGreaterEqual(len(stream_ids), 2)
            # video_b frames use different streamRunId than video_a
            a_streams = {r["streamRunId"] for r in rows if r.get("sourceId") == "video_a"}
            b_streams = {r["streamRunId"] for r in rows if r.get("sourceId") == "video_b"}
            self.assertTrue(a_streams)
            self.assertTrue(b_streams)
            self.assertTrue(a_streams.isdisjoint(b_streams))

    def test_reacquisition_not_counted_after_boundary_id_reuse(self):
        """lost track id reuse after session reset must not inflate reacquisition_count."""
        frames = [
            ReplayFrame(0, [{"bbox": [10, 10, 60, 120], "confidence": 0.9}], now=1.0, source_id="video_a"),
            # force lost via empty frames past max_missing
            ReplayFrame(1, [], now=20.0, source_id="video_a"),
            ReplayFrame(2, [], now=40.0, source_id="video_a"),
            # new video; track_id will restart at 1
            ReplayFrame(
                3,
                [{"bbox": [10, 10, 60, 120], "confidence": 0.95}],
                now=100.0,
                source_id="video_b",
                video_boundary=True,
            ),
        ]
        tracker = SimpleTrackAssigner(
            match_thresh=0.2,
            track_buffer=1,
            min_box_area=1,
            max_missing_seconds=0.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tracking_replay(
                frames,
                tracker=tracker,
                output_dir=Path(tmp) / "reacq",
                run_id="reacq-boundary",
            )
            self.assertEqual(
                result["metrics"]["reacquisition_count"],
                0,
                "ID reuse after session reset must not count as reacquisition",
            )
            self.assertGreaterEqual(result["metrics"]["session_reset_count"], 2)

    def test_filtered_detections_populate_rejection_distribution(self):
        frames = [
            ReplayFrame(
                0,
                [
                    {"bbox": [10, 10, 60, 120], "confidence": 0.9},
                    {"bbox": [200, 200, 250, 300], "confidence": 0.01},  # low_confidence filter
                    {"bbox": [5, 5, 6, 6], "confidence": 0.9},  # tiny_box filter
                ],
                now=1.0,
                source_id="video_a",
            ),
        ]
        tracker = SimpleTrackAssigner(match_thresh=0.2, track_buffer=5, min_box_area=50, track_thresh=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tracking_replay(
                frames,
                tracker=tracker,
                output_dir=Path(tmp) / "rej",
                run_id="rej-filter",
            )
            metrics = result["metrics"]
            self.assertGreaterEqual(metrics["runtime_metrics"]["match_rejected_total"], 2)
            dist = metrics["rejection_reason_distribution"]
            self.assertIn("low_confidence", dist)
            self.assertIn("tiny_box", dist)


if __name__ == "__main__":
    unittest.main()
