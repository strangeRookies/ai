"""Unit tests for offline tracker A/B policies (production SimpleTrackAssigner path)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracking.simple_tracker import SimpleTrackAssigner
from ai.action.faint_post_processing import ExitEventPostProcessor


class TrackingAbPolicyTest(unittest.TestCase):
    def test_new_track_thresh_gates_new_ids_only(self):
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.30,
            match_thresh=0.20,
            track_buffer=90,
            max_missing_seconds=4.0,
            min_box_area=1.0,
        )
        # Low conf cannot mint a new track.
        out = tracker.update([{"bbox": [10, 10, 40, 80], "confidence": 0.20}], now=1.0)
        self.assertEqual(out, [])
        self.assertEqual(tracker._next_track_id, 1)
        # High conf can mint.
        out = tracker.update([{"bbox": [10, 10, 40, 80], "confidence": 0.55}], now=1.1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 1)
        # Low conf can still associate to existing track.
        out = tracker.update([{"bbox": [12, 12, 42, 82], "confidence": 0.18}], now=1.2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 1)

    def test_ultra_soft_refuses_multi_candidate_recovery(self):
        """With two active tracks, a mid detection must NOT ultra-soft-steal either ID.

        Construction: both tracks exist; hard/soft gates fail; both are nearby enough that
        unrestricted nearest-neighbor would reclaim track 1. Multi-candidate refusal must mint 3.
        """
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.95,  # force hard IoU fail
            center_match_ratio=0.05,  # force hard center fail
            soft_iou_scale=0.01,
            soft_center_scale=1.0,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
        )
        # Two well-separated people.
        tracker.update(
            [
                {"bbox": [0, 0, 40, 80], "confidence": 0.9},
                {"bbox": [200, 0, 240, 80], "confidence": 0.9},
            ],
            now=1.0,
        )
        self.assertEqual(set(tracker._tracks.keys()), {1, 2})

        # Midway detection: closer to track 1 so naive recovery would pick 1.
        mid = {"bbox": [70, 0, 110, 80], "confidence": 0.9}
        # Confirm ultra-soft alone would recover track 1 if multi-candidate gate were off.
        meta = {
            "mode": None,
            "rejectedCandidates": [
                {"trackId": 1, "iou": 0.02, "centerRatio": 1.1},
                {"trackId": 2, "iou": 0.01, "centerRatio": 1.4},
            ],
        }
        recovered = tracker._ultra_soft_recover(meta, assigned_track_ids=set())
        self.assertIsNone(recovered, "multi-candidate ultra-soft must refuse recovery")

        out = tracker.update([mid], now=1.1)
        self.assertEqual(len(out), 1)
        # Must allocate a brand-new id (3), not steal 1 or 2.
        self.assertEqual(out[0]["track_id"], 3)
        events = tracker.diagnostics().get("lifecycle_events") or []
        new_ev = [e for e in events if e.get("event") == "new_track"]
        self.assertTrue(new_ev)
        # When multi-det/extra path not used, switch reason should not invent IoU fail without candidates.
        self.assertIn(new_ev[0].get("switchReason"), {"MULTI_DET_EXTRA", "IOU_BELOW_THRESHOLD", "NO_CANDIDATE", "NEW_SCENE"})

    def test_exit_edge_trigger_outside_only_zero_and_inside_to_outside_one(self):
        proc = ExitEventPostProcessor(min_consecutive=1, cooldown_seconds=0.0)
        # Never entered → outside forever → no fire
        self.assertFalse(proc.observe("cam", 1, inside_safe_zone=False, timestamp=1.0))
        self.assertFalse(proc.observe("cam", 1, inside_safe_zone=False, timestamp=2.0))
        # Enter then leave → one fire
        self.assertFalse(proc.observe("cam", 2, inside_safe_zone=True, timestamp=3.0))
        self.assertTrue(proc.observe("cam", 2, inside_safe_zone=False, timestamp=4.0))
        # Stay outside → no second fire
        self.assertFalse(proc.observe("cam", 2, inside_safe_zone=False, timestamp=5.0))

    def test_multi_det_extra_not_labeled_iou_below_threshold(self):
        """Second same-frame detection after both tracks claimed is MULTI_DET_EXTRA, not IoU fail."""
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.2,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
        )
        # Seed two tracks.
        tracker.update(
            [
                {"bbox": [0, 0, 50, 100], "confidence": 0.9},
                {"bbox": [200, 0, 250, 100], "confidence": 0.9},
            ],
            now=1.0,
        )
        # Same two plus a third mid detection: third must be MULTI_DET_EXTRA.
        out = tracker.update(
            [
                {"bbox": [2, 2, 52, 102], "confidence": 0.9},
                {"bbox": [202, 2, 252, 102], "confidence": 0.9},
                {"bbox": [100, 0, 150, 100], "confidence": 0.9},
            ],
            now=1.1,
        )
        self.assertEqual(len(out), 3)
        events = tracker.diagnostics().get("lifecycle_events") or []
        new_ev = [e for e in events if e.get("event") == "new_track"]
        self.assertTrue(new_ev)
        self.assertEqual(new_ev[0].get("switchReason"), "MULTI_DET_EXTRA")
        self.assertIsNone(new_ev[0].get("previousTrackId"))
        self.assertTrue(new_ev[0].get("alreadyAssignedTracks"))

    def test_iou_below_threshold_attaches_previous_candidate(self):
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.95,
            center_match_ratio=0.05,
            soft_iou_scale=0.01,
            soft_center_scale=1.0,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
        )
        tracker.update([{"bbox": [0, 0, 40, 80], "confidence": 0.9}], now=1.0)
        # Far enough that hard/soft/sole/ultra-soft all fail (center ratio >> 1.8).
        out = tracker.update([{"bbox": [400, 0, 440, 80], "confidence": 0.9}], now=1.1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 2)
        events = tracker.diagnostics().get("lifecycle_events") or []
        new_ev = [e for e in events if e.get("event") == "new_track"]
        self.assertTrue(new_ev)
        self.assertEqual(new_ev[0].get("switchReason"), "IOU_BELOW_THRESHOLD")
        self.assertEqual(new_ev[0].get("previousTrackId"), 1)
        snap = new_ev[0].get("previousTrackSnapshot") or {}
        self.assertIsNotNone(snap.get("bbox"))
        rejected = (new_ev[0].get("bestRejected") or {}).get("rejectedCandidates") or []
        self.assertGreaterEqual(len(rejected), 1)
        self.assertIsNotNone(rejected[0].get("iou"))

    def test_cache_replay_deterministic_summaries(self):
        # Minimal synthetic cache
        from scripts.replay_tracking_from_cache import TrackerConfig, run_tracker_config

        frames = []
        for i in range(30):
            frames.append(
                {
                    "frame_id": i,
                    "timestamp_ms": int(i * (1000 / 30)),
                    "source_width": 1280,
                    "source_height": 720,
                    "detections": [
                        {
                            "detection_index": 0,
                            "bbox_xyxy": [100 + i * 0.5, 100, 140 + i * 0.5, 220],
                            "confidence": 0.6,
                            "class_id": 0,
                            "keypoints": [],
                            "avg_keypoint_conf": 0.5,
                            "valid_keypoints": 0,
                        }
                    ],
                }
            )
        cfg = TrackerConfig(name="det_test", new_track_thresh=0.25, match_thresh=0.2)
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "r1"
            p2 = Path(td) / "r2"
            s1 = run_tracker_config(frames, cfg, p1, source_fps=30.0)
            s2 = run_tracker_config(frames, cfg, p2, source_fps=30.0)
            for k in (
                "unexpected_new_tracks",
                "lost_tracks",
                "iou_below_threshold",
                "dominant_track_id",
                "person_present_frames",
            ):
                self.assertEqual(s1[k], s2[k], k)
            f1 = (p1 / "frame_results.jsonl").read_text(encoding="utf-8")
            f2 = (p2 / "frame_results.jsonl").read_text(encoding="utf-8")
            self.assertEqual(f1, f2)

    def test_claimed_iou_suppresses_near_duplicate_mint(self):
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.20,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
            near_dup_suppress_mode="claimed_iou",
            near_dup_iou_thresh=0.70,
            sort_detections_by_conf=True,
        )
        # Seed one person.
        out = tracker.update([{"bbox": [100, 100, 160, 260], "confidence": 0.8}], now=1.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 1)
        # Same frame-style near-dup second box: high IoU with claimed track, must NOT mint.
        out = tracker.update(
            [
                {"bbox": [101, 101, 161, 261], "confidence": 0.75},
                {"bbox": [103, 104, 162, 262], "confidence": 0.55},
            ],
            now=1.1,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_id"], 1)
        events = tracker.diagnostics().get("lifecycle_events") or []
        new_ev = [e for e in events if e.get("event") == "new_track"]
        self.assertEqual(new_ev, [])
        suppress = [e for e in events if e.get("event") == "filter" and e.get("reason") == "near_duplicate_suppress"]
        self.assertGreaterEqual(len(suppress), 1)

    def test_hybrid_suppress_refuses_two_claimed_people(self):
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.20,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
            near_dup_suppress_mode="hybrid",
            near_dup_iou_thresh=0.70,
            near_dup_center_ratio=0.25,
            sort_detections_by_conf=True,
        )
        # Two well-separated people claim tracks.
        tracker.update(
            [
                {"bbox": [0, 0, 50, 100], "confidence": 0.9},
                {"bbox": [300, 0, 350, 100], "confidence": 0.9},
            ],
            now=1.0,
        )
        # Midway third person (not near either) must mint, not suppress wrongly.
        out = tracker.update(
            [
                {"bbox": [0, 0, 50, 100], "confidence": 0.9},
                {"bbox": [300, 0, 350, 100], "confidence": 0.9},
                {"bbox": [140, 0, 190, 100], "confidence": 0.85},
            ],
            now=1.1,
        )
        ids = sorted(int(x["track_id"]) for x in out)
        self.assertEqual(ids, [1, 2, 3])

    def test_iou_only_does_not_suppress_distant_second_person(self):
        tracker = SimpleTrackAssigner(
            track_thresh=0.10,
            new_track_thresh=0.25,
            match_thresh=0.20,
            track_buffer=90,
            max_missing_seconds=6.0,
            min_box_area=1.0,
            near_dup_suppress_mode="claimed_iou",
            near_dup_iou_thresh=0.70,
            sort_detections_by_conf=True,
        )
        tracker.update([{"bbox": [0, 0, 50, 100], "confidence": 0.9}], now=1.0)
        out = tracker.update(
            [
                {"bbox": [0, 0, 50, 100], "confidence": 0.9},
                {"bbox": [400, 0, 450, 100], "confidence": 0.85},
            ],
            now=1.1,
        )
        ids = sorted(int(x["track_id"]) for x in out)
        self.assertEqual(ids, [1, 2])


if __name__ == "__main__":
    unittest.main()
