import unittest

from tracking.simple_tracker import SimpleTrackAssigner, bbox_iou, center_distance_ratio, predicted_bbox, smooth_bbox


class SimpleTrackerTest(unittest.TestCase):
    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_assigns_stable_track_id_for_overlapping_boxes(self):
        tracker = SimpleTrackAssigner(iou_threshold=0.2)

        first = tracker.update([{"bbox": [0, 0, 100, 100]}], now=1.0)
        second = tracker.update([{"bbox": [5, 5, 105, 105]}], now=1.1)

        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

    def test_preserves_detector_track_id(self):
        tracker = SimpleTrackAssigner()
        detections = tracker.update([{"track_id": 42, "bbox": [0, 0, 100, 100]}], now=1.0)
        self.assertEqual(detections[0]["track_id"], 42)

    def test_smooth_motion_keeps_same_id(self):
        tracker = SimpleTrackAssigner(match_thresh=0.2, track_buffer=10, min_box_area=1)

        ids = []
        for frame_idx in range(5):
            detection = {"bbox": [frame_idx * 3, 0, 100 + frame_idx * 3, 100], "confidence": 0.9}
            ids.append(tracker.update([detection], now=float(frame_idx))[0]["track_id"])

        self.assertEqual(len(set(ids)), 1)

    def test_short_missing_gap_keeps_track_id(self):
        tracker = SimpleTrackAssigner(match_thresh=0.2, track_buffer=3, min_box_area=1, max_missing_seconds=10)

        first = tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.9}], now=1.0)[0]
        self.assertEqual(tracker.update([], now=2.0), [])
        self.assertEqual(tracker.update([], now=3.0), [])
        after_gap = tracker.update([{"bbox": [4, 0, 104, 100], "confidence": 0.9}], now=4.0)[0]

        self.assertEqual(first["track_id"], after_gap["track_id"])
        self.assertGreaterEqual(tracker.diagnostics()["active_tracks"], 1)

    def test_center_match_keeps_id_when_pose_bbox_jumps(self):
        tracker = SimpleTrackAssigner(match_thresh=0.5, center_match_ratio=0.55, min_box_area=1)

        first = tracker.update([{"bbox": [100, 100, 180, 260], "confidence": 0.9}], now=1.0)[0]
        jumped = tracker.update([{"bbox": [130, 100, 210, 260], "confidence": 0.9}], now=1.1)[0]

        self.assertEqual(first["track_id"], jumped["track_id"])
        self.assertLess(bbox_iou(first["raw_bbox"], jumped["raw_bbox"]), 0.5)

    def test_predicted_bbox_bridges_short_motion_gap(self):
        tracker = SimpleTrackAssigner(match_thresh=0.5, center_match_ratio=0.40, track_buffer=5, min_box_area=1, max_missing_seconds=10)

        first = tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.9}], now=1.0)[0]
        tracker.update([{"bbox": [10, 0, 110, 100], "confidence": 0.9}], now=1.1)
        tracker.update([], now=1.2)
        after_gap = tracker.update([{"bbox": [30, 0, 130, 100], "confidence": 0.9}], now=1.3)[0]

        self.assertEqual(first["track_id"], after_gap["track_id"])
        self.assertIn("predicted_bbox", tracker.diagnostics()["tracks"][str(first["track_id"])])

    def test_bbox_smoothing_output(self):
        smoothed = smooth_bbox([0, 0, 100, 100], [10, 0, 110, 100], alpha=0.5)

        self.assertEqual(smoothed, [5.0, 0.0, 105.0, 100.0])

    def test_center_distance_ratio(self):
        ratio = center_distance_ratio([0, 0, 100, 100], [10, 0, 110, 100])

        self.assertLess(ratio, 0.1)

    def test_predicted_bbox_uses_velocity_and_missing_frames(self):
        track = {"smoothed_bbox": [10, 0, 110, 100], "velocity": [10, 0, 10, 0], "missing_frames": 2}

        self.assertEqual(predicted_bbox(track), [30.0, 0.0, 130.0, 100.0])

    def test_fall_aspect_change_keeps_same_track_id(self):
        """Standing tall bbox → lying wide bbox should not allocate a new ID."""
        tracker = SimpleTrackAssigner(
            match_thresh=0.25,
            center_match_ratio=0.85,
            max_missing_seconds=6.0,
            track_buffer=30,
            min_box_area=1,
        )
        standing = tracker.update(
            [{"bbox": [100, 40, 160, 220], "confidence": 0.9}],
            now=1.0,
        )[0]
        # Wide lying box, center still near person
        lying = tracker.update(
            [{"bbox": [70, 140, 220, 200], "confidence": 0.85}],
            now=1.1,
        )[0]
        self.assertEqual(standing["track_id"], lying["track_id"])
        self.assertLess(
            bbox_iou(standing["raw_bbox"], lying["raw_bbox"]),
            0.35,
            msg="fixture should simulate low-IoU fall jump",
        )

    def test_filters_low_confidence_and_tiny_boxes(self):
        tracker = SimpleTrackAssigner(track_thresh=0.5, min_box_area=100)

        low_conf = tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.2}], now=1.0)
        tiny = tracker.update([{"bbox": [0, 0, 5, 5], "confidence": 0.9}], now=2.0)

        self.assertEqual(low_conf, [])
        self.assertEqual(tiny, [])

    def test_lifecycle_events_record_filter_and_new_track(self):
        tracker = SimpleTrackAssigner(track_thresh=0.5, min_box_area=100)

        tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.2}], now=1.0)
        filter_events = [e for e in tracker.diagnostics()["lifecycle_events"] if e.get("event") == "filter"]
        self.assertTrue(any(e.get("reason") == "low_confidence" for e in filter_events))

        tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.9}], now=2.0)
        new_events = [e for e in tracker.diagnostics()["lifecycle_events"] if e.get("event") == "new_track"]
        self.assertEqual(len(new_events), 1)
        self.assertEqual(new_events[0]["reason"], "no_match")
        self.assertEqual(tracker.diagnostics()["new_tracks"], 1)

    def test_lifecycle_events_record_lost_by_max_missing_seconds(self):
        tracker = SimpleTrackAssigner(
            match_thresh=0.2,
            track_buffer=100,
            max_missing_seconds=1.0,
            min_box_area=1,
        )
        first = tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.9}], now=1.0)[0]
        tracker.update([], now=3.0)
        lost_events = [e for e in tracker.diagnostics()["lifecycle_events"] if e.get("event") == "lost"]
        self.assertEqual(len(lost_events), 1)
        self.assertEqual(lost_events[0]["reason"], "max_missing_seconds")
        self.assertEqual(lost_events[0]["trackId"], first["track_id"])
        self.assertEqual(tracker.diagnostics()["lost_tracks"], 1)
        self.assertEqual(tracker.diagnostics()["removed_track_ids"], [first["track_id"]])

    def test_lifecycle_events_record_soft_match_on_fall_aspect_change(self):
        tracker = SimpleTrackAssigner(
            match_thresh=0.25,
            center_match_ratio=0.85,
            max_missing_seconds=6.0,
            track_buffer=30,
            min_box_area=1,
        )
        tracker.update([{"bbox": [100, 40, 160, 220], "confidence": 0.9}], now=1.0)
        tracker.update([{"bbox": [70, 140, 220, 200], "confidence": 0.85}], now=1.1)
        match_events = [e for e in tracker.diagnostics()["lifecycle_events"] if e.get("event") == "match"]
        self.assertTrue(any(e.get("reason") in {"soft", "hard", "sole"} for e in match_events))


if __name__ == "__main__":
    unittest.main()
