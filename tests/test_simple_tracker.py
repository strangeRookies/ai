import unittest

from tracking.simple_tracker import SimpleTrackAssigner, bbox_iou, smooth_bbox


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

    def test_bbox_smoothing_output(self):
        smoothed = smooth_bbox([0, 0, 100, 100], [10, 0, 110, 100], alpha=0.5)

        self.assertEqual(smoothed, [5.0, 0.0, 105.0, 100.0])

    def test_filters_low_confidence_and_tiny_boxes(self):
        tracker = SimpleTrackAssigner(track_thresh=0.5, min_box_area=100)

        low_conf = tracker.update([{"bbox": [0, 0, 100, 100], "confidence": 0.2}], now=1.0)
        tiny = tracker.update([{"bbox": [0, 0, 5, 5], "confidence": 0.9}], now=2.0)

        self.assertEqual(low_conf, [])
        self.assertEqual(tiny, [])


if __name__ == "__main__":
    unittest.main()
