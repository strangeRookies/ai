import unittest

from tracking.simple_tracker import SimpleTrackAssigner, bbox_iou


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


if __name__ == "__main__":
    unittest.main()
