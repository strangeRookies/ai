import unittest

from ai.inference.track_selection import deduplicate_tracked_detections, filter_selected_track


class TrackSelectionTest(unittest.TestCase):
    def test_filter_selected_track_keeps_only_matching_track(self):
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
            {"bbox": [30, 30, 60, 60], "track_id": 2, "confidence": 0.8},
            {"bbox": [70, 70, 90, 90], "confidence": 0.7},
        ]

        selected, skipped = filter_selected_track(detections, selected_track_id=2)

        self.assertEqual([item["track_id"] for item in selected], [2])
        self.assertEqual(skipped, ["track_id=1 reason=not_selected_track", "track_id=None reason=not_selected_track"])

    def test_filter_selected_track_keeps_all_tracks_when_unset(self):
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
            {"bbox": [30, 30, 60, 60], "track_id": 2, "confidence": 0.8},
        ]

        selected, skipped = filter_selected_track(detections, selected_track_id=None)

        self.assertEqual(selected, detections)
        self.assertEqual(skipped, [])

    def test_deduplicate_tracked_detections_prefers_keypoints_then_confidence(self):
        detections = [
            {
                "bbox": [0, 0, 100, 100],
                "track_id": 7,
                "confidence": 0.95,
                "keypoints": [{"x": 1, "y": 1, "confidence": 0.9}],
            },
            {
                "bbox": [2, 2, 102, 102],
                "track_id": 7,
                "confidence": 0.70,
                "keypoints": [
                    {"x": 1, "y": 1, "confidence": 0.9},
                    {"x": 2, "y": 2, "confidence": 0.8},
                ],
            },
            {"bbox": [200, 200, 260, 260], "track_id": 8, "confidence": 0.50},
        ]

        unique, removed = deduplicate_tracked_detections(detections)

        self.assertEqual([item["track_id"] for item in unique], [7, 8])
        self.assertEqual(unique[0]["confidence"], 0.70)
        self.assertEqual(removed, ["track_id=7 reason=duplicate_bbox"])

    def test_deduplicate_keeps_distinct_track_ids_even_when_boxes_overlap(self):
        detections = [
            {"bbox": [0, 0, 100, 100], "track_id": 1, "confidence": 0.90},
            {"bbox": [2, 2, 102, 102], "track_id": 2, "confidence": 0.80},
        ]

        unique, removed = deduplicate_tracked_detections(detections)

        self.assertEqual([item["track_id"] for item in unique], [1, 2])
        self.assertEqual(removed, [])

    def test_deduplicate_prefers_tracked_box_over_untracked_overlap(self):
        detections = [
            {"bbox": [0, 0, 100, 100], "confidence": 0.99},
            {"bbox": [2, 2, 102, 102], "track_id": 4, "confidence": 0.70},
        ]

        unique, removed = deduplicate_tracked_detections(detections)

        self.assertEqual([item.get("track_id") for item in unique], [4])
        self.assertEqual(removed, ["track_id=None reason=duplicate_bbox"])


    def test_track_selector_strict_mode_keeps_only_matching(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=2, selected_track_mode="strict")
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
            {"bbox": [30, 30, 60, 60], "track_id": 2, "confidence": 0.8},
        ]
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual([item["track_id"] for item in selected], [2])
        self.assertIn("track_id=1 reason=not_selected_track", skipped)
        self.assertFalse(diag["fallback_active"])

    def test_track_selector_strict_mode_clears_on_missing(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=2, selected_track_mode="strict")
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
        ]
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual(selected, [])
        self.assertIn("track_id=1 reason=selected_track_missing", skipped)
        self.assertEqual(diag["skipped_reason"], "selected_track_missing")
        self.assertFalse(diag["fallback_active"])

    def test_track_selector_fallback_mode_until_threshold(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=2, selected_track_mode="fallback", missing_frames_threshold=3)
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
        ]
        # Frame 1: missing count = 1
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual(selected, [])
        self.assertEqual(diag["missing_frames_count"], 1)
        self.assertFalse(diag["fallback_active"])
        
        # Frame 2: missing count = 2
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual(selected, [])
        self.assertEqual(diag["missing_frames_count"], 2)
        self.assertFalse(diag["fallback_active"])

    def test_track_selector_fallback_mode_kicks_in_at_threshold(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=2, selected_track_mode="fallback", missing_frames_threshold=2)
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9, "keypoints": [1, 2, 3]},
            {"bbox": [30, 30, 60, 60], "track_id": 3, "confidence": 0.5},
        ]
        # Frame 1: missing count = 1, threshold=2이므로 fallback 미발생
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual(selected, [])
        self.assertFalse(diag["fallback_active"])

        # Frame 2: missing count = 2, fallback 발생!
        # 가장 안정적인 track_id = 1 (keypoints 보유)
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual([item["track_id"] for item in selected], [1])
        self.assertTrue(diag["fallback_active"])
        self.assertEqual(diag["fallback_track_id"], 1)
        self.assertEqual(diag["skipped_reason"], "selected_track_missing_fallback")

        # Frame 3: missing count = 3, 기존 fallback track_id = 1 유지
        # 새로운 detections에 1번 트랙이 있으면 1번 선택
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual([item["track_id"] for item in selected], [1])

    def test_track_selector_fallback_mode_resets_on_selected_reappears(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=2, selected_track_mode="fallback", missing_frames_threshold=2)
        detections_missing = [{"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9}]
        detections_present = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
            {"bbox": [30, 30, 60, 60], "track_id": 2, "confidence": 0.8},
        ]

        # missing count = 1
        selector.filter(detections_missing)
        # missing count = 2 -> fallback
        selector.filter(detections_missing)
        self.assertTrue(selector.active_fallback_track_id is not None)

        # selected track reappears -> reset
        selected, skipped, diag = selector.filter(detections_present)
        self.assertEqual([item["track_id"] for item in selected], [2])
        self.assertEqual(selector.missing_frames_count, 0)
        self.assertIsNone(selector.active_fallback_track_id)
        self.assertFalse(diag["fallback_active"])

    def test_track_selector_no_selected_track_keeps_all(self):
        from ai.inference.track_selection import TrackSelector
        selector = TrackSelector(selected_track_id=None)
        detections = [
            {"bbox": [0, 0, 20, 20], "track_id": 1, "confidence": 0.9},
            {"bbox": [30, 30, 60, 60], "track_id": 2, "confidence": 0.8},
        ]
        selected, skipped, diag = selector.filter(detections)
        self.assertEqual(selected, detections)
        self.assertEqual(skipped, [])
        self.assertFalse(diag["fallback_active"])


if __name__ == "__main__":
    unittest.main()
