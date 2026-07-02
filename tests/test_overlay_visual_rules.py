import unittest

from ai.visualization.action_overlay import bbox_visual_state, format_bbox_label
from ai.visualization.draw import bbox_thickness, label_scale


class OverlayVisualRulesTest(unittest.TestCase):
    def test_normal_label_is_compact_track_id(self):
        box = {"track_id": 3, "display_id": 3, "faint_probability": 0.12, "event_triggered": False}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "normal")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: 3")

    def test_warning_label_shows_faint_probability(self):
        box = {"track_id": 3, "display_id": 3, "faint_probability": 0.42, "event_triggered": False}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "warning")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: 3 | FAINT 42%")

    def test_alert_label_is_explicit(self):
        box = {"track_id": 3, "display_id": 3, "faint_probability": 0.81, "event_triggered": True}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "alert")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: 3 | FAINT 81%")

    def test_missing_track_id_has_explicit_fallback_label(self):
        box = {"faint_probability": None, "event_triggered": False}

        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: n/a")

    def test_label_uses_raw_track_id_not_display_id(self):
        box = {"track_id": 73, "display_id": 2, "faint_probability": 0.10, "event_triggered": False}

        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: 73")

    def test_debug_label_includes_track_diagnostics(self):
        box = {
            "track_id": 3,
            "display_id": 3,
            "faint_probability": 0.12,
            "event_triggered": False,
            "overlay_debug_tracks": True,
            "track_age": 12,
            "missing_frames": 1,
            "track_confidence": 0.91,
        }

        self.assertEqual(format_bbox_label(box, threshold=0.3), "track ID: 3 | age 12 | miss 1 | conf 0.91")

    def test_debug_label_shows_raw_when_display_differs(self):
        """When display_id != track_id, display ID stays diagnostic only."""
        box = {
            "track_id": 73,
            "display_id": 2,
            "faint_probability": 0.10,
            "event_triggered": False,
            "overlay_debug_tracks": True,
            "track_age": 5,
            "missing_frames": 0,
            "track_confidence": 0.80,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertIn("track ID: 73", label)
        self.assertIn("display 2", label)

    def test_bbox_thickness_increases_with_state(self):
        normal = bbox_thickness(1280, "normal")
        warning = bbox_thickness(1280, "warning")
        alert = bbox_thickness(1280, "alert")

        self.assertLess(normal, warning)
        self.assertLess(warning, alert)

    def test_label_scale_grows_with_frame_width(self):
        self.assertLess(label_scale(640), label_scale(1280))


if __name__ == "__main__":
    unittest.main()
