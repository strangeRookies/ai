import unittest

from ai.visualization.action_overlay import bbox_visual_state, format_bbox_label
from ai.visualization.draw import bbox_thickness, label_scale


class OverlayVisualRulesTest(unittest.TestCase):
    def test_normal_label_is_compact_track_id(self):
        box = {"track_id": 3, "faint_probability": 0.12, "event_triggered": False}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "normal")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "ID: 3")

    def test_warning_label_shows_faint_probability(self):
        box = {"track_id": 3, "faint_probability": 0.42, "event_triggered": False}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "warning")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "WARN | ID: 3 (Faint: 0.42)")

    def test_alert_label_is_explicit(self):
        box = {"track_id": 3, "faint_probability": 0.81, "event_triggered": True}

        self.assertEqual(bbox_visual_state(box, threshold=0.3), "alert")
        self.assertEqual(format_bbox_label(box, threshold=0.3), "[ALERT] ID: 3 (Faint: 0.81)")

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
