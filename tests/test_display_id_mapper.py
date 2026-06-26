"""Tests for DisplayIdMapper.

Covers:
- Raw IDs can be large while display IDs start from 1
- display_id remains stable across frames for the same raw track
- Removed tracks free their display IDs safely and the IDs can be reused
- Debug label formatting includes both display_id and raw ID when they differ
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.display_id_mapper import DisplayIdMapper
from ai.visualization.action_overlay import format_bbox_label


class DisplayIdMapperTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Basic allocation
    # ------------------------------------------------------------------

    def test_large_raw_ids_get_small_display_ids(self):
        mapper = DisplayIdMapper()
        mapper.update({70, 71, 75})
        # All three should map to 1, 2, 3 (sorted allocation)
        display_ids = sorted(mapper.display_id(r) for r in (70, 71, 75))
        self.assertEqual(display_ids, [1, 2, 3])

    def test_first_track_gets_display_id_1(self):
        mapper = DisplayIdMapper()
        mapper.update({99})
        self.assertEqual(mapper.display_id(99), 1)

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def test_display_id_stable_across_frames(self):
        mapper = DisplayIdMapper()
        # Frame 1
        mapper.update({42})
        id_frame1 = mapper.display_id(42)
        # Frame 2 — same track still active
        mapper.update({42})
        id_frame2 = mapper.display_id(42)
        self.assertEqual(id_frame1, id_frame2)

    def test_new_track_does_not_change_existing_mapping(self):
        mapper = DisplayIdMapper()
        mapper.update({10})
        d10 = mapper.display_id(10)
        mapper.update({10, 20})
        self.assertEqual(mapper.display_id(10), d10)
        self.assertIsNotNone(mapper.display_id(20))

    # ------------------------------------------------------------------
    # ID reuse after track loss
    # ------------------------------------------------------------------

    def test_freed_display_id_is_reused(self):
        mapper = DisplayIdMapper()
        mapper.update({5})
        freed_display_id = mapper.display_id(5)  # should be 1
        # Track 5 disappears
        mapper.update(set())
        # New track arrives — should reuse display_id 1
        mapper.update({99})
        self.assertEqual(mapper.display_id(99), freed_display_id)

    def test_smallest_freed_id_reused_first(self):
        mapper = DisplayIdMapper()
        mapper.update({1, 2, 3})  # raw IDs; display IDs 1,2,3 assigned
        d1 = mapper.display_id(1)
        d2 = mapper.display_id(2)
        # Remove raw IDs 1 and 2
        mapper.update({3})
        # Both freed; next new track should get the smaller freed display ID
        mapper.update({3, 99})
        self.assertEqual(mapper.display_id(99), min(d1, d2))

    def test_unknown_raw_id_returns_none(self):
        mapper = DisplayIdMapper()
        mapper.update({10})
        self.assertIsNone(mapper.display_id(999))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_all_mappings(self):
        mapper = DisplayIdMapper()
        mapper.update({42, 43})
        mapper.reset()
        self.assertIsNone(mapper.display_id(42))
        # After reset, first new track gets display_id 1 again
        mapper.update({999})
        self.assertEqual(mapper.display_id(999), 1)

    # ------------------------------------------------------------------
    # Diagnostics / snapshot
    # ------------------------------------------------------------------

    def test_mapping_snapshot_is_json_serialisable(self):
        import json
        mapper = DisplayIdMapper()
        mapper.update({70, 71})
        snap = mapper.mapping_snapshot()
        # Should not raise
        json.dumps(snap)
        self.assertIn("raw_to_display", snap)
        self.assertIn("display_to_raw", snap)


class DisplayIdOverlayLabelTest(unittest.TestCase):
    """Verify that format_bbox_label uses display_id in the label text."""

    def test_normal_label_uses_display_id(self):
        box = {
            "track_id": 73,
            "display_id": 2,
            "faint_probability": 0.10,
            "event_triggered": False,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertEqual(label, "ID 2")

    def test_warning_label_uses_display_id(self):
        box = {
            "track_id": 73,
            "display_id": 2,
            "faint_probability": 0.45,
            "event_triggered": False,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertEqual(label, "ID 2 | Faint 0.45")

    def test_alert_label_uses_display_id(self):
        box = {
            "track_id": 73,
            "display_id": 2,
            "faint_probability": 0.82,
            "event_triggered": True,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertEqual(label, "ALERT | ID 2 | Faint 0.82")

    def test_debug_label_shows_raw_id_when_different(self):
        box = {
            "track_id": 73,
            "display_id": 2,
            "faint_probability": 0.10,
            "event_triggered": False,
            "overlay_debug_tracks": True,
            "track_age": 5,
            "missing_frames": 0,
            "track_confidence": 0.88,
        }
        label = format_bbox_label(box, threshold=0.3)
        # label must contain "ID 2" and "raw 73"
        self.assertIn("ID 2", label)
        self.assertIn("raw 73", label)

    def test_debug_label_omits_raw_suffix_when_ids_match(self):
        """When display_id == track_id (e.g. first track), no raw suffix."""
        box = {
            "track_id": 1,
            "display_id": 1,
            "faint_probability": 0.10,
            "event_triggered": False,
            "overlay_debug_tracks": True,
            "track_age": 3,
            "missing_frames": 0,
            "track_confidence": 0.95,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertNotIn("raw", label)

    def test_fallback_to_raw_id_when_no_display_id(self):
        """If display_id is absent, raw track_id is shown."""
        box = {
            "track_id": 73,
            "faint_probability": 0.10,
            "event_triggered": False,
        }
        label = format_bbox_label(box, threshold=0.3)
        self.assertEqual(label, "ID 73")


if __name__ == "__main__":
    unittest.main()
