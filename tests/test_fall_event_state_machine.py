"""Fall/Faint lifecycle: NEW_FALL once, then FAINT_SUSPECTED / FALL_UNRECOVERED while lying."""

from __future__ import annotations

import unittest

from ai.action.fall_event_state import (
    EVENT_TYPE_FAINT_SUSPECTED,
    EVENT_TYPE_FALL_UNRECOVERED,
    FallEventStateMachine,
    FallState,
    LifecycleKind,
)
from ai.action.faint_post_processing import FaintEventPostProcessor


class FallEventStateMachineTest(unittest.TestCase):
    def test_confirm_once_then_suppress_new_fall_but_emit_unrecovered_after_delay(self):
        sm = FallEventStateMachine(
            min_consecutive_faint=2,
            recover_consecutive=2,
            unrecovered_after_seconds=10.0,
            unrecovered_repeat_seconds=30.0,
        )

        sm.update("cam_01", 1.0, track_id=7, is_alert=True, prediction={"label": "Faint"})
        d2 = sm.update("cam_01", 2.0, track_id=7, is_alert=True, prediction={"label": "Faint"})
        self.assertEqual(d2.kind, LifecycleKind.NEW_FALL)
        original_id = d2.event_id

        # Within 10s: still suppress NEW_FALL, no unrecovered yet
        d_mid = sm.update("cam_01", 8.0, track_id=7, is_alert=True, prediction={"label": "Faint"})
        self.assertEqual(d_mid.kind, LifecycleKind.SUPPRESS_NEW_FALL)

        # After 10s still lying/alert → unrecovered, not new fall
        d_u = sm.update(
            "cam_01",
            12.5,
            track_id=7,
            is_alert=True,
            prediction={"label": "Faint"},
            movement_level="still",
            posture_label="lying_like",
            lying_like=True,
        )
        self.assertEqual(d_u.kind, LifecycleKind.UNRECOVERED)
        self.assertEqual(d_u.event_type, EVENT_TYPE_FAINT_SUSPECTED)
        self.assertEqual(d_u.original_event_id, original_id)
        self.assertGreaterEqual(d_u.duration_sec or 0, 10.0)
        self.assertEqual(d_u.event_id, original_id)

        # Before repeat window: no second unrecovered
        d_wait = sm.update("cam_01", 20.0, track_id=7, is_alert=True, prediction={"label": "Faint"})
        self.assertEqual(d_wait.kind, LifecycleKind.SUPPRESS_NEW_FALL)


    def test_confirm_freezes_faint_prob_and_reuses_event_id_on_unrecovered(self):
        sm = FallEventStateMachine(
            min_consecutive_faint=2,
            recover_consecutive=2,
            unrecovered_after_seconds=10.0,
            unrecovered_repeat_seconds=30.0,
        )
        pred_high = {"label": "Faint", "score": 0.91, "probabilities": {"Faint": 0.91}}
        sm.update("cam_01", 1.0, track_id=9, is_alert=True, prediction=pred_high)
        d_new = sm.update("cam_01", 2.0, track_id=9, is_alert=True, prediction=pred_high)
        self.assertEqual(d_new.kind, LifecycleKind.NEW_FALL)
        self.assertAlmostEqual(d_new.faint_prob or 0.0, 0.91)
        self.assertEqual(d_new.consecutive_count, 2)
        original = d_new.event_id

        pred_low = {"label": "Faint", "score": 0.33, "probabilities": {"Faint": 0.33}}
        d_u = sm.update(
            "cam_01",
            13.0,
            track_id=9,
            is_alert=True,
            prediction=pred_low,
            movement_level="still",
            posture_label="lying_like",
            lying_like=True,
        )
        self.assertEqual(d_u.kind, LifecycleKind.UNRECOVERED)
        self.assertEqual(d_u.event_id, original)
        self.assertEqual(d_u.original_event_id, original)
        # Unrecovered must keep confirm-time probability, not current 0.33 overlay frame.
        self.assertAlmostEqual(d_u.faint_prob or 0.0, 0.91)

    def test_fall_label_maps_to_fall_unrecovered(self):
        sm = FallEventStateMachine(min_consecutive_faint=1, unrecovered_after_seconds=5.0)
        sm.update("cam_01", 1.0, track_id=1, is_alert=True, prediction={"label": "Fall"})
        d = sm.update(
            "cam_01",
            7.0,
            track_id=1,
            is_alert=True,
            prediction={"label": "Fall"},
            movement_level="high",
            lying_like=True,
        )
        self.assertEqual(d.kind, LifecycleKind.UNRECOVERED)
        self.assertEqual(d.event_type, EVENT_TYPE_FALL_UNRECOVERED)

    def test_recover_then_new_fall_allowed(self):
        sm = FallEventStateMachine(min_consecutive_faint=2, recover_consecutive=2, unrecovered_after_seconds=10.0)

        sm.update("cam_01", 1.0, track_id=1, is_alert=True)
        d_confirm = sm.update("cam_01", 2.0, track_id=1, is_alert=True)
        self.assertEqual(d_confirm.kind, LifecycleKind.NEW_FALL)

        d_r1 = sm.update("cam_01", 3.0, track_id=1, is_alert=False)
        self.assertEqual(d_r1.kind, LifecycleKind.SUPPRESS_NEW_FALL)
        d_r2 = sm.update("cam_01", 4.0, track_id=1, is_alert=False)
        self.assertEqual(d_r2.state, FallState.RECOVERED)

        sm.update("cam_01", 4.5, track_id=1, is_alert=False)
        sm.update("cam_01", 5.0, track_id=1, is_alert=True)
        d_new = sm.update("cam_01", 6.0, track_id=1, is_alert=True)
        self.assertEqual(d_new.kind, LifecycleKind.NEW_FALL)
        self.assertNotEqual(d_new.event_id, d_confirm.event_id)

    def test_reset_all_clears_post_fall_state(self):
        sm = FallEventStateMachine(min_consecutive_faint=1, unrecovered_after_seconds=10.0)
        sm.update("cam_01", 1.0, track_id=3, is_alert=True)
        self.assertEqual(sm.get_state("cam_01", 3), FallState.POST_FALL_LYING)
        sm.reset_all()
        self.assertEqual(sm.get_state("cam_01", 3), FallState.NORMAL)
        d = sm.update("cam_01", 2.0, track_id=3, is_alert=True)
        self.assertEqual(d.kind, LifecycleKind.NEW_FALL)

    def test_tracks_are_independent(self):
        sm = FallEventStateMachine(min_consecutive_faint=2, unrecovered_after_seconds=10.0)
        sm.update("cam_01", 1.0, track_id=1, is_alert=True)
        sm.update("cam_01", 2.0, track_id=1, is_alert=True)
        sm.update("cam_01", 2.0, track_id=2, is_alert=True)
        d2b = sm.update("cam_01", 3.0, track_id=2, is_alert=True)
        self.assertEqual(d2b.kind, LifecycleKind.NEW_FALL)
        d1 = sm.update("cam_01", 4.0, track_id=1, is_alert=True)
        self.assertEqual(d1.kind, LifecycleKind.SUPPRESS_NEW_FALL)

    def test_require_upright_to_lying_blocks_confirm(self):
        sm = FallEventStateMachine(min_consecutive_faint=2, require_upright_to_lying=True)
        # Lying-only posture history, no upright → block
        sm.update("cam_01", 1.0, track_id=1, is_alert=True, posture_label="lying_like")
        d = sm.update("cam_01", 2.0, track_id=1, is_alert=True, posture_label="lying_like")
        self.assertEqual(d.kind, LifecycleKind.NONE)
        self.assertIn("upright_to_lying", d.reason or "")
        # Transition flag set → confirm
        d2 = sm.update(
            "cam_01",
            3.0,
            track_id=1,
            is_alert=True,
            posture_label="lying_like",
            upright_to_lying=True,
        )
        self.assertEqual(d2.kind, LifecycleKind.NEW_FALL)


class FaintPostProcessorLifecycleIntegrationTest(unittest.TestCase):
    def test_evaluate_emits_unrecovered_not_new_fall_after_cooldown_window(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            use_fall_state_machine=True,
            require_upright_to_lying=False,
            recover_consecutive=1,
            unrecovered_after_seconds=5.0,
            unrecovered_repeat_seconds=30.0,
        )
        # First NEW_FALL
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0, track_id=7))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0, track_id=7))

        # should_trigger stays False (no second NEW_FALL)
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 4.0, track_id=7))

        # After unrecovered_after: evaluate emits unrecovered; should_trigger still False
        d = processor.evaluate(
            "cam_01",
            {"label": "Faint"},
            8.0,
            track_id=7,
            posture_label="lying_like",
        )
        self.assertTrue(d.emit)
        self.assertTrue(d.is_unrecovered)
        self.assertIn(d.event_type, {EVENT_TYPE_FAINT_SUSPECTED, EVENT_TYPE_FALL_UNRECOVERED})
        self.assertIsNotNone(d.original_event_id)
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 9.0, track_id=7))

    def test_reset_allows_new_alert_after_video_boundary(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            use_fall_state_machine=True,
            require_upright_to_lying=False,
        )
        processor.should_trigger("cam_01", {"label": "Faint"}, 1.0, track_id=1)
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0, track_id=1))
        processor.reset()
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 21.0, track_id=1))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 22.0, track_id=1))

    def test_start_lying_without_upright_transition_does_not_confirm(self):
        """Phase B: track that is only ever lying does not emit NEW_FALL."""
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            use_fall_state_machine=True,
            require_upright_to_lying=True,
            use_posture_estimator=True,
        )
        lying = {
            "track_id": 3,
            "bbox": [10, 80, 150, 120],
            "pose_horizontal": True,
            "keypoints": [
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 5,
                {"x": 20, "y": 90, "confidence": 0.9},
                {"x": 100, "y": 92, "confidence": 0.9},
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 4,
                {"x": 25, "y": 100, "confidence": 0.9},
                {"x": 95, "y": 102, "confidence": 0.9},
            ],
        }
        # Many Faint frames while only lying — no NEW_FALL
        for t in range(1, 8):
            self.assertFalse(
                processor.should_trigger(
                    "cam_01",
                    {"label": "Faint"},
                    float(t),
                    track_id=3,
                    detection=lying,
                ),
                msg=f"t={t}",
            )

    def test_upright_then_lying_with_faint_confirms(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            use_fall_state_machine=True,
            require_upright_to_lying=True,
            use_posture_estimator=True,
        )
        upright = {
            "track_id": 5,
            "bbox": [40, 20, 80, 160],
            "pose_horizontal": False,
            "keypoints": [
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 5,
                {"x": 50, "y": 40, "confidence": 0.9},
                {"x": 70, "y": 40, "confidence": 0.9},
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 4,
                {"x": 52, "y": 120, "confidence": 0.9},
                {"x": 68, "y": 120, "confidence": 0.9},
            ],
        }
        lying = {
            "track_id": 5,
            "bbox": [20, 80, 160, 120],
            "pose_horizontal": True,
            "keypoints": [
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 5,
                {"x": 40, "y": 90, "confidence": 0.9},
                {"x": 120, "y": 92, "confidence": 0.9},
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 4,
                {"x": 42, "y": 100, "confidence": 0.9},
                {"x": 118, "y": 102, "confidence": 0.9},
            ],
        }
        # Build upright history without alert
        processor.evaluate("cam_01", {"label": "Normal"}, 1.0, track_id=5, detection=upright)
        processor.evaluate("cam_01", {"label": "Normal"}, 2.0, track_id=5, detection=upright)
        # Fall transition + faint
        self.assertFalse(
            processor.should_trigger("cam_01", {"label": "Faint"}, 3.0, track_id=5, detection=lying)
        )
        self.assertTrue(
            processor.should_trigger("cam_01", {"label": "Faint"}, 4.0, track_id=5, detection=lying)
        )

    def test_legacy_mode_without_state_machine_keeps_old_cooldown_behavior(self):
        processor = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            use_fall_state_machine=False,
        )
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 1.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 2.0))
        self.assertFalse(processor.should_trigger("cam_01", {"label": "Faint"}, 6.0))
        self.assertTrue(processor.should_trigger("cam_01", {"label": "Faint"}, 8.0))


if __name__ == "__main__":
    unittest.main()
