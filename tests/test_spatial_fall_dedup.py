"""Camera-scoped spatial dedup for repeated NEW_FALL alerts caused by tracker churn.

Covers the scenario from the 낙상 알림 스팸 investigation: SimpleTrackAssigner drops a
lying person's track_id and re-issues a new one every few seconds, so the fall
lifecycle state machine (keyed by track_id) restarts NEW_FALL confirmation from
scratch for each new id. Spatial dedup suppresses the duplicate NEW_FALL alert for a
new track_id when it lands on the same recent fall location, WITHOUT rolling back the
new track's own internal state machine (that was the bug caught in review: rolling
back via revert_confirm_to_candidate resets confirmed_ts every cycle and permanently
blocks the unrecovered/FAINT escalation for that track).
"""

from __future__ import annotations

import unittest

from ai.action.fall_event_state import LifecycleKind
from ai.action.faint_post_processing import FaintEventPostProcessor

LOCATION_A = [100, 100, 200, 220]  # x1, y1, x2, y2
LOCATION_A_JITTERED = [105, 98, 205, 218]  # same spot, small bbox noise
LOCATION_FAR = [900, 700, 980, 860]  # clearly different spot


def detection(bbox):
    return {"bbox": bbox}


def make_processor(**overrides):
    kwargs = dict(
        min_consecutive_faint=2,
        cooldown_seconds=5.0,
        unrecovered_after_seconds=10.0,
        unrecovered_repeat_seconds=30.0,
        use_posture_estimator=False,  # bbox-only; posture not under test here
        spatial_dedup_seconds=60.0,
        spatial_dedup_distance_ratio=1.0,
    )
    kwargs.update(overrides)
    return FaintEventPostProcessor(**kwargs)


class SpatialFallDedupTest(unittest.TestCase):
    def test_same_track_no_churn_reaches_unrecovered_normally(self):
        """Baseline / regression: no track_id change at all, no spatial dedup involved."""
        p = make_processor()
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm.emit)
        self.assertEqual(d_confirm.kind, "new_fall")

        d_mid = p.evaluate("cam_01", {"label": "Faint"}, 8.0, track_id="A", detection=detection(LOCATION_A))
        self.assertFalse(d_mid.emit)

        d_unrec = p.evaluate("cam_01", {"label": "Faint"}, 13.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_unrec.emit)
        self.assertEqual(d_unrec.kind, "unrecovered")

    def test_track_id_churn_same_location_suppresses_new_fall_but_unrecovered_still_fires(self):
        """The core scenario: A confirms + alerts, tracker swaps to B at the same spot.
        B's NEW_FALL must be suppressed, but B must still reach UNRECOVERED after
        unrecovered_after_seconds of continued alert — this is the bug that was caught
        in review (revert_confirm_to_candidate would have permanently blocked it)."""
        p = make_processor()
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm_a = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_a.emit)
        self.assertEqual(d_confirm_a.kind, "new_fall")

        # Tracker drops A, issues B at (roughly) the same physical spot.
        p.evaluate("cam_01", {"label": "Faint"}, 3.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        d_confirm_b = p.evaluate("cam_01", {"label": "Faint"}, 4.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_b.emit, "B's duplicate NEW_FALL at the same spot must be suppressed")
        self.assertEqual(d_confirm_b.memo_text, "new_fall_blocked_by_spatial_dedup")

        # B keeps being detected as alert/lying at the same spot with no further track
        # churn. It must still escalate to UNRECOVERED once unrecovered_after_seconds
        # has elapsed since B's OWN (internal, unpublished) confirm.
        d_mid_b = p.evaluate("cam_01", {"label": "Faint"}, 10.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_mid_b.emit)
        self.assertEqual(d_mid_b.lifecycle.kind, LifecycleKind.SUPPRESS_NEW_FALL)

        d_unrec_b = p.evaluate("cam_01", {"label": "Faint"}, 15.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        self.assertTrue(d_unrec_b.emit, "B must still escalate to an unrecovered/FAINT alert")
        self.assertEqual(d_unrec_b.kind, "unrecovered")

        # Lineage: B's escalation must point back to A's ACTUALLY-PUBLISHED NEW_FALL
        # event_id, not B's own internal (never-published) confirm id.
        self.assertEqual(
            d_unrec_b.original_event_id,
            d_confirm_a.event_id,
            "unrecovered original_event_id should chain back to the alert that was actually published",
        )

    def test_no_confirm_revert_loop_confirmed_ts_stays_stable(self):
        """Regression guard for the reviewed bug: B's internal confirmed_ts must be set
        once and never reset by repeated spatial-dedup suppression."""
        p = make_processor()
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm_a = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))

        p.evaluate("cam_01", {"label": "Faint"}, 3.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        d_confirm_b = p.evaluate("cam_01", {"label": "Faint"}, 4.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_b.emit)

        track_b_state = p._state_machine.get_track("cam_01", "B")
        self.assertIsNotNone(track_b_state)
        first_confirmed_ts = track_b_state.confirmed_ts
        self.assertIsNotNone(first_confirmed_ts, "B must have a real confirmed_ts, not wiped by revert")
        self.assertEqual(
            track_b_state.last_event_id,
            d_confirm_a.event_id,
            "lineage rewrite should point B's last_event_id at A's published event, not touch confirmed_ts",
        )

        # Several more frames of continued alert at the same spot: if confirm/revert were
        # looping, confirmed_ts would keep flipping to None and back; it must stay fixed,
        # and every one of these calls must resolve as an ordinary "already confirmed,
        # still lying" tick (SUPPRESS_NEW_FALL / UNRECOVERED) rather than NEW_FALL again.
        for t in (5.0, 6.0, 7.0, 8.0, 9.0):
            d = p.evaluate("cam_01", {"label": "Faint"}, t, track_id="B", detection=detection(LOCATION_A_JITTERED))
            self.assertNotEqual(d.memo_text, "new_fall_blocked_by_spatial_dedup", msg=f"t={t}")
            self.assertIn(d.lifecycle.kind, (LifecycleKind.SUPPRESS_NEW_FALL,), msg=f"t={t}")
            track_b_state = p._state_machine.get_track("cam_01", "B")
            self.assertEqual(track_b_state.confirmed_ts, first_confirmed_ts, msg=f"t={t}")

    def test_different_locations_both_alert_independently(self):
        p = make_processor()
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm_a = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_a.emit)

        # Camera cooldown (5s) has to clear before a second confirm can even be
        # attempted on this camera; use a track far enough in time AND space.
        p.evaluate("cam_01", {"label": "Faint"}, 8.0, track_id="C", detection=detection(LOCATION_FAR))
        d_confirm_c = p.evaluate("cam_01", {"label": "Faint"}, 9.0, track_id="C", detection=detection(LOCATION_FAR))
        self.assertTrue(d_confirm_c.emit, "a genuinely different location must not be suppressed")
        self.assertEqual(d_confirm_c.kind, "new_fall")

    def test_window_expiry_allows_new_fall_again(self):
        p = make_processor(spatial_dedup_seconds=60.0)
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))

        # Well past the 60s window and past camera cooldown.
        p.evaluate("cam_01", {"label": "Faint"}, 65.0, track_id="D", detection=detection(LOCATION_A))
        d_confirm_d = p.evaluate("cam_01", {"label": "Faint"}, 66.0, track_id="D", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_d.emit, "outside the dedup window, the same spot should alert again")

    def test_spatial_dedup_disabled_falls_back_to_pre_existing_behavior(self):
        p = make_processor(spatial_dedup_enabled=False)
        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))

        # Same spot, new track_id, well past camera cooldown but within what would be
        # the spatial dedup window — with the feature off this must NOT be suppressed.
        p.evaluate("cam_01", {"label": "Faint"}, 8.0, track_id="B", detection=detection(LOCATION_A))
        d_confirm_b = p.evaluate("cam_01", {"label": "Faint"}, 9.0, track_id="B", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_b.emit)

    def test_chain_of_three_suppressions_keeps_original_event_id(self):
        """A confirms+publishes, then B, C both get suppressed in turn (B never
        confirms either — it's dropped before D), and D is the one that finally
        survives long enough to escalate. original_event_id must chain all the way
        back to A at every step, never drift to B or C along the way."""
        p = make_processor()

        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm_a = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_a.emit)
        self.assertEqual(d_confirm_a.kind, "new_fall")

        p.evaluate("cam_01", {"label": "Faint"}, 3.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        d_confirm_b = p.evaluate("cam_01", {"label": "Faint"}, 4.0, track_id="B", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_b.emit)
        self.assertEqual(
            p._state_machine.get_track("cam_01", "B").last_event_id,
            d_confirm_a.event_id,
        )

        p.evaluate("cam_01", {"label": "Faint"}, 5.0, track_id="C", detection=detection(LOCATION_A_JITTERED))
        d_confirm_c = p.evaluate("cam_01", {"label": "Faint"}, 6.0, track_id="C", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_c.emit)
        self.assertEqual(
            p._state_machine.get_track("cam_01", "C").last_event_id,
            d_confirm_a.event_id,
            "C must still chain back to A, not to B's (never-published) event_id",
        )
        # The stored location record itself must not have drifted to B or C either.
        self.assertEqual(
            p._recent_fall_locations["cam_01"][0]["event_id"],
            d_confirm_a.event_id,
        )

        p.evaluate("cam_01", {"label": "Faint"}, 7.0, track_id="D", detection=detection(LOCATION_A_JITTERED))
        d_confirm_d = p.evaluate("cam_01", {"label": "Faint"}, 8.0, track_id="D", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_d.emit)
        self.assertEqual(
            p._state_machine.get_track("cam_01", "D").last_event_id,
            d_confirm_a.event_id,
        )

        # D survives (no further churn) long enough to escalate.
        d_unrec_d = p.evaluate("cam_01", {"label": "Faint"}, 18.5, track_id="D", detection=detection(LOCATION_A_JITTERED))
        self.assertTrue(d_unrec_d.emit)
        self.assertEqual(d_unrec_d.kind, "unrecovered")
        self.assertEqual(
            d_unrec_d.original_event_id,
            d_confirm_a.event_id,
            "after a 3-hop chain (A->B->C->D), the unrecovered alert must still cite A",
        )

    def test_expired_location_does_not_leak_stale_event_id_to_unrelated_fall(self):
        """After the dedup window elapses, the old location entry (and its event_id)
        must be gone — a later, unrelated confirm at the same spot must get its OWN
        fresh event_id, not silently inherit the expired one."""
        p = make_processor(spatial_dedup_seconds=60.0)

        p.evaluate("cam_01", {"label": "Faint"}, 1.0, track_id="A", detection=detection(LOCATION_A))
        d_confirm_a = p.evaluate("cam_01", {"label": "Faint"}, 2.0, track_id="A", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_a.emit)

        # Past both the camera cooldown and the 60s dedup window: a fresh incident at
        # the same physical spot must publish independently, with its own event_id.
        p.evaluate("cam_01", {"label": "Faint"}, 70.0, track_id="E", detection=detection(LOCATION_A))
        d_confirm_e = p.evaluate("cam_01", {"label": "Faint"}, 71.0, track_id="E", detection=detection(LOCATION_A))
        self.assertTrue(d_confirm_e.emit)
        self.assertEqual(d_confirm_e.kind, "new_fall")
        self.assertNotEqual(
            d_confirm_e.event_id,
            d_confirm_a.event_id,
            "E is a genuinely new incident and must not inherit A's stale event_id",
        )

        # The location table must hold only E's fresh record now, not a leftover
        # A entry sitting alongside it.
        locations = p._recent_fall_locations["cam_01"]
        self.assertEqual(len(locations), 1, "the expired A entry must have been pruned, not accumulated")
        self.assertEqual(locations[0]["event_id"], d_confirm_e.event_id)

        # A later track landing on the same spot shortly after E must now chain to E,
        # proving the lineage genuinely moved on and isn't still anchored to stale A.
        p.evaluate("cam_01", {"label": "Faint"}, 72.0, track_id="F", detection=detection(LOCATION_A_JITTERED))
        d_confirm_f = p.evaluate("cam_01", {"label": "Faint"}, 73.0, track_id="F", detection=detection(LOCATION_A_JITTERED))
        self.assertFalse(d_confirm_f.emit)
        self.assertEqual(
            p._state_machine.get_track("cam_01", "F").last_event_id,
            d_confirm_e.event_id,
        )
        self.assertNotEqual(
            p._state_machine.get_track("cam_01", "F").last_event_id,
            d_confirm_a.event_id,
        )


if __name__ == "__main__":
    unittest.main()
