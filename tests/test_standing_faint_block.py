"""Standing Faint false-positive guards: lifecycle + motion conf gating."""
from __future__ import annotations

import unittest

import numpy as np

from ai.action.classifier import keypoint_sequence_to_features, sequence_to_lstm_features
from ai.action.fall_event_state import FallEventStateMachine, FallState, LifecycleKind
from ai.action.faint_post_processing import FaintEventPostProcessor
from ai.action.feature_schema import KEYPOINT_MOTION54_SCHEMA_VERSION
from ai.action.motion_features import SAFE_TORSO_ANGLE_NORM, append_motion_features


def _pose(y: float, conf: float = 0.9, x: float = 100.0) -> list[dict]:
    points = [{"x": 0.0, "y": 0.0, "confidence": 0.0} for _ in range(17)]
    for idx in (5, 6, 11, 12):
        ox = -10 if idx in (5, 11) else 10
        oy = -50 if idx in (5, 6) else 0
        points[idx] = {"x": x + ox, "y": y + oy, "confidence": conf}
    return points


class StandingFaintBlockTest(unittest.TestCase):
    def test_standing_faint_blocked_no_new_fall_and_no_consecutive(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=0.0,
            block_upright_faint=True,
            use_posture_estimator=False,
        )
        pred = {"label": "Faint", "score": 0.9, "probabilities": {"Normal": 0.1, "Faint": 0.9}}
        d1 = proc.evaluate(
            "cam_01",
            pred,
            1.0,
            track_id=7,
            posture_label="upright_like",
            upright_to_lying=False,
        )
        d2 = proc.evaluate(
            "cam_01",
            pred,
            2.0,
            track_id=7,
            posture_label="upright_like",
            upright_to_lying=False,
        )
        self.assertFalse(d1.emit)
        self.assertFalse(d2.emit)
        self.assertEqual(d2.memo_text, "blocked_currently_upright_without_transition")
        self.assertEqual(proc.consecutive_count("cam_01", 7), 0)
        self.assertEqual(proc._state_machine.get_state("cam_01", 7), FallState.NORMAL)
        self.assertIn("upright", d2.lifecycle.reason or "")

    def test_upright_to_lying_allows_new_fall(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=0.0,
            block_upright_faint=True,
            use_posture_estimator=False,
        )
        pred = {"label": "Faint", "score": 0.85, "probabilities": {"Normal": 0.15, "Faint": 0.85}}
        # Establish upright history then transition
        proc.evaluate(
            "cam_01",
            {"label": "Normal", "score": 0.9},
            0.5,
            track_id=3,
            posture_label="upright_like",
            upright_to_lying=False,
        )
        d1 = proc.evaluate(
            "cam_01",
            pred,
            1.0,
            track_id=3,
            posture_label="lying_like",
            upright_to_lying=True,
        )
        d2 = proc.evaluate(
            "cam_01",
            pred,
            2.0,
            track_id=3,
            posture_label="lying_like",
            upright_to_lying=False,
        )
        self.assertFalse(d1.emit)  # first consecutive
        self.assertTrue(d2.emit)
        self.assertTrue(d2.is_new_fall)
        self.assertEqual(d2.lifecycle.kind, LifecycleKind.NEW_FALL)

    def test_state_machine_blocks_upright_without_transition(self):
        sm = FallEventStateMachine(min_consecutive_faint=2, block_upright_faint=True)
        sm.update("cam", 1.0, track_id=1, is_alert=True, posture_label="upright_like")
        d = sm.update("cam", 2.0, track_id=1, is_alert=True, posture_label="upright_like")
        self.assertEqual(d.kind, LifecycleKind.NONE)
        self.assertEqual(d.reason, "blocked_currently_upright_without_transition")
        self.assertEqual(sm.get_state("cam", 1), FallState.NORMAL)


class MotionConfidenceGatingTest(unittest.TestCase):
    def test_low_confidence_zeros_motion_despite_coord_jump(self):
        # Two frames with large spatial jump but low hip conf
        rows = []
        for y in (100.0, 300.0):
            feat = []
            for i in range(17):
                conf = 0.05 if i in (5, 6, 11, 12) else 0.9
                feat.extend([50.0 / 640.0, y / 480.0, conf])
            rows.append(feat)
        base = np.asarray(rows, dtype=np.float32)
        validity = {}
        out = append_motion_features(base, min_keypoint_conf=0.3, validity_out=validity)
        self.assertEqual(out.shape, (2, 54))
        self.assertEqual(float(out[1, 51]), 0.0)
        self.assertEqual(float(out[1, 52]), 0.0)
        self.assertAlmostEqual(float(out[1, 53]), SAFE_TORSO_ANGLE_NORM, places=5)
        self.assertEqual(validity["low_conf_motion_frames"], 2)

    def test_valid_confidence_keeps_raw_displacement(self):
        rows = []
        for y in (200.0, 220.0):
            feat = []
            for i in range(17):
                conf = 0.9
                # approximate midpoints via placing all kps at (100,y) for simplicity
                # shoulders y-50, hips y — still both conf valid
                yy = (y - 50) if i in (5, 6) else y
                feat.extend([100.0 / 640.0, yy / 480.0, conf])
            rows.append(feat)
        base = np.asarray(rows, dtype=np.float32)
        out = append_motion_features(base, min_keypoint_conf=0.3)
        expected = (220.0 - 200.0) / 480.0  # hip mid y uses hip rows which are at y
        self.assertAlmostEqual(float(out[1, 51]), expected, places=5)
        self.assertAlmostEqual(float(out[1, 52]), abs(expected), places=5)

    def test_discontinuity_and_low_conf_together(self):
        rows = []
        for y in (100.0, 400.0):
            feat = []
            for i in range(17):
                conf = 0.1
                feat.extend([0.2, y / 480.0, conf])
            rows.append(feat)
        base = np.asarray(rows, dtype=np.float32)
        disc = np.array([True, True])
        out = append_motion_features(base, discontinuity_mask=disc, min_keypoint_conf=0.3)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[1, 51]), 0.0)
        self.assertEqual(float(out[1, 52]), 0.0)
        self.assertEqual(out.shape, (2, 54))

    def test_sequence_path_keeps_54_order(self):
        detections = []
        shapes = []
        for t, y in enumerate([200.0, 205.0, 210.0]):
            detections.append({"keypoints": _pose(y, conf=0.9), "bbox": [50, y - 80, 150, y + 20]})
            shapes.append((480, 640, 3))
        seq = {
            "detections": detections,
            "frame_shapes": shapes,
            "frame_ids": [0, 1, 2],
            "frame_idxs": [0, 1, 2],
            "sample_captured_at_ms": [0, 33, 66],
        }
        feats = sequence_to_lstm_features(seq, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        self.assertEqual(feats.shape, (3, 54))
        self.assertIn("motion_validity", seq)
        self.assertGreater(float(feats[2, 52]), 0.0)


if __name__ == "__main__":
    unittest.main()
