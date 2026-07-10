"""Phase B: posture estimator unit tests."""

from __future__ import annotations

import unittest

from ai.action.posture_estimator import (
    PostureEstimator,
    detection_for_track,
    shoulder_hip_angle_degrees,
)


def _kp(x, y, c=0.9):
    return {"x": x, "y": y, "confidence": c}


def _coco_upright():
    # rough upright person: shoulders above hips, tall bbox
    kps = [{"x": 0, "y": 0, "confidence": 0.0}] * 17
    kps[5] = _kp(40, 40)
    kps[6] = _kp(60, 40)
    kps[11] = _kp(42, 120)
    kps[12] = _kp(58, 120)
    return kps


def _coco_lying():
    # Nearly horizontal torso (shoulder-hip line along x-axis)
    kps = [{"x": 0, "y": 0, "confidence": 0.0}] * 17
    kps[5] = _kp(40, 90)
    kps[6] = _kp(50, 90)
    kps[11] = _kp(120, 92)
    kps[12] = _kp(130, 92)
    return kps


class PostureEstimatorTest(unittest.TestCase):
    def test_upright_bbox_and_keypoints(self):
        est = PostureEstimator(upright_frames_required=1, lying_frames_required=1)
        det = {
            "bbox": [30, 20, 70, 160],  # tall
            "keypoints": _coco_upright(),
            "pose_horizontal": False,
        }
        p = est.estimate("cam:1", det)
        self.assertEqual(p.label, "upright_like")
        self.assertGreater(p.aspect_hw or 0, 1.0)

    def test_lying_bbox_and_keypoints(self):
        est = PostureEstimator(lying_frames_required=2, upright_frames_required=1)
        det = {
            "bbox": [20, 80, 160, 120],  # wide
            "keypoints": _coco_lying(),
            "pose_horizontal": True,
        }
        first = est.estimate("cam:1", det)
        self.assertEqual(first.label, "unknown")  # needs N consecutive lying frames
        second = est.estimate("cam:1", det)
        self.assertEqual(second.label, "lying_like")

    def test_lying_frames_required_blocks_first_frame(self):
        est = PostureEstimator(lying_frames_required=3, upright_frames_required=1)
        det = {
            "bbox": [20, 80, 160, 120],
            "keypoints": _coco_lying(),
            "pose_horizontal": True,
        }
        self.assertEqual(est.estimate("t", det).label, "unknown")
        self.assertEqual(est.estimate("t", det).label, "unknown")
        self.assertEqual(est.estimate("t", det).label, "lying_like")

    def test_upright_to_lying_transition_requires_history(self):
        est = PostureEstimator(lying_frames_required=1, upright_frames_required=1)
        upright = {
            "bbox": [30, 20, 70, 160],
            "keypoints": _coco_upright(),
            "pose_horizontal": False,
        }
        lying = {
            "bbox": [20, 80, 160, 120],
            "keypoints": _coco_lying(),
            "pose_horizontal": True,
        }
        # Start lying only → no transition
        first = est.estimate("t1", lying)
        self.assertEqual(first.label, "lying_like")
        self.assertFalse(first.upright_to_lying_transition)

        est2 = PostureEstimator(lying_frames_required=1, upright_frames_required=1)
        est2.estimate("t1", upright)
        est2.estimate("t1", upright)
        second = est2.estimate("t1", lying)
        self.assertTrue(second.upright_to_lying_transition)

    def test_shoulder_hip_angle_vertical_vs_horizontal(self):
        upright_angle = shoulder_hip_angle_degrees(_coco_upright())
        lying_angle = shoulder_hip_angle_degrees(_coco_lying())
        self.assertIsNotNone(upright_angle)
        self.assertIsNotNone(lying_angle)
        self.assertGreater(upright_angle, lying_angle)

    def test_detection_for_track(self):
        dets = [
            {"track_id": 1, "bbox": [0, 0, 1, 2]},
            {"track_id": 2, "bbox": [0, 0, 3, 1]},
        ]
        self.assertEqual(detection_for_track(dets, 2)["bbox"][2], 3)
        self.assertIsNone(detection_for_track(dets, 9))


if __name__ == "__main__":
    unittest.main()
