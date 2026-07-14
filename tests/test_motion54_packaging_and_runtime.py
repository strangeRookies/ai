"""Regression tests for keypoint_motion54 packaging and runtime routing."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ai.action.classifier import (
    LSTMActionClassifier,
    LSTMActionModel,
    keypoint_sequence_to_features,
    sequence_to_lstm_features,
)
from ai.action.feature_schema import (
    KEYPOINT_BBOX54_SCHEMA_VERSION,
    KEYPOINT_MOTION54_SCHEMA_VERSION,
    keypoint_motion54_feature_names,
)
from ai.action.motion_features import append_motion_features
from scripts.package_motion54_checkpoint import model_states_equal, package_checkpoint


def _standing_pose(y_offset: float = 0.0, x_offset: float = 0.0) -> list[dict]:
    # COCO indices used by motion_features: 5/6 shoulders, 11/12 hips
    points = [{"x": 0.0, "y": 0.0, "confidence": 0.0} for _ in range(17)]
    # shoulders around y=100, hips around y=200 (+ offsets)
    points[5] = {"x": 90.0 + x_offset, "y": 100.0 + y_offset, "confidence": 0.9}
    points[6] = {"x": 110.0 + x_offset, "y": 100.0 + y_offset, "confidence": 0.9}
    points[11] = {"x": 95.0 + x_offset, "y": 200.0 + y_offset, "confidence": 0.9}
    points[12] = {"x": 105.0 + x_offset, "y": 200.0 + y_offset, "confidence": 0.9}
    return points


def _base_51_sequence(frames: list[tuple[float, float]]) -> np.ndarray:
    """Build (T,51) features via shipped keypoint path for given hip y/x offsets."""
    detections = []
    shapes = []
    for y_off, x_off in frames:
        detections.append({"keypoints": _standing_pose(y_offset=y_off, x_offset=x_off), "bbox": [50, 50, 150, 250]})
        shapes.append((480, 640, 3))
    seq = {"detections": detections, "frame_shapes": shapes}
    return keypoint_sequence_to_features(seq, expected_input_size=51, feature_schema="keypoint51")


def _make_raw_motion54_checkpoint(path: Path, *, with_schema: bool = False) -> Path:
    model_config = {
        "input_size": 54,
        "hidden_size": 32,
        "num_layers": 1,
        "num_classes": 2,
        "dropout": 0.0,
    }
    model = LSTMActionModel(**model_config)
    payload = {
        "model_state": model.model.state_dict(),
        "model_config": model_config,
        "classes": ["Normal", "Faint"],
        "input_size": 54,
        "feature_type": "keypoints",
        # Intentionally omit schema/names to mimic legacy experiment checkpoint
        "sequence_length": 30,
    }
    if with_schema:
        payload["feature_schema_version"] = KEYPOINT_MOTION54_SCHEMA_VERSION
        payload["feature_names"] = keypoint_motion54_feature_names()
        payload["sequence_stride"] = 15
    torch.save(payload, path)
    return path


class MotionFeatureSemanticsTest(unittest.TestCase):
    def test_feature_order_center_drop_velocity_torso(self):
        base = _base_51_sequence([(0.0, 0.0), (20.0, 0.0)])  # downward
        out = append_motion_features(base)
        self.assertEqual(out.shape, (2, 54))
        # dim 51 center_drop, 52 velocity, 53 torso_angle_norm
        self.assertGreater(float(out[1, 51]), 0.0)
        self.assertGreater(float(out[1, 52]), 0.0)
        self.assertGreaterEqual(float(out[0, 53]), 0.0)
        self.assertLessEqual(float(out[0, 53]), 1.0)

    def test_first_frame_drop_and_velocity_zero(self):
        base = _base_51_sequence([(0.0, 0.0), (10.0, 5.0)])
        out = append_motion_features(base)
        self.assertEqual(float(out[0, 51]), 0.0)
        self.assertEqual(float(out[0, 52]), 0.0)
        self.assertTrue(np.isfinite(out[0, 53]))

    def test_downward_center_drop_positive(self):
        base = _base_51_sequence([(0.0, 0.0), (30.0, 0.0)])
        out = append_motion_features(base)
        self.assertGreater(float(out[1, 51]), 0.0)

    def test_stationary_zero_motion(self):
        base = _base_51_sequence([(5.0, 2.0), (5.0, 2.0)])
        out = append_motion_features(base)
        self.assertAlmostEqual(float(out[1, 51]), 0.0, places=6)
        self.assertAlmostEqual(float(out[1, 52]), 0.0, places=6)

    def test_horizontal_move_velocity_positive_drop_near_zero(self):
        base = _base_51_sequence([(0.0, 0.0), (0.0, 40.0)])
        out = append_motion_features(base)
        self.assertAlmostEqual(float(out[1, 51]), 0.0, places=6)
        self.assertGreater(float(out[1, 52]), 0.0)

    def test_shape_30_51_to_30_54(self):
        base = _base_51_sequence([(float(i), 0.0) for i in range(30)])
        self.assertEqual(base.shape, (30, 51))
        out = append_motion_features(base)
        self.assertEqual(out.shape, (30, 54))

    def test_runtime_schema_routing_motion_not_bbox(self):
        detections = []
        shapes = []
        for i in range(4):
            detections.append(
                {
                    "keypoints": _standing_pose(y_offset=float(i * 5)),
                    "bbox": [10, 20, 110, 220],
                }
            )
            shapes.append((480, 640, 3))
        sequence = {"detections": detections, "frame_shapes": shapes}
        motion = sequence_to_lstm_features(sequence, input_size=54, feature_schema=KEYPOINT_MOTION54_SCHEMA_VERSION)
        bbox = sequence_to_lstm_features(sequence, input_size=54, feature_schema=KEYPOINT_BBOX54_SCHEMA_VERSION)
        self.assertEqual(motion.shape, (4, 54))
        self.assertEqual(bbox.shape, (4, 54))
        # Motion dims are not bbox width/height/area norms from this fixture.
        self.assertFalse(np.allclose(motion[:, 51:], bbox[:, 51:]))


class Motion54PackagingTest(unittest.TestCase):
    def test_package_adds_metadata_without_changing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "best.pt"
            dst = Path(tmp) / "best_motion54_packaged.pt"
            _make_raw_motion54_checkpoint(src, with_schema=False)
            before = torch.load(src, map_location="cpu")
            report = package_checkpoint(src, dst, sequence_length=30, sequence_stride=None, dry_run=False)
            self.assertTrue(report["model_state_identical"])
            self.assertTrue(report["original_unchanged"])
            after = torch.load(dst, map_location="cpu")
            self.assertEqual(after["feature_schema_version"], KEYPOINT_MOTION54_SCHEMA_VERSION)
            self.assertEqual(after["input_size"], 54)
            self.assertEqual(after["model_config"]["input_size"], 54)
            self.assertEqual(len(after["feature_names"]), 54)
            self.assertEqual(after["feature_names"][51:], ["center_drop", "velocity", "torso_angle_norm"])
            self.assertEqual(after["sequence_length"], 30)
            self.assertEqual(after["sequence_stride"], 15)
            self.assertTrue(model_states_equal(before["model_state"], after["model_state"]))
            for key in before["model_state"]:
                self.assertTrue(torch.equal(before["model_state"][key], after["model_state"][key]))
            # Original file still loadable as raw (no schema forced rewrite)
            original = torch.load(src, map_location="cpu")
            self.assertIsNone(original.get("feature_schema_version"))

    def test_package_rejects_bbox54(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bbox.pt"
            dst = Path(tmp) / "out.pt"
            model_config = {"input_size": 54, "hidden_size": 16, "num_layers": 1, "num_classes": 2, "dropout": 0.0}
            model = LSTMActionModel(**model_config)
            torch.save(
                {
                    "model_state": model.model.state_dict(),
                    "model_config": model_config,
                    "classes": ["Normal", "Faint"],
                    "feature_schema_version": KEYPOINT_BBOX54_SCHEMA_VERSION,
                },
                src,
            )
            with self.assertRaisesRegex(ValueError, "keypoint_bbox54"):
                package_checkpoint(src, dst, sequence_length=30, sequence_stride=15, dry_run=False)

    def test_schema_collision_motion_checkpoint_vs_bbox_runtime(self):
        """motion54 metadata must fail-fast when forced into bbox54 (and reverse)."""
        from ai.action.feature_schema import keypoint_bbox54_feature_names

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "best.pt"
            packaged = Path(tmp) / "packaged.pt"
            _make_raw_motion54_checkpoint(src)
            package_checkpoint(src, packaged, sequence_length=30, sequence_stride=15, dry_run=False)

            # Happy path still works for packaged motion54.
            clf_motion = LSTMActionClassifier(str(packaged), device="cpu")
            self.assertEqual(clf_motion.feature_schema, KEYPOINT_MOTION54_SCHEMA_VERSION)

            # motion54 weights + motion names but schema flipped to bbox54 → collision fail
            motion_as_bbox = torch.load(packaged, map_location="cpu")
            motion_as_bbox["feature_schema_version"] = KEYPOINT_BBOX54_SCHEMA_VERSION
            motion_as_bbox_path = Path(tmp) / "motion_as_bbox.pt"
            torch.save(motion_as_bbox, motion_as_bbox_path)
            with self.assertRaisesRegex(ValueError, "schema collision|does not match feature_names"):
                LSTMActionClassifier(str(motion_as_bbox_path), device="cpu")

            # bbox54 names + motion54 schema → collision fail
            bbox_as_motion = torch.load(packaged, map_location="cpu")
            bbox_as_motion["feature_schema_version"] = KEYPOINT_MOTION54_SCHEMA_VERSION
            bbox_as_motion["feature_names"] = keypoint_bbox54_feature_names()
            bbox_as_motion_path = Path(tmp) / "bbox_as_motion.pt"
            torch.save(bbox_as_motion, bbox_as_motion_path)
            with self.assertRaisesRegex(ValueError, "schema collision|does not match feature_names"):
                LSTMActionClassifier(str(bbox_as_motion_path), device="cpu")

            # Packaging path also rejects bbox54 sources for motion package
            bbox_src = Path(tmp) / "bbox_src.pt"
            model_config = {"input_size": 54, "hidden_size": 16, "num_layers": 1, "num_classes": 2, "dropout": 0.0}
            model = LSTMActionModel(**model_config)
            torch.save(
                {
                    "model_state": model.model.state_dict(),
                    "model_config": model_config,
                    "classes": ["Normal", "Faint"],
                    "feature_schema_version": KEYPOINT_BBOX54_SCHEMA_VERSION,
                    "feature_names": keypoint_bbox54_feature_names(),
                },
                bbox_src,
            )
            with self.assertRaisesRegex(ValueError, "keypoint_bbox54"):
                package_checkpoint(bbox_src, Path(tmp) / "out.pt", sequence_length=30, sequence_stride=15)

    def test_bbox_checkpoint_rejects_as_motion_package_source_and_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            # motion schema with wrong input_size fails load
            model_config = {"input_size": 51, "hidden_size": 16, "num_layers": 1, "num_classes": 2, "dropout": 0.0}
            model = LSTMActionModel(**model_config)
            path = Path(tmp) / "mismatch.pt"
            torch.save(
                {
                    "model_state": model.model.state_dict(),
                    "model_config": model_config,
                    "classes": ["Normal", "Faint"],
                    "feature_schema_version": KEYPOINT_MOTION54_SCHEMA_VERSION,
                    "feature_names": keypoint_motion54_feature_names(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "Metadata Mismatch"):
                LSTMActionClassifier(str(path), device="cpu")


class Motion54ClassifierLoadTest(unittest.TestCase):
    def test_load_packaged_and_emit_contract_fields(self):
        from ai.inference.rtsp_runtime import classifier_contract_summary, log_classifier_contract
        import io
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "best.pt"
            dst = Path(tmp) / "best_motion54_packaged.pt"
            _make_raw_motion54_checkpoint(src)
            package_checkpoint(src, dst, sequence_length=30, sequence_stride=15)
            clf = LSTMActionClassifier(str(dst), device="cpu")
            summary = classifier_contract_summary(clf)
            self.assertEqual(summary["checkpoint_input_size"], 54)
            self.assertEqual(summary["feature_schema_version"], KEYPOINT_MOTION54_SCHEMA_VERSION)
            self.assertEqual(summary["feature_names_count"], 54)
            self.assertEqual(summary["sequence_length"], 30)
            self.assertEqual(summary["sequence_stride"], 15)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                log_classifier_contract("[lstm-checkpoint]", "cam_03", clf)
            text = buf.getvalue()
            self.assertIn("[lstm-contract]", text)
            self.assertIn("feature_schema=keypoint_motion54", text)
            self.assertIn("input_size=54", text)
            self.assertIn("cameraLoginId=cam_03", text)


if __name__ == "__main__":
    unittest.main()
