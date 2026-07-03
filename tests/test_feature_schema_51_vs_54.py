import unittest
import numpy as np
import tempfile
from pathlib import Path
import torch

from ai.action.classifier import (
    sequence_to_lstm_features,
    keypoint_sequence_to_features,
    LSTMActionClassifier,
    LSTMActionModel,
)
from ai.action.feature_schema import (
    KEYPOINT_BBOX54_INPUT_SIZE,
    KEYPOINT_BBOX54_SCHEMA_VERSION,
    keypoint_bbox54_feature_names,
)
from ai.action.motion_features import append_motion_features


class TestFeatureSchema51Vs54(unittest.TestCase):
    def setUp(self):
        # Create standard test sequence
        self.seq_len = 16
        detections = []
        frame_shapes = []
        for t in range(self.seq_len):
            kps = []
            for i in range(17):
                kps.append({
                    "x": 100.0 + i * 10 + t * 5,
                    "y": 150.0 + i * 12 + t * 4,
                    "confidence": 0.9 if i % 2 == 0 else 0.4
                })
            # bbox: x1, y1, x2, y2
            bbox = [10.0, 20.0, 110.0 + t, 220.0 + t * 2]
            detections.append({
                "bbox": bbox,
                "keypoints": kps
            })
            frame_shapes.append((1080, 1920, 3))

        self.sequence = {
            "detections": detections,
            "frame_shapes": frame_shapes
        }

    def test_feature_builder_returns_correct_shapes(self):
        # 1. 51-dimensional schema shape
        feat_51 = sequence_to_lstm_features(
            self.sequence, input_size=51, feature_schema="keypoint51"
        )
        self.assertEqual(feat_51.shape, (self.seq_len, 51))

        # 2. 54-dimensional motion schema shape
        feat_motion54 = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_motion54"
        )
        self.assertEqual(feat_motion54.shape, (self.seq_len, 54))

        # 3. 54-dimensional bbox schema shape
        feat_bbox54 = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_bbox54"
        )
        self.assertEqual(feat_bbox54.shape, (self.seq_len, 54))

    def test_prefix_matches_exactly(self):
        feat_51 = sequence_to_lstm_features(
            self.sequence, input_size=51, feature_schema="keypoint51"
        )
        feat_motion54 = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_motion54"
        )
        feat_bbox54 = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_bbox54"
        )

        # Front 51 values must be exactly equal
        np.testing.assert_array_almost_equal(feat_51, feat_motion54[:, :51])
        np.testing.assert_array_almost_equal(feat_51, feat_bbox54[:, :51])

    def test_bbox_missing_fallback(self):
        # Create sequence where bbox is missing in some frames
        seq_missing_bbox = {
            "detections": [
                {
                    "bbox": None,
                    "keypoints": [{"x": 100.0, "y": 200.0, "confidence": 0.9} for _ in range(17)]
                }
                for _ in range(self.seq_len)
            ],
            "frame_shapes": [(1080, 1920, 3) for _ in range(self.seq_len)]
        }
        
        feat_bbox54 = sequence_to_lstm_features(
            seq_missing_bbox, input_size=54, feature_schema="keypoint_bbox54"
        )
        
        # The last 3 values should fall back to 0.0
        self.assertEqual(feat_bbox54.shape[-1], 54)
        np.testing.assert_array_equal(feat_bbox54[:, 51:], np.zeros((self.seq_len, 3)))

    def test_lstm_forward_pass_51_and_54(self):
        # Test input size 51
        model_51 = LSTMActionModel(input_size=51).model
        model_51.eval()
        x_51 = torch.randn(2, self.seq_len, 51)
        with torch.no_grad():
            out_51 = model_51(x_51)
        self.assertEqual(out_51.shape, (2, 2))

        # Test input size 54
        model_54 = LSTMActionModel(input_size=54).model
        model_54.eval()
        x_54 = torch.randn(2, self.seq_len, 54)
        with torch.no_grad():
            out_54 = model_54(x_54)
        self.assertEqual(out_54.shape, (2, 2))

    def test_checkpoint_mismatch_fails_immediately(self):
        # Save a mock checkpoint with input_size=51
        model_config = {
            "input_size": 51,
            "hidden_size": 64,
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.0
        }
        model = LSTMActionModel(**model_config)
        
        checkpoint_payload = {
            "model_state": model.model.state_dict(),
            "model_config": model_config,
            "classes": ["Normal", "Faint"],
            "input_size": 51,
            "feature_schema_version": "keypoint51",
            "feature_names": [f"kp{i}" for i in range(51)]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model_51.pt"
            torch.save(checkpoint_payload, ckpt_path)

            # Load it into LSTMActionClassifier
            classifier = LSTMActionClassifier(str(ckpt_path), device="cpu")
            self.assertEqual(classifier.input_size, 51)

            # Try to run predict on a 54-dimensional mock sequence
            # Create a mock sequence that generates 54 dimension because we simulate a 54-dim input sequence
            # or wait, if the classifier has self.input_size = 51, it calls sequence_to_lstm_features with input_size=51,
            # which produces 51 features. So it succeeds.
            # But what if we change classifier.input_size = 54 (or force sequence to yield 54 features)?
            # Yes, the predict method checks:
            # actual_input_size = int(features.shape[-1])
            # if actual_input_size != self.input_size: raise ValueError
            # So let's test that mismatch check directly
            
            # 1. Test running classifier on matching input size works
            classifier.predict(self.sequence) # matches 51-input size

            # 2. Force input size mismatch by patching feature generation shape
            import ai.action.classifier
            orig_seq_to_lstm = ai.action.classifier.sequence_to_lstm_features
            try:
                ai.action.classifier.sequence_to_lstm_features = lambda *args, **kwargs: np.zeros((self.seq_len, 54), dtype=np.float32)
                with self.assertRaises(ValueError):
                    classifier.predict(self.sequence)
            finally:
                ai.action.classifier.sequence_to_lstm_features = orig_seq_to_lstm

            # 3. Test constructor safeguard checks metadata mismatch (e.g. input_size=51 but schema=keypoint_bbox54)
            bad_payload = dict(checkpoint_payload)
            bad_payload["feature_schema_version"] = "keypoint_bbox54"
            ckpt_bad_path = Path(tmpdir) / "test_model_bad.pt"
            torch.save(bad_payload, ckpt_bad_path)
            with self.assertRaises(ValueError):
                LSTMActionClassifier(str(ckpt_bad_path), device="cpu")

    def test_54_dim_checkpoint_without_schema_fails_clearly(self):
        model_config = {
            "input_size": 54,
            "hidden_size": 64,
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.0,
        }
        model = LSTMActionModel(**model_config)
        checkpoint_payload = {
            "model_state": model.model.state_dict(),
            "model_config": model_config,
            "classes": ["Normal", "Faint"],
            "input_size": 54,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model_54_no_schema.pt"
            torch.save(checkpoint_payload, ckpt_path)

            with self.assertRaisesRegex(ValueError, "feature_schema_version"):
                LSTMActionClassifier(str(ckpt_path), device="cpu")

    def test_feature_builders_are_identical(self):
        # Verify training-time keypoint extraction matches inference-time extraction
        feat_train = keypoint_sequence_to_features(
            self.sequence, expected_input_size=54, feature_schema="keypoint_bbox54"
        )
        
        # Instantiate mock sequence inside inference classifier
        feat_inf = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_bbox54"
        )
        
        np.testing.assert_array_equal(feat_train, feat_inf)

    def test_keypoint_bbox54_schema_documents_exact_feature_order(self):
        names = keypoint_bbox54_feature_names()

        self.assertEqual(KEYPOINT_BBOX54_SCHEMA_VERSION, "keypoint_bbox54")
        self.assertEqual(KEYPOINT_BBOX54_INPUT_SIZE, 54)
        self.assertEqual(len(names), 54)
        self.assertEqual(names[:6], ["kp0_x", "kp0_y", "kp0_conf", "kp1_x", "kp1_y", "kp1_conf"])
        self.assertEqual(names[51:], ["bbox_width_norm", "bbox_height_norm", "bbox_area_norm"])

    def test_keypoint_bbox54_appends_normalized_bbox_width_height_area(self):
        features = sequence_to_lstm_features(
            self.sequence, input_size=54, feature_schema="keypoint_bbox54"
        )

        self.assertAlmostEqual(float(features[0][51]), 100.0 / 1920.0)
        self.assertAlmostEqual(float(features[0][52]), 200.0 / 1080.0)
        self.assertAlmostEqual(float(features[0][53]), (100.0 / 1920.0) * (200.0 / 1080.0))

    def test_keypoint_bbox54_rejects_silent_padding_from_51_dim_features(self):
        features = np.ones((self.seq_len, 51), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "keypoint_bbox54"):
            from ai.action.classifier import normalize_feature_width

            normalize_feature_width(features, 54, feature_schema="keypoint_bbox54")

    def test_classifier_shape_mismatch_error_includes_camera_and_track_context(self):
        model_config = {
            "input_size": 54,
            "hidden_size": 64,
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.0,
        }
        model = LSTMActionModel(**model_config)
        checkpoint_payload = {
            "model_state": model.model.state_dict(),
            "model_config": model_config,
            "classes": ["Normal", "Faint"],
            "input_size": 54,
            "feature_schema_version": "keypoint_bbox54",
            "feature_names": keypoint_bbox54_feature_names(),
        }
        sequence = dict(self.sequence)
        sequence["camera_login_id"] = "cam_05"
        sequence["track_id"] = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model_54.pt"
            torch.save(checkpoint_payload, ckpt_path)
            classifier = LSTMActionClassifier(str(ckpt_path), device="cpu")
            import ai.action.classifier
            original = ai.action.classifier.sequence_to_lstm_features
            try:
                ai.action.classifier.sequence_to_lstm_features = lambda *args, **kwargs: np.zeros((self.seq_len, 51), dtype=np.float32)
                with self.assertRaisesRegex(ValueError, "cameraLoginId=cam_05.*trackId=7.*expected input_size=54.*runtime_feature_dim=51"):
                    classifier.predict(sequence)
            finally:
                ai.action.classifier.sequence_to_lstm_features = original


if __name__ == "__main__":
    unittest.main()
