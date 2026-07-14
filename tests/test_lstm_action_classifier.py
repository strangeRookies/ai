import unittest
from pathlib import Path

import numpy as np

from ai.action.classifier import (
    DEFAULT_CLASSES,
    KEYPOINT_FEATURE_DIM,
    LSTMActionClassifier,
    MOTION_KEYPOINT_FEATURE_DIM,
    classes_from_checkpoint,
    crops_to_features,
    keypoint_sequence_to_features,
    normalize_feature_width,
    normalize_torch_device,
    sequence_to_lstm_features,
    threshold_prediction,
)
from ai.action.feature_schema import KEYPOINT_BBOX54_SCHEMA_VERSION, keypoint_bbox54_feature_names
from ai.action.train_lstm import build_checkpoint_payload, load_training_rows, summarize_metadata, target_ranges_for_row
from ai.inference.rtsp_runtime import classifier_contract_summary

try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class LSTMActionClassifierTest(unittest.TestCase):
    @unittest.skipIf(not CV2_AVAILABLE, "cv2 is required for crop feature tests")
    def test_crops_to_features_shape(self):
        crops = [
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30, 3), 255, dtype=np.uint8),
        ]

        features = crops_to_features(crops, feature_size=8)

        self.assertEqual(features.shape, (2, 64))
        self.assertEqual(float(features[0].max()), 0.0)
        self.assertEqual(float(features[1].min()), 1.0)

    def test_keypoint_sequence_to_features_matches_benchmark_shape(self):
        sequence = {
            "detections": [
                {
                    "keypoints": [
                        {"x": 50.0, "y": 25.0, "confidence": 0.9},
                        {"x": 20.0, "y": 10.0, "confidence": 0.1},
                    ]
                }
            ],
            "frame_shapes": [(100, 200, 3)],
        }

        features = keypoint_sequence_to_features(sequence)

        self.assertEqual(features.shape, (1, 51))
        self.assertAlmostEqual(float(features[0][0]), 0.25)
        self.assertAlmostEqual(float(features[0][1]), 0.25)
        self.assertAlmostEqual(float(features[0][2]), 0.9)

    def test_keypoint_sequence_to_features_can_emit_legacy_54_dim_motion_input(self):
        sequence = {
            "detections": [
                {"keypoints": [{"x": 50.0, "y": 25.0, "confidence": 0.9} for _ in range(17)]},
                {"keypoints": [{"x": 70.0, "y": 45.0, "confidence": 0.8} for _ in range(17)]},
            ],
            "frame_shapes": [(100, 200, 3), (100, 200, 3)],
        }

        features = keypoint_sequence_to_features(
            sequence,
            expected_input_size=MOTION_KEYPOINT_FEATURE_DIM,
            feature_schema="keypoint_motion54",
        )

        self.assertEqual(features.shape, (2, 54))

    def test_keypoint_sequence_to_features_matches_51_dim_checkpoint(self):
        sequence = {
            "detections": [
                {"keypoints": [{"x": 50.0, "y": 25.0, "confidence": 0.9} for _ in range(18)]}
            ],
            "frame_shapes": [(100, 200, 3)],
        }

        features = keypoint_sequence_to_features(sequence, expected_input_size=KEYPOINT_FEATURE_DIM)

        self.assertEqual(features.shape, (1, 51))
        self.assertAlmostEqual(float(features[0][0]), 0.25)
        self.assertAlmostEqual(float(features[0][1]), 0.25)
        self.assertAlmostEqual(float(features[0][2]), 0.9)

    def test_sequence_to_lstm_features_uses_checkpoint_input_size_for_keypoints(self):
        sequence = {
            "detections": [
                {"keypoints": [{"x": 10.0, "y": 20.0, "confidence": 0.8} for _ in range(17)]}
            ],
            "frame_shapes": [(100, 200, 3)],
        }

        features = sequence_to_lstm_features(sequence, input_size=51)

        self.assertEqual(features.shape, (1, 51))

    def test_normalize_feature_width_truncates_54_to_51(self):
        features = np.ones((2, 54), dtype=np.float32)

        normalized = normalize_feature_width(features, 51, camera_login_id="cam_04", checkpoint_path="model.pt")

        self.assertEqual(normalized.shape, (2, 51))

    @unittest.skipIf(not CV2_AVAILABLE, "cv2 is required for crop feature tests")
    def test_sequence_to_lstm_features_uses_crop_size_for_crop_checkpoint(self):
        crops = [
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.full((20, 30, 3), 255, dtype=np.uint8),
        ]

        features = sequence_to_lstm_features({"crops": crops}, input_size=64, crop_feature_size=8)

        self.assertEqual(features.shape, (2, 64))

    def test_classes_fallback_uses_normal_faint_and_preserves_checkpoint_classes(self):
        self.assertEqual(DEFAULT_CLASSES, ("Normal", "Faint"))
        self.assertEqual(classes_from_checkpoint({}), ["Normal", "Faint"])
        self.assertEqual(classes_from_checkpoint({"classes": ["Normal", "Fall"]}), ["Normal", "Fall"])

    def test_train_lstm_checkpoint_payload_records_crop_metadata(self):
        class Args:
            feature_size = 32
            sequence_length = 16
            sequence_stride = 8

        model_config = {
            "input_size": 1024,
            "hidden_size": 128,
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.0,
        }

        payload = build_checkpoint_payload({"weight": "state"}, model_config, Args(), 0.75, {"train": 1}, {"val": 1})

        self.assertEqual(payload["classes"], ["Normal", "Faint"])
        self.assertEqual(payload["sequence_stride"], 8)
        self.assertEqual(payload["feature_type"], "crop")
        self.assertEqual(payload["crop_feature_size"], 32)
        self.assertEqual(payload["feature_size"], 32)
        self.assertEqual(payload["input_size"], 1024)
        self.assertEqual(payload["label_mapping"], {"Normal": 0, "Faint": 1})

    def test_faint_keypoint54_checkpoint_payload_defaults_to_bbox54_schema(self):
        class Args:
            feature_size = 32
            sequence_length = 30
            sequence_stride = 15

        model_config = {
            "input_size": 54,
            "hidden_size": 128,
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.0,
        }

        payload = build_checkpoint_payload({"weight": "state"}, model_config, Args(), 0.75, {"train": 1}, {"val": 1})

        self.assertEqual(payload["classes"], ["Normal", "Faint"])
        self.assertEqual(payload["feature_type"], "keypoint")
        self.assertEqual(payload["input_size"], 54)
        self.assertEqual(payload["feature_schema_version"], KEYPOINT_BBOX54_SCHEMA_VERSION)
        self.assertEqual(payload["feature_names"], keypoint_bbox54_feature_names())

    def test_classifier_contract_summary_reports_checkpoint_schema_and_input_size(self):
        class FakeClassifier:
            checkpoint_path = "models/faint54.pt"
            input_size = 54
            feature_schema = "keypoint_bbox54"
            feature_names = keypoint_bbox54_feature_names()
            checkpoint_sequence_length = 30
            checkpoint_sequence_stride = 15

        summary = classifier_contract_summary(FakeClassifier())

        self.assertEqual(summary["checkpoint_path"], "models/faint54.pt")
        self.assertEqual(summary["checkpoint_input_size"], 54)
        self.assertEqual(summary["feature_schema_version"], "keypoint_bbox54")
        self.assertEqual(summary["feature_names_count"], 54)
        self.assertEqual(summary["sequence_length"], 30)
        self.assertEqual(summary["sequence_stride"], 15)

    def test_numeric_torch_device_is_normalized_for_lstm_classifier(self):
        class FakeCuda:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        class FakeTorch:
            def __init__(self, available):
                self.cuda = FakeCuda(available)

        self.assertEqual(normalize_torch_device("0", FakeTorch(True)), "cuda:0")
        self.assertEqual(normalize_torch_device("0", FakeTorch(False)), "cpu")

    def test_threshold_prediction_uses_faint_probability(self):
        self.assertEqual(threshold_prediction({"Normal": 0.7, "Faint": 0.3}, 0.4), ("Normal", 0.7))
        self.assertEqual(threshold_prediction({"Normal": 0.55, "Faint": 0.45}, 0.4), ("Faint", 0.45))

    def test_lstm_classifier_reports_missing_checkpoint_before_torch_import(self):
        missing_checkpoint = Path("missing-checkpoint.pt")

        with self.assertRaises(FileNotFoundError) as ctx:
            LSTMActionClassifier(missing_checkpoint)

        self.assertIn("LSTM action checkpoint not found", str(ctx.exception))

    def test_loads_ai_fall_metadata_csv(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metadata.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_path", "label", "split"])
                writer.writeheader()
                writer.writerow({"clip_path": "clip_a.mp4", "label": "1", "split": "train"})

            rows = load_training_rows(csv_path, split="train")

        self.assertEqual(rows[0]["video_path"], "clip_a.mp4")
        self.assertEqual(rows[0]["label"], 1)

    def test_event_frame_range_is_preferred(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "label.json"
            label_path.write_text(
                json.dumps(
                    {
                        "metadata": {"file_name": "clip.mp4", "frame_count": 100},
                        "annotations": {"event_class": "Faint", "event_frame": [[20, 40]]},
                    }
                ),
                encoding="utf-8",
            )
            row = {"label_path": str(label_path), "label": 1, "start_frame": 0, "end_frame": 99}

            ranges = target_ranges_for_row(row)

        self.assertEqual(ranges, [(20, 40, True)])

    def test_preprocess_summary_reports_fallback_and_zero_sequence(self):
        metadata = [
            {
                "crop_source": "fallback_full_frame",
                "used_event_frame": True,
            }
        ]
        clips = [
            {"sequences_generated": 1, "skipped_frames_no_person": 2},
            {"sequences_generated": 0, "skipped_frames_no_person": 3},
        ]

        summary = summarize_metadata(metadata, clips)

        self.assertEqual(summary["total_sequences_generated"], 1)
        self.assertEqual(summary["sequences_from_event_frame_ranges"], 1)
        self.assertEqual(summary["sequences_using_fallback_full_frame_crops"], 1)
        self.assertEqual(summary["zero_sequence_clips"], 1)
        self.assertEqual(summary["skipped_frames_due_to_no_person"], 5)


if __name__ == "__main__":
    unittest.main()
