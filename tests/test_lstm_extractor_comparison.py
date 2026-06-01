import unittest

import numpy as np

from benchmark.compare_lstm_extractors import (
    choose_best_model,
    classification_metrics,
    keypoints_to_feature,
    parse_model_specs,
    summarize_split,
)


class LstmExtractorComparisonTest(unittest.TestCase):
    def test_keypoints_to_feature_normalizes_xy_and_counts_missing(self):
        detection = {
            "keypoints": [
                {"x": 50.0, "y": 25.0, "confidence": 0.9},
                {"x": 20.0, "y": 10.0, "confidence": 0.1},
            ]
        }

        features, missing, total = keypoints_to_feature(detection, (100, 200, 3), 0.3)

        self.assertEqual(features.shape, (51,))
        self.assertAlmostEqual(float(features[0]), 0.25)
        self.assertAlmostEqual(float(features[1]), 0.25)
        self.assertEqual(missing, 16)
        self.assertEqual(total, 17)

    def test_classification_metrics_treats_faint_as_positive_class(self):
        metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 1])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 1], [1, 1]])

    def test_choose_best_model_prioritizes_faint_recall_then_sequences(self):
        summaries = [
            {
                "model_label": "YOLO26n-pose",
                "eval_sequence_summary": {"generated_sequences": 10, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.8, "f1_score": 0.7},
            },
            {
                "model_label": "YOLOv11n-pose",
                "eval_sequence_summary": {"generated_sequences": 50, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.7, "f1_score": 0.9},
            },
        ]

        self.assertEqual(choose_best_model(summaries), "YOLO26n-pose")

    def test_summarize_split_reports_requested_counts(self):
        summary = summarize_split(
            rows=[],
            y_rows=[0, 1, 1],
            totals={
                "clips_requested": 2,
                "clips_processed": 2,
                "person_detections": 12,
                "keypoints_extracted": 10,
                "generated_sequences": 3,
                "zero_sequence_clips": 0,
                "fallback_usage": 0,
                "missing_keypoints": 3,
                "total_keypoints": 51,
            },
        )

        self.assertEqual(summary["sequence_class_counts"], {"Normal": 1, "Faint": 2})
        self.assertEqual(summary["keypoint_missing_rate"], round(3 / 51, 6))

    def test_parse_model_specs_accepts_label_model_pairs(self):
        specs = parse_model_specs("A:a.pt,B:b.pt")

        self.assertEqual(specs, [{"label": "A", "model": "a.pt"}, {"label": "B", "model": "b.pt"}])


if __name__ == "__main__":
    unittest.main()
