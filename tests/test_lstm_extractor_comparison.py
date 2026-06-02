import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from benchmark.compare_lstm_extractors import (
    choose_best_model,
    classification_metrics,
    keypoints_to_feature,
    parse_model_specs,
    summarize_split,
    write_final_summary,
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
        self.assertEqual(metrics["per_class_metrics"]["Normal"]["recall"], 0.5)
        self.assertEqual(metrics["per_class_metrics"]["Faint"]["recall"], 0.5)

    def test_choose_best_model_prioritizes_faint_recall_then_sequences(self):
        summaries = [
            {
                "model_label": "YOLO26n-pose",
                "eval_sequence_summary": {"generated_sequences": 10, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.8, "f1_score": 0.7, "confusion_matrix": {"matrix": [[9, 1], [2, 8]]}},
            },
            {
                "model_label": "YOLOv11n-pose",
                "eval_sequence_summary": {"generated_sequences": 50, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.7, "f1_score": 0.9, "confusion_matrix": {"matrix": [[9, 1], [3, 7]]}},
            },
        ]

        self.assertEqual(choose_best_model(summaries), "YOLO26n-pose")

    def test_choose_best_model_uses_f1_before_sequence_count_when_recall_ties(self):
        summaries = [
            {
                "model_label": "YOLO26n-pose",
                "eval_sequence_summary": {"generated_sequences": 10, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.8, "f1_score": 0.9, "confusion_matrix": {"matrix": [[9, 1], [2, 8]]}},
            },
            {
                "model_label": "YOLOv11n-pose",
                "eval_sequence_summary": {"generated_sequences": 50, "zero_sequence_clips": 0, "keypoint_missing_rate": 0.1},
                "lstm_metrics": {"recall": 0.8, "f1_score": 0.7, "confusion_matrix": {"matrix": [[9, 1], [2, 8]]}},
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

    def test_write_final_summary_creates_markdown_report_with_required_sections(self):
        summaries = [
            {
                "model_label": "YOLOv11n-pose",
                "pose_model": "yolo11n-pose.pt",
                "eval_sequence_summary": {
                    "clips_requested": 1,
                    "clips_processed": 1,
                    "person_detections": 4,
                    "keypoints_extracted": 4,
                    "generated_sequences": 2,
                    "zero_sequence_clips": 0,
                    "keypoint_missing_rate": 0.1,
                    "fallback_usage": 0,
                },
                "lstm_metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5, "status": "OK"},
                "runtime_seconds": 1.25,
            }
        ]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            write_final_summary(output_dir, summaries)

            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Pose-Only Benchmark", report)
            self.assertIn("## Sequence Generation Benchmark", report)
            self.assertIn("## LSTM Classification Benchmark", report)
            self.assertIn("YOLOv11n-pose", report)
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
