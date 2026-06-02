import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from benchmark.compare_lstm_extractors import (
    CUDA_CPU_FALLBACK_WARNING,
    CpuFallbackDisabledError,
    choose_best_model,
    classification_metrics,
    dataset_class_counts,
    keypoints_to_feature,
    limit_rows_by_split_and_class,
    lstm_readiness_status,
    normalize_torch_device,
    parse_frame_range,
    parse_model_specs,
    prediction_audit_rows,
    prediction_counts,
    sequence_class_counts,
    summarize_split,
    threshold_audit_metrics,
    write_final_summary,
    zero_sequence_reason,
)


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    def __init__(self, cuda_available):
        self.cuda = FakeCuda(cuda_available)


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

    def test_prediction_audit_rows_include_labels_and_probabilities(self):
        rows = prediction_audit_rows(
            y_true=[0, 1],
            probabilities=[[0.8, 0.2], [0.4, 0.6]],
            sequence_rows=[
                {"clip_id": "n0", "frame_start": 0, "frame_end": 15},
                {"clip_id": "f0", "frame_start": 100, "frame_end": 115},
            ],
        )

        self.assertEqual(rows[0]["true_label"], "Normal")
        self.assertEqual(rows[0]["pred_label"], "Normal")
        self.assertEqual(rows[0]["normal_prob"], 0.8)
        self.assertEqual(rows[0]["faint_prob"], 0.2)
        self.assertEqual(rows[1]["clip_id"], "f0")
        self.assertEqual(rows[1]["pred_label"], "Faint")

    def test_threshold_audit_metrics_reports_faint_recall_and_f1(self):
        audit = threshold_audit_metrics(
            y_true=[0, 0, 1, 1],
            faint_probs=[0.2, 0.45, 0.35, 0.65],
            thresholds=[0.3, 0.5],
        )

        self.assertEqual(audit[0]["threshold"], 0.3)
        self.assertEqual(audit[0]["faint_recall"], 1.0)
        self.assertEqual(audit[0]["f1_score"], 0.8)
        self.assertEqual(audit[1]["threshold"], 0.5)
        self.assertEqual(audit[1]["faint_recall"], 0.5)
        self.assertEqual(audit[1]["f1_score"], 0.666667)

    def test_prediction_counts_reports_predicted_normal_and_faint(self):
        self.assertEqual(prediction_counts([0, 1, 1, 0, 1]), {"Normal": 2, "Faint": 3})

    def test_normalize_torch_device_converts_numeric_gpu_id_to_cuda_device(self):
        self.assertEqual(normalize_torch_device("0", FakeTorch(True)), "cuda:0")
        self.assertEqual(normalize_torch_device(0, FakeTorch(True)), "cuda:0")
        self.assertEqual(normalize_torch_device("cuda:0", FakeTorch(True)), "cuda:0")
        self.assertEqual(normalize_torch_device("cpu", FakeTorch(True)), "cpu")

    def test_normalize_torch_device_falls_back_to_cpu_when_cuda_unavailable(self):
        self.assertEqual(normalize_torch_device("0", FakeTorch(False)), "cpu")
        self.assertEqual(normalize_torch_device("cuda:0", FakeTorch(False)), "cpu")
        self.assertEqual(normalize_torch_device("auto", FakeTorch(False)), "cpu")

    def test_normalize_torch_device_can_disable_cpu_fallback_for_requested_cuda(self):
        with self.assertRaises(CpuFallbackDisabledError):
            normalize_torch_device("0", FakeTorch(False), no_cpu_fallback=True)
        with self.assertRaises(CpuFallbackDisabledError):
            normalize_torch_device("cuda:0", FakeTorch(False), no_cpu_fallback=True)

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
            requested_class_counts={"Normal": 3, "Faint": 3, "total": 6},
        )

        self.assertEqual(summary["sequence_class_counts"], {"Normal": 1, "Faint": 2})
        self.assertEqual(summary["requested_class_counts"], {"Normal": 3, "Faint": 3, "total": 6})
        self.assertEqual(summary["keypoint_missing_rate"], round(3 / 51, 6))

    def test_lstm_readiness_status_reports_specific_sequence_and_class_gaps(self):
        self.assertEqual(lstm_readiness_status([], [0, 1]), "no_train_sequences")
        self.assertEqual(lstm_readiness_status([0, 1], []), "no_eval_sequences")
        self.assertEqual(lstm_readiness_status([0], [0, 1]), "insufficient_sequences")
        self.assertEqual(lstm_readiness_status([0, 0], [0, 1]), "missing_class_in_train")
        self.assertEqual(lstm_readiness_status([0, 1], [1, 1]), "missing_class_in_eval")
        self.assertEqual(lstm_readiness_status([0, 1], [0, 1]), "OK")

    def test_sequence_class_counts_reports_normal_and_faint_sequences(self):
        self.assertEqual(sequence_class_counts([0, 1, 1]), {"Normal": 1, "Faint": 2})

    def test_limit_rows_by_split_and_class_caps_each_class_independently(self):
        rows = []
        for split in ("train", "val", "test"):
            rows.extend({"split": split, "label": "0", "clip_id": f"{split}-normal-{idx}"} for idx in range(5))
            rows.extend({"split": split, "label": "1", "clip_id": f"{split}-faint-{idx}"} for idx in range(4))

        limited = limit_rows_by_split_and_class(rows, 3)
        counts = dataset_class_counts(limited)

        self.assertEqual(len(limited), 18)
        self.assertEqual(counts["train"], {"Normal": 3, "Faint": 3, "total": 6})
        self.assertEqual(counts["val"], {"Normal": 3, "Faint": 3, "total": 6})
        self.assertEqual(counts["test"], {"Normal": 3, "Faint": 3, "total": 6})

    def test_limit_rows_by_split_and_class_prefers_normals_near_faint_context(self):
        rows = [
            {"split": "train", "label": "0", "_resolved_video_path": "cam01__000000_000031.mp4"},
            {"split": "train", "label": "0", "_resolved_video_path": "cam01__000008_000039.mp4"},
            {"split": "train", "label": "0", "_resolved_video_path": "cam01__004780_004811.mp4"},
            {"split": "train", "label": "0", "_resolved_video_path": "cam01__004856_004887.mp4"},
            {"split": "train", "label": "1", "_resolved_video_path": "cam01__004816_004847.mp4"},
        ]

        limited = limit_rows_by_split_and_class(rows, 2, seed=42)
        normal_paths = [row["_resolved_video_path"] for row in limited if row["label"] == "0"]

        self.assertEqual(normal_paths, ["cam01__004780_004811.mp4", "cam01__004856_004887.mp4"])

    def test_limit_rows_by_split_and_class_keeps_available_minority_rows(self):
        rows = [
            {"split": "train", "label": "0", "clip_id": "n0"},
            {"split": "train", "label": "0", "clip_id": "n1"},
            {"split": "train", "label": "0", "clip_id": "n2"},
            {"split": "train", "label": "1", "clip_id": "f0"},
        ]

        limited = limit_rows_by_split_and_class(rows, 2)

        self.assertEqual(dataset_class_counts(limited)["train"], {"Normal": 2, "Faint": 1, "total": 3})

    def test_parse_frame_range_reads_processed_clip_suffix(self):
        self.assertEqual(parse_frame_range("100-1_cam01__004816_004847.mp4"), (4816, 4847))
        self.assertEqual(parse_frame_range("no_range.mp4"), (None, None))

    def test_zero_sequence_reason_distinguishes_no_person_and_no_keypoints(self):
        self.assertEqual(zero_sequence_reason({"person_detections": 0, "keypoints_extracted": 0}), "no_person_detections")
        self.assertEqual(zero_sequence_reason({"person_detections": 2, "keypoints_extracted": 0}), "no_keypoints_extracted")
        self.assertEqual(zero_sequence_reason({"person_detections": 2, "keypoints_extracted": 2}), "no_complete_sequence_window")

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
                "lstm_metrics": {
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1_score": 0.5,
                    "prediction_counts": {"Normal": 3, "Faint": 2},
                    "threshold_audit": [
                        {"threshold": 0.3, "faint_recall": 0.75, "f1_score": 0.6},
                        {"threshold": 0.5, "faint_recall": 0.5, "f1_score": 0.5},
                    ],
                    "repeated_seed_audit": {
                        "enabled": True,
                        "seeds": [42, 43, 44],
                        "faint_recall_mean": 0.6,
                        "faint_recall_std": 0.1,
                        "f1_score_mean": 0.55,
                        "f1_score_std": 0.05,
                    },
                    "status": "OK",
                    "train_sequence_class_counts": {"Normal": 2, "Faint": 1},
                    "eval_sequence_class_counts": {"Normal": 1, "Faint": 1},
                    "warnings": [CUDA_CPU_FALLBACK_WARNING],
                    "torch_device": "cpu",
                },
                "runtime_seconds": 1.25,
            }
        ]
        selected_class_counts = {"train": {"Normal": 30, "Faint": 30, "total": 60}, "val": {"Normal": 30, "Faint": 30, "total": 60}, "test": {"Normal": 30, "Faint": 30, "total": 60}}
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            write_final_summary(output_dir, summaries, selected_class_counts)

            report = (output_dir / "report.md").read_text(encoding="utf-8")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("## Pose-Only Benchmark", report)
            self.assertIn("## Warnings", report)
            self.assertIn(CUDA_CPU_FALLBACK_WARNING, report)
            self.assertIn("## Selected Dataset Class Counts", report)
            self.assertIn("| train | 30 | 30 | 60 |", report)
            self.assertIn("| val | 30 | 30 | 60 |", report)
            self.assertIn("| test | 30 | 30 | 60 |", report)
            self.assertIn("## Sequence Generation Benchmark", report)
            self.assertIn("## LSTM Classification Benchmark", report)
            self.assertIn("train Normal", report)
            self.assertIn("| YOLOv11n-pose | 2 | 1 | 1 | 1 | 0.5 | 0.5 | 0.5 | 0.5 | OK |", report)
            self.assertIn("## Prediction Distribution Audit", report)
            self.assertIn("| YOLOv11n-pose | 3 | 2 |", report)
            self.assertIn("## Threshold Audit", report)
            self.assertIn("| YOLOv11n-pose | 0.3 | 0.75 | 0.6 |", report)
            self.assertIn("## Repeated Seed Audit", report)
            self.assertIn("| YOLOv11n-pose | 42,43,44 | 0.6 | 0.1 | 0.55 | 0.05 |", report)
            self.assertIn("YOLOv11n-pose", report)
            self.assertEqual(summary["warnings"], [CUDA_CPU_FALLBACK_WARNING])
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
