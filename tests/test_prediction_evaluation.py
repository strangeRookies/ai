import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai.evaluation.prediction_metrics import metrics, read_prediction_logs, threshold_sweep
from scripts.evaluate_prediction_logs import evaluate


class PredictionEvaluationTest(unittest.TestCase):
    def test_read_prediction_logs_treats_hard_negative_as_normal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "hard_negative" / "sample.jsonl",
                [{"prediction": "Faint", "confidence": 0.8, "event_emitted": True}],
            )

            rows = read_prediction_logs(root)

        self.assertEqual(rows[0]["ground_truth"], "Normal")
        self.assertEqual(metrics(rows)["confusion_matrix"]["FP"], 1)

    def test_rtsp_real_without_ground_truth_is_not_counted_as_labeled_metric(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "rtsp_real" / "sample.jsonl", [{"prediction": "Faint", "confidence": 0.8, "event_emitted": True}])

            rows = read_prediction_logs(root)

        self.assertIsNone(rows[0]["ground_truth"])
        self.assertEqual(metrics(rows)["total_labeled"], 0)

    def test_metrics_report_required_binary_counts(self):
        rows = [
            {"prediction": "Faint", "ground_truth": "Faint", "event_emitted": True},
            {"prediction": "Faint", "ground_truth": "Normal", "event_emitted": True},
            {"prediction": "Normal", "ground_truth": "Faint", "event_emitted": False},
            {"prediction": "Normal", "ground_truth": "Normal", "event_emitted": False},
        ]

        payload = metrics(rows)

        self.assertEqual(payload["confusion_matrix"], {"TP": 1, "FP": 1, "FN": 1, "TN": 1})
        self.assertEqual(payload["precision"], 0.5)
        self.assertEqual(payload["recall"], 0.5)
        self.assertEqual(payload["f1_score"], 0.5)
        self.assertEqual(payload["false_positive_count"], 1)
        self.assertEqual(payload["false_negative_count"], 1)

    def test_threshold_sweep_applies_keypoint_and_consecutive_filters(self):
        rows = [
            base_row("Faint", "Faint", 0.9, consecutive=2, missing=0.1, keypoint_conf=0.7, timestamp=1),
            base_row("Faint", "Normal", 0.9, consecutive=1, missing=0.1, keypoint_conf=0.7, timestamp=2),
            base_row("Faint", "Normal", 0.9, consecutive=2, missing=0.9, keypoint_conf=0.7, timestamp=3),
        ]

        results = threshold_sweep(
            rows,
            {
                "faint_confidence_threshold": [0.8],
                "consecutive_faint_count": [2],
                "event_cooldown_seconds": [0.0],
                "max_keypoint_missing_rate": [0.5],
                "min_avg_keypoint_conf": [0.5],
            },
        )

        self.assertEqual(results[0]["confusion_matrix"], {"TP": 1, "FP": 0, "FN": 0, "TN": 2})

    def test_evaluate_writes_summary_and_sweep_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "faint" / "sample.jsonl", [base_row("Faint", None, 0.9, event_emitted=True)])
            write_jsonl(root / "normal_basic" / "sample.jsonl", [base_row("Normal", None, 0.2, event_emitted=False)])
            output = root / "summary.json"
            sweep_csv = root / "sweep.csv"

            payload = evaluate(
                Namespace(
                    sample_root=str(root),
                    output=str(output),
                    sweep_csv=str(sweep_csv),
                    faint_confidence_thresholds="0.3,0.5",
                    consecutive_faint_counts="1",
                    event_cooldown_seconds="0",
                    max_keypoint_missing_rates="1.0",
                    min_avg_keypoint_confs="0.0",
                )
            )
            output_exists = output.exists()
            sweep_csv_exists = sweep_csv.exists()

        self.assertEqual(payload["metrics"]["confusion_matrix"]["TP"], 1)
        self.assertTrue(output_exists)
        self.assertTrue(sweep_csv_exists)


def base_row(
    prediction,
    ground_truth,
    confidence,
    consecutive=1,
    missing=0.0,
    keypoint_conf=0.9,
    timestamp=1,
    event_emitted=None,
):
    return {
        "source_id": "sample",
        "video_id": "sample",
        "camera_id": "cam_01",
        "frame_idx": int(timestamp),
        "timestamp": float(timestamp),
        "track_id": 1,
        "prediction": prediction,
        "confidence": confidence,
        "ground_truth": ground_truth,
        "keypoint_missing_rate": missing,
        "avg_keypoint_conf": keypoint_conf,
        "sequence_length": 8,
        "consecutive_faint_count": consecutive,
        "cooldown_active": False,
        "event_emitted": prediction == "Faint" if event_emitted is None else event_emitted,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
