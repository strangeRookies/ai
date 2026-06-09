import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_lstm_thresholds import audit, metrics_at_threshold, recommend_threshold


class ThresholdAuditScriptTest(unittest.TestCase):
    def test_metrics_report_false_positive_and_false_negative_counts(self):
        rows = [
            {"true_label": "Normal", "faint_prob": "0.2"},
            {"true_label": "Normal", "faint_prob": "0.6"},
            {"true_label": "Faint", "faint_prob": "0.8"},
            {"true_label": "Faint", "faint_prob": "0.4"},
        ]

        metrics = metrics_at_threshold(rows, 0.5)

        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["faint_recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)

    def test_recommend_threshold_prioritizes_recall_then_f1_then_false_alarms(self):
        rows = [
            {"threshold": 0.3, "faint_recall": 1.0, "f1_score": 0.7, "false_positives": 4},
            {"threshold": 0.4, "faint_recall": 1.0, "f1_score": 0.8, "false_positives": 5},
            {"threshold": 0.5, "faint_recall": 0.9, "f1_score": 0.9, "false_positives": 0},
        ]

        self.assertEqual(recommend_threshold(rows), 0.4)

    def test_audit_writes_json_csv_and_markdown_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "eval_predictions.csv"
            output_dir = root / "audit"
            with predictions.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["true_label", "faint_prob"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"true_label": "Normal", "faint_prob": "0.2"},
                        {"true_label": "Normal", "faint_prob": "0.65"},
                        {"true_label": "Faint", "faint_prob": "0.8"},
                        {"true_label": "Faint", "faint_prob": "0.35"},
                    ]
                )

            payload = audit(predictions, output_dir)

            self.assertTrue((output_dir / "threshold_audit.csv").exists())
            self.assertTrue((output_dir / "threshold_audit.md").exists())
            saved = json.loads((output_dir / "threshold_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["recommended_threshold"], payload["recommended_threshold"])
            self.assertIn("Faint recall first", saved["selection_rule"])


if __name__ == "__main__":
    unittest.main()
