import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_retraining_manifest_v2 import (
    filter_approved_rows,
    segment_by_tag,
    calculate_metrics_from_preds,
)


class EvaluateRetrainingTest(unittest.TestCase):
    def test_filter_approved_rows_excludes_pending_or_rejected(self):
        rows = [
            {"clip_id": "real_1", "source_type": "real", "review_status": ""},
            {"clip_id": "hard_1", "source_type": "hard_negative", "review_status": "approved"},
            {"clip_id": "hard_2", "source_type": "hard_negative", "review_status": "pending"},
            {"clip_id": "synthetic_1", "source_type": "synthetic", "review_status": "rejected"},
            {"clip_id": "faint_1", "source_type": "faint_reinforcement", "review_status": "needs_review"},
        ]

        filtered = filter_approved_rows(rows)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["clip_id"], "real_1")
        self.assertEqual(filtered[1]["clip_id"], "hard_1")

    def test_segment_by_tag_groups_fps_and_fns(self):
        seq_metadata = [
            {"scenario_tag": "tag1", "augmentation_type": "noise"},
            {"scenario_tag": "tag1", "augmentation_type": "blur"},
            {"scenario_tag": "tag2", "augmentation_type": "noise"},
        ]
        test_y = [0, 1, 0]
        preds = [1, 0, 0]  # idx 0: FP, idx 1: FN, idx 2: TN
        probs = [0.8, 0.2, 0.1]

        tag_stats, aug_stats = segment_by_tag(seq_metadata, test_y, preds, probs)

        self.assertEqual(tag_stats["tag1"]["FP"], 1)
        self.assertEqual(tag_stats["tag1"]["FN"], 1)
        self.assertEqual(tag_stats["tag2"]["FP"], 0)
        self.assertEqual(tag_stats["tag2"]["FN"], 0)

        self.assertEqual(aug_stats["noise"]["FP"], 1)
        self.assertEqual(aug_stats["blur"]["FN"], 1)

    def test_calculate_metrics_handles_thresholds(self):
        test_y = [0, 1, 1, 0]
        preds = [0, 1, 1, 0]
        faint_probs = [0.1, 0.9, 0.8, 0.2]
        tag_stats = {}
        aug_stats = {}

        metrics = calculate_metrics_from_preds(test_y, preds, faint_probs, tag_stats, aug_stats)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertIn(0.5, metrics["thresholds"])
        self.assertEqual(metrics["thresholds"][0.5]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
