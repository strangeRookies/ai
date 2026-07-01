import unittest
from scripts.evaluate_retraining_manifest_v2 import (
    filter_approved_rows,
    segment_by_tag,
    calculate_metrics_from_preds,
    select_and_balance_rows,
    verify_split_isolation,
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

    def test_select_and_balance_rows_per_class(self):
        rows = [
            {"clip_id": "c1", "label": "0", "label_name": "Normal"},
            {"clip_id": "c2", "label": "0", "label_name": "Normal"},
            {"clip_id": "c3", "label": "0", "label_name": "Normal"},
            {"clip_id": "c4", "label": "1", "label_name": "Faint"},
            {"clip_id": "c5", "label": "1", "label_name": "Faint"},
        ]

        # Limit per class = 2 (each class gets 2 rows, total = 4)
        selected = select_and_balance_rows(rows, limit=2, per_class=True, balance=True, seed=42, split_name="test")
        
        self.assertEqual(len(selected), 4)
        normals = [r for r in selected if r["label_name"] == "Normal"]
        faints = [r for r in selected if r["label_name"] == "Faint"]
        self.assertEqual(len(normals), 2)
        self.assertEqual(len(faints), 2)

    def test_select_and_balance_rows_total_limit(self):
        rows = [
            {"clip_id": "c1", "label": "0", "label_name": "Normal"},
            {"clip_id": "c2", "label": "0", "label_name": "Normal"},
            {"clip_id": "c3", "label": "1", "label_name": "Faint"},
            {"clip_id": "c4", "label": "1", "label_name": "Faint"},
        ]

        # Limit total = 2 (each class gets limit // 2 = 1 row, total = 2)
        selected = select_and_balance_rows(rows, limit=2, per_class=False, balance=True, seed=42, split_name="test")
        
        self.assertEqual(len(selected), 2)
        normals = [r for r in selected if r["label_name"] == "Normal"]
        faints = [r for r in selected if r["label_name"] == "Faint"]
        self.assertEqual(len(normals), 1)
        self.assertEqual(len(faints), 1)

    def test_verify_split_isolation_raises_on_leakage(self):
        train = [{"clip_id": "c1", "parent_clip_id": "parent_x", "split_group_id": "g1", "source_video": "v1"}]
        val = [{"clip_id": "c2", "parent_clip_id": "parent_y", "split_group_id": "g2", "source_video": "v2"}]
        test = [{"clip_id": "c3", "parent_clip_id": "parent_x", "split_group_id": "g3", "source_video": "v3"}]

        # train shares parent_x with test -> should raise RuntimeError
        with self.assertRaises(RuntimeError):
            verify_split_isolation(train, val, test)

    def test_verify_split_isolation_passes_on_no_leakage(self):
        train = [{"clip_id": "c1", "parent_clip_id": "parent_x", "split_group_id": "g1", "source_video": "v1"}]
        val = [{"clip_id": "c2", "parent_clip_id": "parent_y", "split_group_id": "g2", "source_video": "v2"}]
        test = [{"clip_id": "c3", "parent_clip_id": "parent_z", "split_group_id": "g3", "source_video": "v3"}]

        # No intersection of parent_clip_id, split_group_id, or source_video between train/val and test
        try:
            verify_split_isolation(train, val, test)
        except RuntimeError as e:
            self.fail(f"verify_split_isolation raised RuntimeError unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
