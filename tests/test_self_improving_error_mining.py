import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.learning.self_improving_error_mining import (
    append_jsonl,
    mine_prediction_rows,
    quarantine_reason_counts,
)
from ai.learning.self_improving_samples import sample_prediction_rows
from ai.learning.self_improving_synthetic import build_synthetic_preview


class SelfImprovingErrorMiningTest(unittest.TestCase):
    def test_fp_fn_rows_are_routed_to_appendable_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = sample_prediction_rows()

            result = mine_prediction_rows(rows, root)

            hard_rows = read_jsonl(root / "hard_negative_candidates.jsonl")
            reinforce_rows = read_jsonl(root / "faint_fall_reinforcement_candidates.jsonl")
            self.assertEqual(result["accepted_counts"], {"hard_negative": 1, "faint_fall_reinforcement": 1})
            self.assertEqual(hard_rows[0]["candidate_type"], "hard_negative")
            self.assertEqual(hard_rows[0]["prediction_label"], "Faint")
            self.assertEqual(hard_rows[0]["ground_truth_label"], "Normal")
            self.assertEqual(hard_rows[0]["feature_schema"], "keypoint_bbox54")
            self.assertEqual(hard_rows[0]["feature_dim"], 54)
            self.assertEqual(hard_rows[0]["bbox_features"], [0.25, 0.5, 0.125])
            self.assertEqual(reinforce_rows[0]["candidate_type"], "faint_fall_reinforcement")

            append_jsonl(root / "hard_negative_candidates.jsonl", hard_rows[0])
            self.assertEqual(len(read_jsonl(root / "hard_negative_candidates.jsonl")), 2)

    def test_invalid_label_or_missing_bbox_goes_to_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = sample_prediction_rows(include_invalid=True)

            result = mine_prediction_rows(rows, root)

            quarantine_rows = read_jsonl(root / "quarantine.jsonl")
            reasons = quarantine_reason_counts(quarantine_rows)
            self.assertEqual(result["quarantine_count"], 2)
            self.assertEqual(reasons["missing_bbox_feature"], 1)
            self.assertEqual(reasons["missing_label_interval"], 1)

    def test_missing_frame_metadata_goes_to_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = sample_prediction_rows()[0]
            row.pop("frameId")

            result = mine_prediction_rows([row], root)

            quarantine_rows = read_jsonl(root / "quarantine.jsonl")
            self.assertEqual(result["quarantine_count"], 1)
            self.assertEqual(quarantine_rows[0]["reason"], "missing_frame_id")

    def test_non_numeric_frame_metadata_goes_to_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = sample_prediction_rows()[0]
            row["frameId"] = "not-a-number"

            result = mine_prediction_rows([row], root)

            quarantine_rows = read_jsonl(root / "quarantine.jsonl")
            self.assertEqual(result["quarantine_count"], 1)
            self.assertEqual(quarantine_rows[0]["reason"], "missing_frame_id")

    def test_malformed_bbox_goes_to_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = sample_prediction_rows()[0]
            sequence = row["sequence"]
            sequence["detections"][0]["bbox"] = [1.0, 2.0, 3.0]

            result = mine_prediction_rows([row], root)

            quarantine_rows = read_jsonl(root / "quarantine.jsonl")
            self.assertEqual(result["quarantine_count"], 1)
            self.assertEqual(quarantine_rows[0]["reason"], "malformed_bbox")

    def test_synthetic_preview_keeps_feature_dim_54(self):
        feature = np.zeros((3, 54), dtype=np.float32)
        feature[:, 51:] = np.asarray([0.25, 0.5, 0.125], dtype=np.float32)
        candidate = {
            "candidate_id": "cand-1",
            "clip_id": "clip-1",
            "feature_dim": 54,
            "feature_schema": "keypoint_bbox54",
            "feature_sequence": feature.tolist(),
        }

        previews = build_synthetic_preview([candidate], seed=7)

        self.assertEqual(
            {row["augmentation_type"] for row in previews},
            {
                "keypoint_noise",
                "bbox_scale_jitter",
                "bbox_aspect_jitter",
                "confidence_drop",
                "frame_drop",
                "temporal_jitter",
                "horizontal_flip",
            },
        )
        self.assertTrue(all(row["feature_dim"] == 54 for row in previews))
        self.assertTrue(all(row["auto_merge_to_train"] is False for row in previews))

    def test_synthetic_preview_rejects_non_54_dim_candidates(self):
        candidate = {
            "candidate_id": "bad",
            "clip_id": "clip-bad",
            "feature_dim": 51,
            "feature_schema": "keypoint51",
            "feature_sequence": [[0.0] * 51],
        }

        previews = build_synthetic_preview([candidate], seed=7)

        self.assertEqual(previews, [])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
