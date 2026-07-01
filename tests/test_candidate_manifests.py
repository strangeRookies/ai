import csv
import json
import tempfile
import unittest
from pathlib import Path

from ai.learning.candidate_manifests import (
    CANDIDATE_FIELDS,
    REQUIRED_TRAINING_MANIFEST_V2_FIELDS,
    build_error_candidates,
    build_synthetic_candidates,
    build_training_manifest,
)


class CandidateManifestTest(unittest.TestCase):
    def test_hard_negative_schema_and_duplicate_evidence_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp_csv = root / "false_positives.csv"
            output = root / "hard_negative_candidates.csv"
            self._write_csv(
                fp_csv,
                [
                    {
                        "error_type": "FP",
                        "clip_id": "normal_fp",
                        "clip_path": "clips/normal_fp.mp4",
                        "label": "Normal",
                        "prediction": "Faint",
                        "faint_prob": "0.81",
                        "evidence_id": "cam-a-10-1000",
                    },
                    {
                        "error_type": "FP",
                        "clip_id": "normal_fp_dup",
                        "clip_path": "clips/normal_fp_dup.mp4",
                        "label": "Normal",
                        "prediction": "Faint",
                        "faint_prob": "0.82",
                        "evidence_id": "cam-a-10-1000",
                    },
                ],
            )

            summary = build_error_candidates(fp_csv, output, "hard_negative")

            rows = self._read_csv(output)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(list(rows[0].keys()), CANDIDATE_FIELDS)
            self.assertEqual(rows[0]["label"], "0")
            self.assertEqual(rows[0]["label_name"], "Normal")
            self.assertEqual(rows[0]["source_type"], "hard_negative")
            self.assertEqual(rows[0]["failure_type"], "false_positive")
            self.assertEqual(rows[0]["scenario_tag"], "normal_false_positive")
            self.assertEqual(rows[0]["review_status"], "pending")

    def test_candidate_export_preserves_parent_split_context_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp_csv = root / "false_positives.csv"
            metadata = root / "metadata.csv"
            output = root / "hard_negative_candidates.csv"
            self._write_csv(fp_csv, [{"error_type": "FP", "clip_id": "test_parent", "prediction": "Faint"}])
            self._write_csv(
                metadata,
                [
                    {
                        "clip_id": "test_parent",
                        "clip_path": "clips/test_parent.mp4",
                        "source_video": "source/test.mp4",
                        "split": "test",
                        "split_group_id": "source-test",
                    }
                ],
            )

            build_error_candidates(fp_csv, output, "hard_negative", metadata_csv=metadata)

            rows = self._read_csv(output)
            self.assertEqual(rows[0]["split"], "test")
            self.assertEqual(rows[0]["source_video"], "source/test.mp4")
            self.assertEqual(rows[0]["split_group_id"], "source-test")

    def test_faint_reinforcement_uses_faint_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fn_csv = root / "false_negatives.csv"
            output = root / "faint_reinforcement_candidates.csv"
            self._write_csv(
                fn_csv,
                [
                    {
                        "error_type": "FN",
                        "clip_id": "faint_missed",
                        "clip_path": "clips/faint_missed.mp4",
                        "label": "Faint",
                        "prediction": "Normal",
                        "faint_prob": "0.22",
                        "reason": "night_false_negative",
                    }
                ],
            )

            build_error_candidates(fn_csv, output, "faint_reinforcement")

            rows = self._read_csv(output)
            self.assertEqual(rows[0]["label"], "1")
            self.assertEqual(rows[0]["label_name"], "Faint")
            self.assertEqual(rows[0]["source_type"], "faint_reinforcement")
            self.assertEqual(rows[0]["failure_type"], "false_negative")
            self.assertEqual(rows[0]["scenario_tag"], "night_false_negative")

    def test_synthetic_metadata_keeps_parent_clip_id_seed_config_and_review_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "candidates.csv"
            output = root / "synthetic_candidates.csv"
            self._write_csv(
                input_csv,
                [{"clip_id": "faint_01", "clip_path": "clips/faint_01.mp4", "label": "1", "label_name": "Faint", "split": "train"}],
            )

            summary = build_synthetic_candidates(input_csv, output, synthetic_types=("brightness",), seed=7)

            rows = self._read_csv(output)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(rows[0]["source_type"], "synthetic")
            self.assertEqual(rows[0]["augmentation_type"], "brightness")
            self.assertEqual(rows[0]["synthetic_type"], "brightness")
            self.assertEqual(rows[0]["parent_clip_id"], "faint_01")
            self.assertEqual(rows[0]["parent_clip_path"], "clips/faint_01.mp4")
            self.assertEqual(rows[0]["parent_split"], "train")
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[0]["review_status"], "pending")
            self.assertEqual(rows[0]["random_seed"], "7")
            self.assertEqual(json.loads(rows[0]["augmentation_config"])["seed"], 7)

    def test_low_visibility_synthetic_is_marked_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "candidates.csv"
            output = root / "synthetic_candidates.csv"
            self._write_csv(
                input_csv,
                [{"clip_id": "faint_occ", "clip_path": "clips/faint_occ.mp4", "label": "1", "label_name": "Faint"}],
            )

            build_synthetic_candidates(input_csv, output, synthetic_types=("partial_occlusion",), min_visibility_for_approved=0.6)

            rows = self._read_csv(output)
            self.assertEqual(rows[0]["review_status"], "pending")
            self.assertEqual(rows[0]["estimated_visibility"], "0.55")

    def test_training_manifest_v2_includes_only_approved_candidates_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "metadata.csv"
            hard = root / "hard_negative_candidates.csv"
            output = root / "training_manifest_v2.csv"
            self._write_csv(base, [{"clip_id": "real_faint", "clip_path": "clips/real_faint.mp4", "label": "1", "label_name": "Faint"}])
            self._write_csv(
                hard,
                [
                    {"clip_id": "hard_01", "clip_path": "clips/hard_01.mp4", "label": "0", "label_name": "Normal", "source_type": "hard_negative", "review_status": "approved"},
                    {"clip_id": "hard_02", "clip_path": "clips/hard_02.mp4", "label": "0", "label_name": "Normal", "source_type": "hard_negative", "review_status": "pending"},
                ],
            )

            summary = build_training_manifest(base, output, [hard])

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["class_counts"], {"Faint": 1, "Normal": 1})
            self.assertEqual(summary["source_type_counts"], {"hard_negative": 1, "real": 1})

            rows = self._read_csv(output)
            self.assertEqual(list(rows[0].keys())[: len(REQUIRED_TRAINING_MANIFEST_V2_FIELDS)], REQUIRED_TRAINING_MANIFEST_V2_FIELDS)
            self.assertNotIn("hard_02", {row["clip_id"] for row in rows})

    def test_training_manifest_skips_synthetic_from_test_parent_and_caps_train_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "metadata.csv"
            synthetic = root / "synthetic_candidates.csv"
            output = root / "training_manifest_v2.csv"
            self._write_csv(
                base,
                [
                    {"clip_id": f"real_{index}", "clip_path": f"clips/real_{index}.mp4", "label": "0", "label_name": "Normal", "split": "train"}
                    for index in range(7)
                ],
            )
            self._write_csv(
                synthetic,
                [
                    {
                        "clip_id": f"synthetic_{index}",
                        "clip_path": f"data/synthetic/brightness_down/synthetic_{index}.mp4",
                        "label": "1",
                        "label_name": "Faint",
                        "source_type": "synthetic",
                        "review_status": "approved",
                        "parent_clip_id": f"real_{index}",
                        "split_group_id": f"real_{index}",
                        "parent_split": "train" if index < 4 else "test",
                        "split": "train" if index < 4 else "test",
                    }
                    for index in range(5)
                ],
            )

            summary = build_training_manifest(base, output, [synthetic], max_synthetic_ratio=0.3)

            rows = self._read_csv(output)
            synthetic_rows = [row for row in rows if row["source_type"] == "synthetic"]
            self.assertEqual(len(synthetic_rows), 3)
            self.assertNotIn("test", {row["parent_split"] for row in synthetic_rows})
            self.assertEqual(summary["source_type_counts"]["synthetic"], 3)
            self.assertLessEqual(len(synthetic_rows) / len(rows), 0.3)

    def test_dry_run_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "metadata.csv"
            output = root / "training_manifest_v2.csv"
            self._write_csv(base, [{"clip_id": "real_normal", "clip_path": "clips/real_normal.mp4", "label": "0", "label_name": "Normal"}])

            summary = build_training_manifest(base, output, [], dry_run=True)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".summary.json").exists())
            self.assertFalse(summary["written"])

    def test_training_manifest_refuses_to_overwrite_base_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "metadata.csv"
            self._write_csv(base, [{"clip_id": "real_normal", "clip_path": "clips/real_normal.mp4", "label": "0", "label_name": "Normal"}])

            with self.assertRaises(RuntimeError):
                build_training_manifest(base, base, [])

    def _write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path):
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            return list(csv.DictReader(fp))


if __name__ == "__main__":
    unittest.main()
