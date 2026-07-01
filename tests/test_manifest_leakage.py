import csv
import tempfile
import unittest
from pathlib import Path

from ai.learning.candidate_manifests import check_manifest_leakage


class ManifestLeakageTest(unittest.TestCase):
    def test_manifest_leakage_check_rejects_parent_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "training_manifest_v2.csv"
            self._write_csv(
                manifest,
                [
                    self._row("parent_train", "real", "", "real-parent-split-group", "train"),
                    self._row("parent_val_aug", "synthetic", "parent_train", "synthetic-child-split-group", "val"),
                ],
            )

            with self.assertRaises(RuntimeError):
                check_manifest_leakage(manifest)

    def test_manifest_leakage_check_accepts_approved_synthetic_with_parent_and_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "training_manifest_v2.csv"
            rows = [self._row(f"real_{index}", "real", "", f"real_{index}", "train") for index in range(7)]
            rows.extend(self._row(f"synthetic_{index}", "synthetic", f"real_{index}", f"real_{index}", "train") for index in range(3))
            self._write_csv(manifest, rows)

            summary = check_manifest_leakage(manifest, max_synthetic_ratio=0.3)

            self.assertEqual(summary["rows"], 10)
            self.assertEqual(summary["synthetic_rows"], 3)
            self.assertEqual(summary["split_group_leaks"], 0)

    def _row(self, clip_id, source_type, parent_clip_id, split_group_id, split):
        return {
            "clip_id": clip_id,
            "clip_path": f"clips/{clip_id}.mp4",
            "label": "1",
            "label_name": "Faint",
            "source_type": source_type,
            "parent_clip_id": parent_clip_id,
            "review_status": "approved",
            "failure_type": "",
            "scenario_tag": "night_false_negative",
            "augmentation_type": "blur" if source_type == "synthetic" else "",
            "augmentation_config": "{}",
            "random_seed": "42" if source_type == "synthetic" else "",
            "split_group_id": split_group_id,
            "created_at": "2026-07-01T00:00:00Z",
            "split": split,
        }

    def _write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
