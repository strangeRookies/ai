import csv
import tempfile
import unittest
from pathlib import Path

from scripts.check_dataset_split import leakage_report, stratified_group_split, summarize


class DatasetSplitCheckTest(unittest.TestCase):
    def test_reports_split_counts_and_source_leakage(self):
        rows = [
            {"clip_id": "a1", "source_video": "a.mp4", "label": "1", "label_name": "Faint", "split": "train"},
            {"clip_id": "a2", "source_video": "a.mp4", "label": "1", "label_name": "Faint", "split": "test"},
            {"clip_id": "b1", "source_video": "b.mp4", "label": "0", "label_name": "Normal", "split": "val"},
        ]

        summary = summarize(rows)
        leakage = leakage_report(rows)

        self.assertEqual(summary[0]["split"], "train")
        self.assertEqual(summary[0]["Faint"], 1)
        self.assertIn("a.mp4", leakage)

    def test_stratified_group_split_keeps_source_together(self):
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "clip_id": f"clip_{idx}",
                    "source_video": f"source_{idx // 2}.mp4",
                    "label": "1" if idx % 3 == 0 else "0",
                    "label_name": "Faint" if idx % 3 == 0 else "Normal",
                }
            )

        fixed = stratified_group_split(rows, seed=1)
        leakage = leakage_report(fixed)

        self.assertEqual(leakage, {})
        self.assertTrue(all(row.get("split") in {"train", "test", "val"} for row in fixed))


if __name__ == "__main__":
    unittest.main()
