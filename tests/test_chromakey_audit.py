import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_chromakey_split import audit, classify_domain, is_chromakey


class ChromakeyAuditTest(unittest.TestCase):
    def test_classifies_domain_column_as_chromakey(self):
        row = {"domain": "indoor_chromakey", "source_video": "cam01.mp4"}

        self.assertTrue(is_chromakey(row))
        self.assertEqual(classify_domain(row), "indoor_chromakey")

    def test_classifies_path_keywords_when_domain_missing(self):
        row = {"video_path": "data/raw/indoor/chroma_green_screen_clip.mp4"}

        self.assertTrue(is_chromakey(row))
        self.assertEqual(classify_domain(row), "indoor_chromakey")

    def test_audit_writes_split_outputs_for_non_chromakey_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "all.csv"
            output_dir = root / "audit"
            write_rows(
                input_csv,
                [
                    {"split": "train", "label": "1", "label_name": "Faint", "domain": "indoor_chromakey", "source_video": "a.mp4"},
                    {"split": "test", "label": "1", "label_name": "Faint", "domain": "outdoor", "source_video": "b.mp4"},
                    {"split": "test", "label": "0", "label_name": "Normal", "domain": "indoor_background", "source_video": "c.mp4"},
                ],
            )

            summary = audit([input_csv], output_dir)
            with (output_dir / "test_non_chromakey.csv").open(encoding="utf-8", newline="") as fp:
                non_chromakey_test = list(csv.DictReader(fp))
            saved_summary = json.loads((output_dir / "chromakey_audit_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["by_eval_group"], {"chromakey": 1, "non_chromakey": 2})
        self.assertEqual(len(non_chromakey_test), 2)
        self.assertEqual(saved_summary["by_split"]["test"]["eval_groups"]["non_chromakey"], 2)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
