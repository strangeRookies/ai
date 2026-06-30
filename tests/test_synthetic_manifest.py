import csv
import tempfile
import unittest
from pathlib import Path

from ai.learning.synthetic_manifest import build_synthetic_candidate_manifest


class SyntheticManifestTest(unittest.TestCase):
    def test_build_synthetic_manifest_adds_source_type_reason_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            output = root / "synthetic_candidates.csv"
            with metadata.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["clip_id", "clip_path", "label_name", "domain"])
                writer.writeheader()
                writer.writerow(
                    {
                        "clip_id": "clip-night",
                        "clip_path": "clips/night.mp4",
                        "label_name": "Faint",
                        "domain": "night_far",
                    }
                )

            summary = build_synthetic_candidate_manifest(metadata, output)

            self.assertEqual(summary["input_rows"], 1)
            self.assertEqual(summary["candidate_rows"], 6)
            with output.open(encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(rows[0]["source_type"], "synthetic_candidate")
            self.assertIn(rows[0]["synthetic_type"], {"brightness", "noise", "blur", "crop", "occlusion", "distance"})
            self.assertIn("weak-condition", rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()
