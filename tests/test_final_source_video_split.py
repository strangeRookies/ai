import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.create_final_source_video_split import create_split, leakage, read_rows


def write_metadata(path: Path) -> None:
    rows = []
    domains = ["indoor_background", "indoor_chromakey", "outdoor"]
    for index in range(30):
        source = f"source_{index:03d}.mp4"
        domain = domains[index % len(domains)]
        rows.append(
            {
                "clip_id": f"{source}:faint",
                "clip_path": f"clips/{source}/faint.mp4",
                "label": "1",
                "label_name": "Faint",
                "domain": domain,
                "source_video": source,
                "start_frame": "100",
                "end_frame": "160",
                "fps": "30",
                "fall_intervals": "100-160",
            }
        )
        for normal_index in range(2):
            rows.append(
                {
                    "clip_id": f"{source}:normal:{normal_index}",
                    "clip_path": f"clips/{source}/normal_{normal_index}.mp4",
                    "label": "0",
                    "label_name": "Normal",
                    "domain": domain,
                    "source_video": source,
                    "start_frame": str(normal_index * 40),
                    "end_frame": str(normal_index * 40 + 32),
                    "fps": "30",
                    "fall_intervals": "",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class FinalSourceVideoSplitTest(unittest.TestCase):
    def test_create_split_writes_balanced_source_video_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            output_dir = root / "split"
            write_metadata(metadata)

            summary = create_split(metadata, output_dir, seed=7)
            all_rows = read_rows(output_dir / "all.csv")

            self.assertEqual(leakage(all_rows), {})
            self.assertEqual(summary["leakage_check"], "PASS")
            self.assertEqual(summary["class_balance_check"], "PASS")
            for split in ("train", "val", "test"):
                counts = summary["class_counts"][split]
                self.assertGreater(counts["Normal"], 0)
                self.assertEqual(counts["Normal"], counts["Faint"])
                self.assertTrue((output_dir / f"{split}.csv").exists())
            self.assertTrue((output_dir / "split_summary.json").exists())
            self.assertTrue((output_dir / "split_report.md").exists())

    def test_cli_creates_summary_json_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            output_dir = root / "split"
            write_metadata(metadata)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/create_final_source_video_split.py",
                    "--metadata-csv",
                    str(metadata),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "11",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            report = (output_dir / "split_report.md").read_text(encoding="utf-8")
            self.assertEqual(payload["leakage_check"], "PASS")
            self.assertIn("Final Source-Video Split Report", report)


if __name__ == "__main__":
    unittest.main()
