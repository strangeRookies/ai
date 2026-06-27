import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ai.action.fight_vs_normal_dataset import collect_sequences, load_fight_rows, stratified_split
from ai.action.fight_vs_normal_metrics import classification_metrics


def write_keypoint_npz(path: Path, offset: float = 0.0) -> None:
    keypoints = np.ones((6, 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = 0.25 + offset
    keypoints[:, :, 1] = 0.5
    keypoints[:, :, 2] = 0.9
    np.savez_compressed(path, data=keypoints)


def write_dataset(tmp: Path, rows_per_class: int = 10) -> Path:
    csv_path = tmp / "fight_vs_normal_npz.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["npz_path", "label", "label_name", "clip_id", "domain"])
        writer.writeheader()
        for label, name in ((0, "Normal"), (1, "Fight")):
            for index in range(rows_per_class):
                npz_path = tmp / f"{name.lower()}_{index}.npz"
                write_keypoint_npz(npz_path, offset=0.01 * label)
                writer.writerow({"npz_path": str(npz_path), "label": label, "label_name": name, "clip_id": f"{name}_{index}", "domain": "test"})
    return csv_path


class FightVsNormalLstmTest(unittest.TestCase):
    def test_load_rows_and_stratified_split_keep_class_balance(self):
        with TemporaryDirectory() as tmpdir:
            rows = load_fight_rows(write_dataset(Path(tmpdir), rows_per_class=10))

            splits = stratified_split(rows, seed=7)

        self.assertEqual(len(rows), 20)
        self.assertEqual([row.label_name for row in rows].count("Normal"), 10)
        self.assertEqual([row.label_name for row in rows].count("Fight"), 10)
        self.assertEqual([row.label_name for row in splits.train].count("Normal"), 7)
        self.assertEqual([row.label_name for row in splits.train].count("Fight"), 7)
        self.assertEqual([row.label_name for row in splits.val].count("Normal"), 2)
        self.assertEqual([row.label_name for row in splits.test].count("Fight"), 1)

    def test_collect_sequences_reads_keypoint_npz_as_51_dim_lstm_input(self):
        with TemporaryDirectory() as tmpdir:
            rows = load_fight_rows(write_dataset(Path(tmpdir), rows_per_class=1))

            batch = collect_sequences(rows, sequence_length=4, sequence_stride=2)

        self.assertEqual(batch.x.shape, (4, 4, 51))
        self.assertEqual(batch.y.tolist(), [0, 0, 1, 1])
        self.assertEqual(batch.rows[0]["label_name"], "Normal")

    def test_classification_metrics_use_fight_as_positive_class(self):
        metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 1])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["labels"], ["Normal", "Fight"])
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 1], [1, 1]])

    def test_smoke_cli_trains_one_epoch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is required for the smoke training test")
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = write_dataset(tmp, rows_per_class=6)
            output_dir = tmp / "run"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_fight_vs_normal_lstm.py",
                    "--csv",
                    str(csv_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--sequence-length",
                    "4",
                    "--sequence-stride",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "confusion_matrix.csv").exists())


if __name__ == "__main__":
    unittest.main()
