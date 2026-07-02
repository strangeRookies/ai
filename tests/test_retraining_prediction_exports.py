import csv
import tempfile
import unittest
from pathlib import Path

from ai.learning.retraining_prediction_exports import write_prediction_exports


class RetrainingPredictionExportsTest(unittest.TestCase):
    def test_write_prediction_exports_creates_prediction_fp_and_fn_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = [
                {
                    "clip_id": "normal_clip",
                    "clip_path": "clips/normal.mp4",
                    "label_name": "Normal",
                    "source_video": "source_a.mp4",
                    "sequence_index": 0,
                    "frame_id": "100",
                    "scenario_tag": "bending",
                },
                {
                    "clip_id": "faint_clip",
                    "clip_path": "clips/faint.mp4",
                    "label_name": "Faint",
                    "source_video": "source_b.mp4",
                    "sequence_index": 1,
                    "frame_id": "200",
                    "scenario_tag": "night",
                },
                {
                    "clip_id": "normal_ok",
                    "clip_path": "clips/normal_ok.mp4",
                    "label_name": "Normal",
                    "source_video": "source_c.mp4",
                    "sequence_index": 2,
                    "frame_id": "300",
                    "scenario_tag": "none",
                },
            ]

            summary = write_prediction_exports(
                root,
                "baseline",
                test_y=[0, 1, 0],
                preds=[1, 0, 0],
                faint_probs=[0.82, 0.2, 0.1],
                seq_metadata=metadata,
                threshold=0.5,
            )

            self.assertEqual(summary["predictions"], 3)
            self.assertEqual(summary["false_positives"], 1)
            self.assertEqual(summary["false_negatives"], 1)

            fp_rows = self._read_csv(root / "baseline_false_positives.csv")
            fn_rows = self._read_csv(root / "baseline_false_negatives.csv")
            all_rows = self._read_csv(root / "baseline_predictions.csv")

            self.assertEqual(len(all_rows), 3)
            self.assertEqual(fp_rows[0]["error_type"], "FP")
            self.assertEqual(fp_rows[0]["clip_id"], "normal_clip")
            self.assertEqual(fp_rows[0]["prediction"], "Faint")
            self.assertEqual(fp_rows[0]["ground_truth_label"], "Normal")
            self.assertEqual(fp_rows[0]["faint_prob"], "0.820000")
            self.assertEqual(fp_rows[0]["source_video"], "source_a.mp4")
            self.assertEqual(fn_rows[0]["error_type"], "FN")
            self.assertEqual(fn_rows[0]["clip_id"], "faint_clip")
            self.assertEqual(fn_rows[0]["prediction"], "Normal")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as fp:
            return list(csv.DictReader(fp))


if __name__ == "__main__":
    unittest.main()
