import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extract_lstm_fp_fn import extract_errors, resolve_predictions_path


class ExtractLstmFpFnTest(unittest.TestCase):
    def test_extract_errors_writes_fp_and_fn_with_metadata_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "eval_predictions.csv"
            metadata = root / "metadata.csv"
            output_dir = root / "errors"
            self._write_csv(
                predictions,
                [
                    {
                        "sequence_index": "0",
                        "clip_id": "clip_fn",
                        "frame_start": "10",
                        "frame_end": "39",
                        "true_label": "Faint",
                        "pred_label": "Normal",
                        "normal_prob": "0.72",
                        "faint_prob": "0.28",
                    },
                    {
                        "sequence_index": "1",
                        "clip_id": "clip_fp",
                        "frame_start": "40",
                        "frame_end": "69",
                        "true_label": "Normal",
                        "pred_label": "Faint",
                        "normal_prob": "0.22",
                        "faint_prob": "0.78",
                    },
                    {
                        "sequence_index": "2",
                        "clip_id": "clip_tp",
                        "frame_start": "70",
                        "frame_end": "99",
                        "true_label": "Faint",
                        "pred_label": "Faint",
                        "normal_prob": "0.1",
                        "faint_prob": "0.9",
                    },
                ],
            )
            self._write_csv(
                metadata,
                [
                    {"clip_id": "clip_fn", "clip_path": "clips/fn.mp4", "source_video": "source/fn.mp4"},
                    {"clip_id": "clip_fp", "clip_path": "clips/fp.mp4", "source_video": "source/fp.mp4"},
                ],
            )

            result = extract_errors(predictions, output_dir, metadata, threshold=None, sequence_length=30)

            self.assertEqual(result["false_negatives"], 1)
            self.assertEqual(result["false_positives"], 1)
            fn_rows = self._read_csv(output_dir / "false_negatives.csv")
            fp_rows = self._read_csv(output_dir / "false_positives.csv")
            self.assertEqual(fn_rows[0]["error_type"], "FN")
            self.assertEqual(fn_rows[0]["clip_path"], "clips/fn.mp4")
            self.assertEqual(fn_rows[0]["sequence_start"], "10")
            self.assertEqual(fn_rows[0]["sequence_end"], "39")
            self.assertEqual(fp_rows[0]["error_type"], "FP")
            self.assertEqual(fp_rows[0]["confidence"], "0.78")

    def test_extract_errors_can_recompute_prediction_with_threshold(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "eval_predictions.csv"
            output_dir = root / "errors"
            self._write_csv(
                predictions,
                [
                    {
                        "sequence_index": "0",
                        "clip_id": "clip_threshold_fn",
                        "frame_start": "0",
                        "frame_end": "29",
                        "true_label": "Faint",
                        "pred_label": "Faint",
                        "normal_prob": "0.55",
                        "faint_prob": "0.45",
                    },
                ],
            )

            result = extract_errors(predictions, output_dir, None, threshold=0.5, sequence_length=30)

            self.assertEqual(result["false_negatives"], 1)
            self.assertEqual(self._read_csv(output_dir / "false_negatives.csv")[0]["prediction"], "Normal")

    def test_resolve_predictions_path_prefers_sequence_30_model_output(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "lstm_sequence_length_8_16_30_yolo26n_cache_v2"
            predictions = run_dir / "sequence_length_30" / "YOLO26n-pose" / "eval_predictions.csv"
            predictions.parent.mkdir(parents=True)
            predictions.write_text("sequence_index,clip_id\n", encoding="utf-8")

            self.assertEqual(resolve_predictions_path(None, run_dir), predictions)

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
