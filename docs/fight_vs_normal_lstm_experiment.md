# Normal vs Fight LSTM Experiment

## Purpose

This experiment checks whether a separate LSTM can classify `Normal` and `Fight` from YOLO26n-pose keypoint NPZ files. It does not replace the current Normal/Faint or fall pipeline.

## Dataset

- CSV: `data/fight_vs_normal_npz.csv`
- Columns: `npz_path`, `label`, `label_name`, `clip_id`, `domain`
- Class mapping: `0: Normal`, `1: Fight`
- Expected sample count for the first run: 120 rows, 60 Normal, 60 Fight

## Smoke Test

```bash
python scripts/train_fight_vs_normal_lstm.py \
  --csv data/fight_vs_normal_npz.csv \
  --epochs 1 \
  --batch-size 4 \
  --output-dir runs/fight_vs_normal_lstm_smoke
```

## Full Run

```bash
python scripts/train_fight_vs_normal_lstm.py \
  --csv data/fight_vs_normal_npz.csv \
  --epochs 20 \
  --batch-size 32 \
  --output-dir runs/fight_vs_normal_lstm
```

## Outputs

The script writes `best.pt`, `history.json`, `summary.json`, `npz_inspection.json`, `test_predictions.csv`, and `confusion_matrix.csv` under the selected output directory.

## Notes

The dataset has only 120 clips, so this is a first feasibility check, not an operations-ready violence detector. Normal samples come from the fall/faint experiment context, so future work should add harder negative examples, more assault/fight sub-labels, and validation on real camera video.
