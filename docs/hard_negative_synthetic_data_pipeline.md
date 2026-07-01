# Retraining Data Manifest v2 Pipeline

This pipeline keeps the original `metadata.csv` as source data and writes retraining rows to a separate `training_manifest_v2.csv`.

## Outputs

```text
data/manifests/error_rows/false_positives.csv
data/manifests/error_rows/false_negatives.csv
data/manifests/hard_negative_candidates.csv
data/manifests/faint_reinforcement_candidates.csv
data/manifests/synthetic_candidates.csv
data/manifests/training_manifest_v2.csv
```

Candidate exports are review queues. They default to `review_status=pending` and are not used for training until a human changes selected rows to `review_status=approved`.

## Required Columns

`training_manifest_v2.csv` always starts with these columns:

```text
clip_id,clip_path,label,label_name,source_type,parent_clip_id,review_status,
failure_type,scenario_tag,augmentation_type,augmentation_config,random_seed,
split_group_id,created_at
```

Recommended metadata such as `reviewer`, `reviewed_at`, `original_clip_id`, `source_video`, `start_frame`, `end_frame`, `fps`, `width`, `height`, and `notes` is preserved when present. Legacy input aliases such as `reason`, `synthetic_type`, and `augmentation_seed` are read for compatibility, but the v2 contract uses `scenario_tag`, `augmentation_type`, and `random_seed`.

## Workflow

Extract FP/FN rows from evaluation predictions:

```bash
python scripts/extract_lstm_fp_fn.py \
  --predictions benchmark/results/lstm_yolo26n_final_split_test_audit/YOLO26n-pose/eval_predictions.csv \
  --metadata-csv data/splits/final_source_video_split/all.csv \
  --output-dir data/manifests/error_rows \
  --threshold 0.3
```

Export FP rows as Hard Negative candidates and FN rows as Faint Reinforcement candidates:

```bash
python scripts/export_hard_negative_candidates.py \
  --false-positives-csv data/manifests/error_rows/false_positives.csv \
  --false-negatives-csv data/manifests/error_rows/false_negatives.csv \
  --metadata-csv data/splits/final_source_video_split/all.csv \
  --output-dir data/manifests
```

Generate lightweight synthetic candidates from approved Faint reinforcement rows:

```bash
python scripts/generate_synthetic_candidates.py \
  --input-csv data/manifests/faint_reinforcement_candidates.csv \
  --output-csv data/manifests/synthetic_candidates.csv \
  --synthetic-types brightness,noise,blur,compression,scale_down,partial_occlusion,horizontal_flip \
  --seed 42
```

To actually write augmented video clips from existing parent clips, add:

```bash
python scripts/generate_synthetic_candidates.py \
  --input-csv data/manifests/faint_reinforcement_candidates.csv \
  --output-csv data/manifests/synthetic_candidates.csv \
  --generate-media \
  --media-output-dir data/synthetic
```

Build the approved-only training manifest:

```bash
python scripts/build_training_manifest_v2.py \
  --base-metadata-csv data/splits/final_source_video_split/all.csv \
  --output-csv data/manifests/training_manifest_v2.csv \
  --max-synthetic-ratio 0.3
```

Check review, synthetic parent, ratio, and parent/split-group leakage rules:

```bash
python scripts/check_manifest_leakage.py \
  --manifest data/manifests/training_manifest_v2.csv \
  --max-synthetic-ratio 0.3
```

Train by explicitly pointing the existing pipeline at the v2 manifest:

```bash
METADATA_CSV=data/manifests/training_manifest_v2.csv \
bash scripts/run_yolo26n_final_lstm.sh train
```

## Dry Runs

These commands exercise the shape of the pipeline without requiring production metadata or writing manifests:

```bash
python scripts/export_hard_negative_candidates.py --sample --dry-run
python scripts/generate_synthetic_candidates.py --sample --dry-run
python scripts/build_training_manifest_v2.py --sample --dry-run
```

## Safety Rules

- `metadata.csv` is never overwritten. The builder refuses to use the same input and output path.
- `pending`, `rejected`, and `needs_review` candidate rows are excluded from `training_manifest_v2.csv`.
- Synthetic rows must have `parent_clip_id`.
- Synthetic rows inherit `split_group_id` from `split_group_id`, `parent_clip_id`, or `source_video`.
- The leakage checker rejects any `split_group_id` or `parent_clip_id` appearing across multiple splits.
- Synthetic rows are capped by `--max-synthetic-ratio`, default `0.3`.
- Supported OpenCV augmentation types are `brightness`, `noise`, `blur`, `compression`, `scale_down`, `partial_occlusion`, and `horizontal_flip`.

## Current Risks

- Human review is CSV-based; there is no dedicated review UI yet.
- FP/FN export quality depends on the evaluation logs and scenario tags supplied by the current run.
- Synthetic augmentation is classical OpenCV transformation only; actual model improvement still requires controlled retraining and evaluation.
