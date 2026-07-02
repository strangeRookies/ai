# Self-Improving Error Mining And Synthetic Feature Plan

Date: 2026-07-02

## Scope

This work prepares a Self-Improving AI data loop without changing production inference defaults. It does not apply a 54-dim model, does not change the default action model, does not change the production threshold, and does not auto-merge hard negatives or synthetic samples into a train set.

## Implemented Flow

1. Read prediction rows from realtime or stored-video evaluation output.
2. Compare `prediction_label` with `ground_truth_label`.
3. Route FP rows (`Faint` predicted, `Normal` truth) to hard-negative candidates.
4. Route FN rows (`Normal` predicted, `Faint/Fall` truth) to faint/fall reinforcement candidates.
5. Validate source video, clip id, frame range, label interval verification, feature schema, bbox features, and feature dimension.
6. Write invalid rows to quarantine instead of train candidates.
7. Build preview-only feature-level synthetic augmentation manifests from accepted 54-dim candidates.

## Candidate Manifest Outputs

Sample/manual CLI evidence:

- Hard negative: `runs/self_improving_error_mining/verification/manual_cli_final/hard_negative_candidates.jsonl`
- FN reinforcement: `runs/self_improving_error_mining/verification/manual_cli_final/faint_fall_reinforcement_candidates.jsonl`
- Quarantine: `runs/self_improving_error_mining/verification/manual_cli_final/quarantine.jsonl`
- Synthetic preview: `runs/self_improving_error_mining/verification/manual_cli_final/synthetic_preview.jsonl`
- Summary: `runs/self_improving_error_mining/verification/manual_cli_final/summary.json`
- Pipeline log: `runs/self_improving_error_mining/verification/manual_cli_final/pipeline.log`

Appendable JSONL fields include:

- `source_video`
- `clip_id`
- `start_frame`
- `end_frame`
- `frameId`
- `prediction_label`
- `prediction_score`
- `ground_truth_label`
- `threshold`
- `model_name`
- `checkpoint_path`
- `sequence_length`
- `sequence_stride`
- `feature_schema`
- `feature_dim`
- `bbox_features`
- `keypoint_confidence_summary`
- `auto_merge_to_train=false`
- `review_status=pending`

## Quarantine Policy

Rows are quarantined when they cannot be safely reviewed for retraining. Current reasons:

- `missing_source_video`
- `missing_clip_id`
- `missing_prediction_label`
- `missing_ground_truth_label`
- `missing_model_name`
- `missing_checkpoint_path`
- `missing_label_interval`
- `missing_frame_id`
- `invalid_frame_range`
- `unsupported_feature_schema`
- `missing_sequence`
- `missing_bbox_feature`
- `malformed_bbox_feature`
- `feature_dim_mismatch`

Sample run counts:

```json
{
  "missing_bbox_feature": 1,
  "missing_label_interval": 1
}
```

## keypoint_bbox54 Safety

`keypoint_bbox54` is used only for candidate validation and synthetic preview generation in this work.

The pipeline calls the existing feature builder with `feature_schema=keypoint_bbox54` and `input_size=54`, then rejects the sample if:

- the sequence is missing,
- any detection in the sequence lacks a bbox,
- the resulting feature width is not 54,
- bbox feature columns `51..53` are all zero.

This avoids treating a 51-dim vector plus `[0, 0, 0]` as a valid 54-dim bbox sample.

## Feature-Level Synthetic Augmentation

Synthetic Data is defined as feature-level transformation over keypoint/bbox sequences, not generated video. The dry-run creates preview manifest rows only and never writes train manifests.

Included preview types:

- `keypoint_noise`
- `bbox_scale_jitter`
- `bbox_aspect_jitter`
- `confidence_drop`
- `frame_drop`
- `temporal_jitter`
- `horizontal_flip`

All preview rows retain:

- `feature_schema=keypoint_bbox54`
- `feature_dim=54`
- `source_type=synthetic_preview`
- `auto_merge_to_train=false`
- `review_status=preview_only`

Sample run generated 14 preview rows from 2 accepted candidates.

## Low-Memory And Resume Plan

The pipeline writes JSONL append files, so long runs can resume by continuing append output and deduping by `candidate_id` in the later dataset export step. For GPU or VRAM-constrained environments, later real-data mining should:

- run on CPU/RAM when CUDA is unavailable,
- process prediction logs in chunks,
- flush candidates and quarantine rows after each chunk,
- reduce evaluation batch size before retrying,
- keep a rolling progress log with current file, processed sample count, FP/FN counts, synthetic preview count, quarantine count, and last error location.

## Verification Evidence

Commands run:

```powershell
python -m unittest discover -s tests -p "test_self_improving_error_mining.py"
python -m unittest discover -s tests -p "test_lstm_action_classifier.py"
python scripts/inspect_lstm_feature_shape.py --feature-schema keypoint_bbox54 --expected-input-size 54 --sequence-length 3 --sample 1
python scripts/run_self_improving_error_mining.py --sample --output-dir runs/self_improving_error_mining/verification/manual_cli_final
python -m py_compile ai/learning/self_improving_error_mining.py ai/learning/self_improving_samples.py ai/learning/self_improving_synthetic.py scripts/run_self_improving_error_mining.py tests/test_self_improving_error_mining.py
```

Evidence files:

- `runs/self_improving_error_mining/verification/unit_final.log`
- `runs/self_improving_error_mining/verification/keypoint51_smoke_final.log`
- `runs/self_improving_error_mining/verification/bbox54_feature_builder_final.log`
- `runs/self_improving_error_mining/verification/cli_final.log`
- `runs/self_improving_error_mining/verification/py_compile_final.log`

The unit suite covers FP/FN routing, appendable manifests, quarantine routing, non-numeric frame metadata, malformed bbox metadata, synthetic dry-run output, feature dimension preservation, and rejection of non-54-dim synthetic candidates.

## Local Verification Limits

The local workspace has no `strange_ai/data` directory, so real dataset validation was not run. The completed checks are mock/sample based. Still pending for GPU server or real dataset:

- mine real realtime/stored-video prediction logs,
- verify source-video and label-interval joins against real metadata,
- compare candidate counts by source video and frame range,
- run chunked large-scale error mining,
- review quarantined rows with real label files,
- export approved candidates into a retraining dataset.

## Conditions For Next Dataset Export Step

Before implementing retraining dataset export:

1. Every candidate must have `review_status=approved`.
2. Hard negative frame ranges must be proven non-overlapping with fall/faint intervals.
3. FN reinforcement ranges must overlap verified fall/faint intervals.
4. Dataset export must dedupe by `candidate_id`, `source_video`, `clip_id`, and frame range.
5. Addition ratios must be explicit, for example hard-negative ratios `0.05`, `0.10`, `0.20`.
6. Synthetic preview rows must remain excluded until separately approved and materialized as feature-level training samples.
7. Production defaults must remain unchanged until post-training evaluation selects a candidate model and policy.
