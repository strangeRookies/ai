# Hard Negative Retraining Comparison Guidebook

Date: 2026-07-02

This guidebook explains how a human operator should compare the existing baseline model with hard-negative retraining variants. Codex did not run Python, unittest, training, evaluation, or GPU commands for this guidebook.

## 0. Decision: Rebuild Around Strict Bbox54

Use this guidebook only for a `keypoint_bbox54` experiment. The current `keypoint51` balanced result is useful as a historical baseline, but it is not the target experiment.

The dataset and training path may need to be rebuilt because the existing evaluation/data-loader path contains `np.pad` fallback logic that can turn 51-dim keypoint features into fake 54-dim vectors. That path is not valid for this comparison.

Required order:

1. Build or locate the full v2 metadata pool.
2. Add a strict bbox54 validation gate.
3. Derive a bbox54-only baseline train/val/test split.
4. Mine FP rows from the bbox54 baseline evaluation.
5. Export only approved hard-negative candidates.
6. Build `baseline`, `hn_0.05`, `hn_0.10`, and `hn_0.20` experiment manifests.
7. Train/evaluate all variants on the same bbox54 evaluation split.

Do not continue if any sample enters through 51-to-54 padding.

## 1. Current Purpose

The goal is to compare `baseline`, `hn_0.05`, `hn_0.10`, and `hn_0.20` fairly on the same evaluation split.

The comparison must answer:

- Did false positives decrease?
- Did false negatives avoid increasing?
- Did Recall stay stable enough for safety monitoring?
- Did F1 improve?
- Does the trend hold across thresholds `0.3`, `0.4`, `0.5`, `0.6`, and `0.7`?
- Is the improvement broad, or only concentrated in a specific `source_video` or scenario?

This work must not change production inference defaults, production threshold defaults, or the existing `keypoint51` path. Hard-negative rows are experiment inputs only; they are not automatically merged into the production train set.

The comparison must be based on real 54-dim bbox features:

- `input_size=54`
- `feature_schema=keypoint_bbox54`
- checkpoint metadata must record `feature_schema_version=keypoint_bbox54`
- bbox feature columns `51..53` must be real bbox width, height, and area values

If the result shows `input_size=51` or `feature_schema_version=keypoint51`, it is not a bbox54 comparison result.

## 2. Files And Directories To Check Before Running

Check these paths before running any comparison:

| Path | What To Check |
| --- | --- |
| `data/manifests/hard_negative_candidates.csv` or `.jsonl` | Hard-negative candidate source. Only `review_status=approved` rows may be exported. |
| `runs/self_improving_error_mining/verification/manual_cli_final/hard_negative_candidates.jsonl` | Sample candidate manifest from the preparation pipeline, useful for schema inspection only. |
| `data/metadata/metadata.csv` | Baseline metadata or baseline split source. |
| `data/manifests/training_manifest_v2.csv` | Candidate retraining manifest output, if already generated. |
| `runs/hard_negative_retraining_comparison/` | Expected run output root for comparison logs, exports, metrics, and reports. |
| `docs/hard_negative_retraining_performance_comparison.md` | Final human-readable comparison report path. |
| `scripts/evaluate_retraining_manifest_v2.py` | Existing evaluator. For strict `keypoint_bbox54`, verify it does not pad 51-dim data into 54-dim data before using it. |
| `ai/action/fight_vs_normal_dataset.py` | Existing dataset loader. For strict `keypoint_bbox54`, verify it does not pad 51-dim features into 54-dim data. |

Critical feature rule:

- `feature_schema` must be `keypoint_bbox54`.
- `feature_dim` must be `54`.
- bbox columns `51..53` must contain real bbox width, height, and area features.
- Do not use any path that turns 51-dim vectors into 54-dim vectors with `[0, 0, 0]` or `np.pad`.

Known invalid path to guard before bbox54 execution:

```bash
grep -R "np.pad" -n scripts/evaluate_retraining_manifest_v2.py ai/action/fight_vs_normal_dataset.py
```

If this returns padding branches that run when `feature_schema=keypoint_bbox54`, do not use those branches for the bbox54 experiment. Add a strict failure instead:

```text
keypoint_bbox54 requires real 54-dim bbox features; 51-to-54 padding is forbidden
```

## 3. Build The V2 Pool, Then Derive Bbox54 Data

`training_manifest_v2.csv` is the full metadata pool. It is not automatically the final bbox54 training set.

Example commands for the user:

| Example Command | What This Confirms |
| --- | --- |
| `python scripts/build_training_manifest_v2.py --base-metadata-csv data/metadata/metadata.csv --hard-negative-csv data/manifests/hard_negative_candidates.csv --faint-reinforcement-csv data/manifests/faint_reinforcement_candidates.csv --synthetic-csv data/manifests/synthetic_candidates.csv --output-csv data/manifests/training_manifest_v2.csv` | Builds the full v2 metadata pool. If hard-negative files are missing, the output may contain only `source_type=real`. |
| `python scripts/check_manifest_leakage.py --manifest data/manifests/training_manifest_v2.csv` | Checks split leakage and unapproved candidate rows. |
| `python scripts/inspect_manifest_v2.py --manifest data/manifests/training_manifest_v2.csv --group-by source_type` | Confirms whether hard-negative candidates are actually present. If this helper does not exist, create a read-only inspection tool first. |
| `python scripts/inspect_manifest_v2.py --manifest data/manifests/training_manifest_v2.csv --group-by label` | Confirms class balance before sampling. |

If the output has only:

```json
{"source_type_counts": {"real": 215541}}
```

then hard negatives are not included yet. Treat that result as a real-data pool or keypoint51/bbox54 baseline pool, not as a hard-negative experiment.

For the actual bbox54 run, derive a strict bbox54-only split:

| Example Command | What This Confirms |
| --- | --- |
| `python scripts/export_strict_bbox54_manifest.py --input data/manifests/training_manifest_v2.csv --output data/manifests/training_manifest_v2_bbox54.csv --feature-schema keypoint_bbox54 --feature-dim 54 --reject-padding` | Keeps only samples with real bbox54 features and rejects 51-to-54 padding. |
| `python scripts/split_bbox54_manifest.py --manifest data/manifests/training_manifest_v2_bbox54.csv --train-limit 7000 --val-limit 1500 --test-limit 1400 --per-class --balance-labels --output-dir runs/hard_negative_retraining_comparison/bbox54_splits` | Creates class-balanced bbox54 train/val/test splits. |

Expected bbox54 split files:

- `runs/hard_negative_retraining_comparison/bbox54_splits/train.csv`
- `runs/hard_negative_retraining_comparison/bbox54_splits/val.csv`
- `runs/hard_negative_retraining_comparison/bbox54_splits/test.csv`
- `runs/hard_negative_retraining_comparison/bbox54_splits/split_summary.json`
- `runs/hard_negative_retraining_comparison/bbox54_splits/rejected_non_bbox54.csv`

## 4. Test Execution Order

The following commands are examples for the user to run. Codex did not run them.

| Example Command | What This Confirms |
| --- | --- |
| `python -m unittest discover -s tests -p "test_lstm_action_classifier.py"` | Existing keypoint/LSTM classifier smoke behavior still works. |
| `python -m unittest discover -s tests -p "test_feature_schema_51_vs_54.py"` | `keypoint_bbox54` feature builder can produce 54-dim features and checkpoint schema guards still work. |
| `python -m unittest discover -s tests -p "test_self_improving_error_mining.py"` | FP/FN mining, approved candidate shape, quarantine, and synthetic dry-run guards still behave as expected. |
| `python -m unittest discover -s tests -p "*hard_negative*"` | Hard-negative comparison/export tests, if implemented, pass before any GPU training is started. |

Pass condition:

- All tests must pass before training or evaluation.
- If a test reveals 51-to-54 padding, stop and fix the comparison path before running real bbox54 experiments.

## 5. Bbox54 Baseline Before Hard Negatives

Before hard-negative mining, run a strict bbox54 baseline. This baseline is the reference for FP mining and later ratio comparison.

Example commands for the user:

| Example Command | What This Confirms |
| --- | --- |
| `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/bbox54_splits/train.csv --val-csv runs/hard_negative_retraining_comparison/bbox54_splits/val.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/bbox54_baseline` | Trains the strict bbox54 baseline model. |
| `python scripts/evaluate_bbox54_checkpoint.py --checkpoint runs/hard_negative_retraining_comparison/models/bbox54_baseline/best.pt --eval-csv runs/hard_negative_retraining_comparison/bbox54_splits/test.csv --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/bbox54_baseline_eval` | Evaluates the bbox54 baseline on the fixed bbox54 test split. |
| `python scripts/inspect_checkpoint_metadata.py --checkpoint runs/hard_negative_retraining_comparison/models/bbox54_baseline/best.pt` | Confirms checkpoint metadata has `input_size=54` and `feature_schema_version=keypoint_bbox54`. |

Only after this stage should FP rows be mined as hard-negative candidates.

## 6. Hard Negative Candidate Validation Order

Validate hard-negative candidates before exporting ratio-specific train manifests.

| Step | Example Command | What This Confirms |
| --- | --- | --- |
| Count review statuses | `python scripts/inspect_candidate_manifest.py --input data/manifests/hard_negative_candidates.csv --group-by review_status` | Confirms how many candidates are `approved`, `pending`, `rejected`, or `needs_review`. |
| Check feature schema | `python scripts/inspect_candidate_manifest.py --input data/manifests/hard_negative_candidates.csv --require-feature-schema keypoint_bbox54 --require-feature-dim 54` | Confirms exported candidates are strict 54-dim bbox features. |
| Reject synthetic by default | `python scripts/inspect_candidate_manifest.py --input data/manifests/hard_negative_candidates.csv --exclude-source-type synthetic_preview` | Confirms synthetic preview rows are not included unless a separate approved-synthetic flag is intentionally used. |
| Check source distribution | `python scripts/inspect_candidate_manifest.py --input data/manifests/hard_negative_candidates.csv --group-by source_video` | Confirms candidates are not dominated by one source video. |
| Check scenario distribution | `python scripts/inspect_candidate_manifest.py --input data/manifests/hard_negative_candidates.csv --group-by scenario_tag` | Confirms candidates cover more than one failure scenario when metadata is available. |

If the inspection helper does not exist yet, implement it as a read-only/reporting tool first. Do not export candidates by manually editing CSV files.

Candidate rows must be excluded or quarantined when:

- `review_status` is not `approved`.
- `candidate_type` is not `hard_negative`.
- `feature_schema` is missing or not `keypoint_bbox54`.
- `feature_dim` is not `54`.
- bbox feature columns are missing or all zero.
- `source_video`, label interval, or frame range cannot be verified.
- `source_type=synthetic_preview` and `--include-synthetic-approved` was not explicitly chosen.

## 7. Ratio-Specific Export Method

Export ratio-specific experimental train manifests. The ratios should be explicit and reproducible.

| Variant | Intended Meaning |
| --- | --- |
| `baseline` | Existing baseline train manifest without hard-negative additions. |
| `hn_0.05` | Add approved hard negatives up to 5% of baseline train count. |
| `hn_0.10` | Add approved hard negatives up to 10% of baseline train count. |
| `hn_0.20` | Add approved hard negatives up to 20% of baseline train count. |

Example commands for the user:

| Example Command | What This Confirms |
| --- | --- |
| `python scripts/export_hard_negative_ratios.py --baseline-manifest data/metadata/metadata.csv --hard-negative-candidates data/manifests/hard_negative_candidates.csv --ratios 0.05,0.10,0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/train_exports` | Creates `baseline`, `hn_0.05`, `hn_0.10`, and `hn_0.20` experiment manifests from approved strict bbox54 candidates only. |
| `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv` | Confirms `hn_0.05` does not leak test/eval source videos or split groups into train. |
| `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv` | Same check for `hn_0.10`. |
| `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv` | Same check for `hn_0.20`. |

Expected export files:

- `runs/hard_negative_retraining_comparison/train_exports/baseline.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv`
- `runs/hard_negative_retraining_comparison/train_exports/export_summary.json`

`export_summary.json` should include:

- baseline train sample count,
- approved hard-negative available count,
- ratio target count,
- actual exported count per ratio,
- excluded count by reason,
- synthetic included/excluded status.

## 8. Baseline Vs Hard Negative Model Comparison Method

All variants must use the same evaluation split. The train manifest changes by ratio; the test/evaluation rows must not change.

Example commands for the user:

| Example Command | What This Confirms |
| --- | --- |
| `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/baseline.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/baseline` | Trains the strict bbox54 baseline experiment checkpoint. |
| `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/hn_0.05` | Trains the 5% hard-negative variant. |
| `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/hn_0.10` | Trains the 10% hard-negative variant. |
| `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/hn_0.20` | Trains the 20% hard-negative variant. |
| `python scripts/compare_hard_negative_retraining.py --eval-split runs/hard_negative_retraining_comparison/eval_split.csv --checkpoints runs/hard_negative_retraining_comparison/models/baseline/best.pt,runs/hard_negative_retraining_comparison/models/hn_0.05/best.pt,runs/hard_negative_retraining_comparison/models/hn_0.10/best.pt,runs/hard_negative_retraining_comparison/models/hn_0.20/best.pt --labels baseline,hn_0.05,hn_0.10,hn_0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison --report-path docs/hard_negative_retraining_performance_comparison.md` | Evaluates every checkpoint on the identical split and writes comparison outputs. |

If `scripts/compare_hard_negative_retraining.py` is not implemented yet, implement it before GPU execution. Do not use a comparison script that silently falls back to mock predictions or pads 51-dim arrays.

Low-memory fallback:

- First reduce `--batch-size` from `64` to `32`, then `16`, then `8`.
- If CUDA is unavailable, use CPU only for functional validation, not final performance claims.
- For long runs, require checkpoint files, intermediate metrics, and resume state under each experiment directory.

## 9. Threshold Sweep Check

The sweep must include exactly:

- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`

Example commands for the user:

| Example Command | What This Confirms |
| --- | --- |
| `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/baseline_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/baseline` | Produces baseline threshold metrics from saved predictions. |
| `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/hn_0.05_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/hn_0.05` | Produces 5% hard-negative threshold metrics. |
| `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/hn_0.10_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/hn_0.10` | Produces 10% hard-negative threshold metrics. |
| `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/hn_0.20_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/hn_0.20` | Produces 20% hard-negative threshold metrics. |

Expected threshold files:

- `runs/hard_negative_retraining_comparison/threshold_audit/<experiment>/threshold_audit.csv`
- `runs/hard_negative_retraining_comparison/threshold_audit/<experiment>/threshold_audit.json`
- `runs/hard_negative_retraining_comparison/threshold_audit/<experiment>/threshold_audit.md`

## 10. Expected Result File Locations

The run should produce:

- `runs/hard_negative_retraining_comparison/pipeline.log`
- `runs/hard_negative_retraining_comparison/progress_state.json`
- `runs/hard_negative_retraining_comparison/train_exports/export_summary.json`
- `runs/hard_negative_retraining_comparison/train_exports/baseline.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv`
- `runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv`
- `runs/hard_negative_retraining_comparison/models/<experiment>/best.pt`
- `runs/hard_negative_retraining_comparison/predictions/<experiment>_eval_predictions.csv`
- `runs/hard_negative_retraining_comparison/metrics/threshold_sweep.csv`
- `runs/hard_negative_retraining_comparison/metrics/experiment_summary.csv`
- `runs/hard_negative_retraining_comparison/metrics/scenario_breakdown.csv`
- `runs/hard_negative_retraining_comparison/metrics/source_video_breakdown.csv`
- `runs/hard_negative_retraining_comparison/metrics/comparison_summary.json`
- `docs/hard_negative_retraining_performance_comparison.md`

The report must state whether it is based on real GPU results or sample/mock validation. Do not treat mock output as model performance.

## 11. Result Interpretation Criteria

Use the same primary threshold, then inspect the full threshold sweep.

Core decision table:

| Criterion | Good Sign | Stop Or Reject Sign |
| --- | --- | --- |
| FP | FP decreases versus baseline. | FP is unchanged or only improves in one narrow source/scenario. |
| FN | FN stays equal or decreases. | FN increases, especially at lower thresholds such as `0.3` or `0.4`. |
| Recall | Recall is unchanged or only drops within a pre-agreed tolerance. | Recall drops materially. For safety monitoring, recall loss can outweigh FP reduction. |
| F1 | F1 increases versus baseline. | F1 drops, or increases only because Recall collapsed and FP fell artificially. |
| Threshold sweep | Improvement trend holds from `0.3` to `0.7`. | Only one threshold improves while adjacent thresholds degrade. |
| Scenario/source spread | Improvement appears across several scenarios and videos. | Improvement is concentrated in a single `source_video`, camera, place, season, or day/night group. |

Recommended ratio rule:

1. Reject any ratio with material Recall loss.
2. Prefer the smallest ratio that meaningfully reduces FP.
3. If `hn_0.05` and `hn_0.10` are close, prefer `hn_0.05`.
4. Use `hn_0.20` only if it reduces FP without increasing FN and without scenario overfitting.
5. Do not change production threshold based only on this comparison; treat threshold values as candidates until RTSP replay/smoke testing is complete.

## 12. Error Types To Check When A Run Fails

| Error Type | Likely Cause | What To Check |
| --- | --- | --- |
| `FileNotFoundError` | Manifest, checkpoint, or NPZ cache path is wrong. | Re-check paths in Section 2. |
| `review_status` violation | Pending/rejected rows entered export. | Candidate export must filter to `review_status=approved`. |
| `feature_dim_mismatch` | Non-54-dim row entered bbox54 experiment. | Confirm `feature_dim=54` and `feature_sequence` width is 54. |
| zero bbox columns | 51-dim data was padded or bbox was missing. | Stop; do not use that sample for keypoint_bbox54 comparison. |
| split leakage | Same source/clip/group appears in train and eval. | Rebuild split/export and run leakage checks again. |
| CUDA OOM | Batch size too large. | Reduce batch size and resume from checkpoint if available. |
| misleading mock fallback | Script generated plausible metrics without real data. | Do not report these as real performance. Re-run on GPU with real data. |
| threshold CSV missing | Evaluation predictions were not written or audit path is wrong. | Check `*_eval_predictions.csv` paths and rerun threshold audit. |
| result still says `input_size=51` | The run used keypoint51, not bbox54. | Rebuild strict bbox54 data and rerun with `--input-size 54 --feature-schema keypoint_bbox54`. |
| `np.pad` appears in bbox54 path | 51-to-54 fallback is still active. | Stop and add a hard failure for `keypoint_bbox54` when actual feature width is 51. |

## 13. GPU Server Additional Checks

On the GPU server, additionally confirm:

- CUDA is available and the selected GPU has enough free VRAM.
- All four experiments use the same `eval_split.csv`.
- All checkpoints record `input_size=54` and `feature_schema_version=keypoint_bbox54`.
- The loader never pads 51-dim features to 54-dim.
- Training logs include current experiment name, epoch, processed sample count, FP/FN, best F1, best Recall, and last error location.
- Long runs write checkpoint, intermediate metrics, and resume state.
- Scenario breakdown includes `scenario_tag`, `source_video`, `camera`, `day_night`, `place`, and `season` when those fields exist.
- The final report clearly distinguishes real GPU results from mock/sample validation.

## User Execution Checklist

Commands to run directly, in order, after the missing strict comparison/export helpers are available:

1. `python -m unittest discover -s tests -p "test_lstm_action_classifier.py"`
   - Confirms the existing baseline/keypoint path is not broken.
2. `python -m unittest discover -s tests -p "test_feature_schema_51_vs_54.py"`
   - Confirms bbox54 feature creation and schema guards.
3. `python -m unittest discover -s tests -p "*hard_negative*"`
   - Confirms approved-only export, ratio counts, same split, threshold sweep, and report generation tests.
4. `python scripts/export_strict_bbox54_manifest.py --input data/manifests/training_manifest_v2.csv --output data/manifests/training_manifest_v2_bbox54.csv --feature-schema keypoint_bbox54 --feature-dim 54 --reject-padding`
   - Creates a strict bbox54-only manifest and rejects padded 51-dim data.
5. `python scripts/split_bbox54_manifest.py --manifest data/manifests/training_manifest_v2_bbox54.csv --train-limit 7000 --val-limit 1500 --test-limit 1400 --per-class --balance-labels --output-dir runs/hard_negative_retraining_comparison/bbox54_splits`
   - Creates balanced bbox54 train/val/test splits.
6. `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/bbox54_splits/train.csv --val-csv runs/hard_negative_retraining_comparison/bbox54_splits/val.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/bbox54_baseline`
   - Trains the strict bbox54 baseline.
7. `python scripts/evaluate_bbox54_checkpoint.py --checkpoint runs/hard_negative_retraining_comparison/models/bbox54_baseline/best.pt --eval-csv runs/hard_negative_retraining_comparison/bbox54_splits/test.csv --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/bbox54_baseline_eval`
   - Evaluates bbox54 baseline and produces FP rows for mining.
8. `python scripts/export_hard_negative_ratios.py --baseline-manifest data/metadata/metadata.csv --hard-negative-candidates data/manifests/hard_negative_candidates.csv --ratios 0.05,0.10,0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/train_exports`
   - Exports baseline and ratio-specific train manifests.
9. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv`
   - Checks split leakage for `hn_0.05`.
10. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv`
   - Checks split leakage for `hn_0.10`.
11. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv`
   - Checks split leakage for `hn_0.20`.
12. `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/<experiment>.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/<experiment>`
   - Trains each experiment variant. Run once per experiment label.
13. `python scripts/compare_hard_negative_retraining.py --eval-split runs/hard_negative_retraining_comparison/bbox54_splits/test.csv --checkpoints <four-checkpoints> --labels baseline,hn_0.05,hn_0.10,hn_0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison --report-path docs/hard_negative_retraining_performance_comparison.md`
   - Compares every checkpoint on the same evaluation split.
14. `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/<experiment>_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/<experiment>`
    - Produces per-experiment threshold sweep artifacts.

Final judgment should be made only after real GPU output exists. Do not infer real performance from sample/mock data.
