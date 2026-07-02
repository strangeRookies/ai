# Hard Negative Retraining Comparison Guidebook

Date: 2026-07-02

This guidebook explains how a human operator should compare the existing baseline model with hard-negative retraining variants. Codex did not run Python, unittest, training, evaluation, or GPU commands for this guidebook.

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

## 2. Files And Directories To Check Before Running

Check these paths before running any comparison:

| Path | What To Check |
| --- | --- |
| `data/manifests/hard_negative_candidates.csv` or `.jsonl` | Hard-negative candidate source. Only `review_status=approved` rows may be exported. |
| `runs/self_improving_error_mining/verification/manual_cli_final/hard_negative_candidates.jsonl` | Sample candidate manifest from the preparation pipeline, useful for schema inspection only. |
| `data/splits/final_source_video_split/all.csv` | Baseline metadata or baseline split source. |
| `data/manifests/training_manifest_v2.csv` | Candidate retraining manifest output, if already generated. |
| `runs/hard_negative_retraining_comparison/` | Expected run output root for comparison logs, exports, metrics, and reports. |
| `docs/hard_negative_retraining_performance_comparison.md` | Final human-readable comparison report path. |
| `scripts/evaluate_retraining_manifest_v2.py` | Existing evaluator. For strict `keypoint_bbox54`, verify it does not pad 51-dim data into 54-dim data before using it. |

Critical feature rule:

- `feature_schema` must be `keypoint_bbox54`.
- `feature_dim` must be `54`.
- bbox columns `51..53` must contain real bbox width, height, and area features.
- Do not use any path that turns 51-dim vectors into 54-dim vectors with `[0, 0, 0]` or `np.pad`.

## 3. Test Execution Order

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

## 4. Hard Negative Candidate Validation Order

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

## 5. Ratio-Specific Export Method

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
| `python scripts/export_hard_negative_ratios.py --baseline-manifest data/splits/final_source_video_split/all.csv --hard-negative-candidates data/manifests/hard_negative_candidates.csv --ratios 0.05,0.10,0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/train_exports` | Creates `baseline`, `hn_0.05`, `hn_0.10`, and `hn_0.20` experiment manifests from approved strict bbox54 candidates only. |
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

## 6. Baseline Vs Hard Negative Model Comparison Method

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

## 7. Threshold Sweep Check

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

## 8. Expected Result File Locations

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

## 9. Result Interpretation Criteria

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

## 10. Error Types To Check When A Run Fails

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

## 11. GPU Server Additional Checks

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
4. `python scripts/export_hard_negative_ratios.py --baseline-manifest data/splits/final_source_video_split/all.csv --hard-negative-candidates data/manifests/hard_negative_candidates.csv --ratios 0.05,0.10,0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison/train_exports`
   - Exports baseline and ratio-specific train manifests.
5. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.05.csv`
   - Checks split leakage for `hn_0.05`.
6. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.10.csv`
   - Checks split leakage for `hn_0.10`.
7. `python scripts/check_manifest_leakage.py --manifest runs/hard_negative_retraining_comparison/train_exports/hn_0.20.csv`
   - Checks split leakage for `hn_0.20`.
8. `python -m ai.action.train_lstm --dataset-csv runs/hard_negative_retraining_comparison/train_exports/<experiment>.csv --input-size 54 --feature-schema keypoint_bbox54 --device cuda --epochs 30 --batch-size 64 --output-dir runs/hard_negative_retraining_comparison/models/<experiment>`
   - Trains each experiment variant. Run once per experiment label.
9. `python scripts/compare_hard_negative_retraining.py --eval-split runs/hard_negative_retraining_comparison/eval_split.csv --checkpoints <four-checkpoints> --labels baseline,hn_0.05,hn_0.10,hn_0.20 --feature-schema keypoint_bbox54 --feature-dim 54 --output-dir runs/hard_negative_retraining_comparison --report-path docs/hard_negative_retraining_performance_comparison.md`
   - Compares every checkpoint on the same evaluation split.
10. `python scripts/audit_lstm_thresholds.py --predictions runs/hard_negative_retraining_comparison/predictions/<experiment>_eval_predictions.csv --output-dir runs/hard_negative_retraining_comparison/threshold_audit/<experiment>`
    - Produces per-experiment threshold sweep artifacts.

Final judgment should be made only after real GPU output exists. Do not infer real performance from sample/mock data.
