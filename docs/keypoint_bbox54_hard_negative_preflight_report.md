# keypoint_bbox54 + Hard Negative Preflight Report

Date: 2026-07-02
Scope: pre-application audit before enabling a `keypoint_bbox54` checkpoint and hard-negative retraining.

## Verdict

Do not apply the 54-dim model or merge hard negatives yet.

The real-time inference path can load and preflight a `keypoint_bbox54` checkpoint with `input_size=54`, and the direct feature builder produces `(sequence_length, 54)` without padding when detections include bbox + frame shape. However, the training/evaluation cache paths still contain `51 -> 54` zero-padding fallbacks for `keypoint_bbox54`, and hard-negative candidate generation does not yet validate original label intervals, source video, or frame ranges strongly enough to prevent mislabeled negative samples.

## Feature Schema Confirmed From Code

Code sources:

- `ai/action/classifier.py`
- `ai/action/lstm_contract.py`
- `tests/test_feature_schema_51_vs_54.py`

`keypoint_bbox54` order:

| Index range | Feature | Normalization |
| :--- | :--- | :--- |
| `0..50` | 17 keypoints in detector order, each as `(x, y, confidence)` | `x / frame_width`, `y / frame_height`, raw confidence |
| `51` | `bbox_width_norm` | `(x2 - x1) / frame_width` |
| `52` | `bbox_height_norm` | `(y2 - y1) / frame_height` |
| `53` | `bbox_area_norm` | `bbox_width_norm * bbox_height_norm` |

Frame size comes from `sequence["frame_shapes"]` when present. If absent, `infer_frame_size()` falls back to bbox `x2, y2`, which is weaker and should not be used for production 54-dim training data. Missing bbox produces zeros in dims `51..53`; this is acceptable only for genuinely missing detections, not for upgrading 51-dim cached arrays.

## Schema Consistency Check

| Surface | Status | Evidence |
| :--- | :--- | :--- |
| Direct feature builder | PASS | `runs/preflight_keypoint_bbox54/feature_vector_54.log` shows shape `(30, 54)` and bbox dims `[0.05208333, 0.18518518, 0.00964506]`. |
| LSTM dummy inference | PASS | `runs/preflight_keypoint_bbox54/dummy_54_forward.log` shows dummy input `(1, 30, 54)` and output `(1, 2)`. |
| Real-time preflight | PASS | `runs/preflight_keypoint_bbox54/rtsp_preflight_54.log` shows `input_size=54`, `sequence_length=30`, `sequence_stride=15`, and `lstm_input_size=54`. |
| Existing 51-dim smoke | PASS | `runs/preflight_keypoint_bbox54/keypoint51_smoke.log` completed `Ran 15 tests ... OK`. |
| Training script | BLOCKED | `scripts/train_fight_vs_normal_lstm.py` has no `--input-size` or `--feature-schema`; it calls `collect_sequences(...)` with defaults, so it trains `keypoint51` only. |
| Evaluation script | RISK | `scripts/evaluate_retraining_manifest_v2.py` accepts `--input-size 54 --feature-schema keypoint_bbox54`, but cached 51-dim arrays are padded to 54 for bbox schema. |
| Dataset cache loader | RISK | `ai/action/fight_vs_normal_dataset.py` also pads 51-dim cached arrays for `keypoint_bbox54`. This violates the no-padding constraint. |

## Hard Negative Label Quality

Code sources:

- `scripts/export_hard_negative_candidates.py`
- `ai/learning/manifest_builders.py`
- `ai/learning/manifest_schema.py`
- `ai/learning/manifest_leakage.py`

Candidate generation maps false positives to:

- `label=0`
- `label_name=Normal`
- `source_type=hard_negative`
- `failure_type=false_positive`
- `review_status=pending` unless supplied

Observed risks:

- `CANDIDATE_FIELDS` includes `source_video` and `frame_id`, but not `start_frame`, `end_frame`, `fps`, `original_label`, or fall/faint interval fields.
- `build_error_candidates()` does not compare a candidate frame/window against original fall/faint intervals.
- Metadata merge is keyed only by `clip_id`; if FP rows do not match metadata rows, `source_video` and split information remain empty.
- Dry-run sample evidence shows `source_video_distribution={'<missing>': 1}`.
- `build_training_manifest()` includes only candidates with `review_status=approved`, which is good, but approval must be based on an interval-aware review artifact before merge.

Required before merge:

- Add candidate fields or a sidecar review table for `source_video`, `start_frame`, `end_frame`, `original_label`, `original_event_intervals`, and `interval_overlap_status`.
- Reject hard negatives when their frame range overlaps any fall/faint interval.
- Print and persist candidate count, label distribution, source video distribution, missing source video count, and interval-overlap rejects.
- Keep hard-negative addition ratio configurable, for example `0.05`, `0.10`, `0.20`, instead of a single all-in merge.

## Evaluation Structure

Available structure:

- `scripts/evaluate_retraining_manifest_v2.py` compares baseline vs retrained metrics and writes FP, FN, Precision, Recall, F1.
- `scripts/compare_lstm_feature_dims.py` compares two metrics JSON files for `keypoint51` vs `keypoint_bbox54`.
- `ai/evaluation/prediction_metrics.py` supports threshold sweeps that include threshold, consecutive count, cooldown, keypoint missing rate, and keypoint confidence.

Preflight dry-run output:

- `runs/preflight_keypoint_bbox54/evaluation/dry_run_eval.md`
- `runs/preflight_keypoint_bbox54/evaluation_dry_run.log`

The structure is ready, but real evaluation is blocked until a real manifest/data root exists locally or on the GPU server. The local workspace currently has no `strange_ai/data` directory.

## Runtime Config And Checkpoint Metadata

Runtime sources:

- `scripts/rtsp_inference_args.py`
- `scripts/run_rtsp_inference.py`
- `ai/action/classifier.py`
- `ai/action/lstm_contract.py`

Confirmed:

- `LSTMActionClassifier` reads `model_config.input_size`, `feature_schema_version`, `sequence_length`, and `sequence_stride` from checkpoint metadata.
- It rejects `feature_schema_version=keypoint_bbox54` when `input_size != 54`.
- `run_rtsp_inference.py --preflight-only` logs runtime `sequence_length`, `sequence_stride`, and classifier `input_size`.
- `log_lstm_config()` warns when checkpoint sequence length/stride differ from runtime values.

Gap:

- `feature_schema_version` itself is not printed by `log_lstm_config()`. The report relies on checkpoint construction plus `lstm_input_size=54`; adding schema to preflight output would make this audit stronger.

## Alert Policy Plan

Current defaults:

- Threshold: `DEFAULT_FAINT_THRESHOLD = 0.3`
- Consecutive condition: `DEFAULT_MIN_CONSECUTIVE_FAINT = 3`
- Cooldown: `DEFAULT_CAMERA_COOLDOWN_SECONDS = 10.0`

Current implementation:

- `FaintEventPostProcessor.should_trigger()` requires alert prediction, increments consecutive count per `(camera_id, track_id)`, and suppresses repeated camera alerts during cooldown.
- `ai/evaluation/prediction_metrics.py` can sweep `faint_confidence_threshold`, `consecutive_faint_count`, and `event_cooldown_seconds`.

Plan:

- Keep operational defaults unchanged.
- Evaluate candidate grids such as thresholds `0.25..0.70`, consecutive counts `1..5`, cooldowns `0, 5, 10, 20, 30`.
- Rank by F1 first, recall second, and lower FP third, then propose candidates only after real data evaluation.

## FrameId Binding

Code sources:

- `ai/frame_sync.py`
- `scripts/run_rtsp_inference.py`
- `ai/inference/rtsp_runtime.py`
- `tests/test_rtsp_event_payload.py`

Confirmed:

- Capture assigns a monotonically increasing per-camera `frame_id`.
- `FramePacket` carries `frame_id`, `captured_at_ms`, frame data, dimensions, `frame_idx`, and timestamp.
- Detection, keypoint sequence buffering, classification, payload, and evidence use the same `FramePacket` / `FrameMetadata`.
- Event payload includes top-level `frameId`, event-level `frameId`, sequence start/end frame IDs, `evidenceId`, and `traceId`.
- Smoke logs include payloads where top-level `frameId` and event `frameId` match.

Gap:

- `build_inference_event_log()` stores `frame_idx` and sequence window, but not `frame_id`. Add `frame_id` to event logs before using them as label-review source material.

## Verification Commands Run

Initial RED/import-shape check:

```powershell
python -m unittest tests.test_feature_schema_51_vs_54 tests.test_lstm_action_classifier tests.test_rtsp_inference_config tests.test_rtsp_inference tests.test_rtsp_event_payload
```

Result: failed before import because `tests` is not an importable package in that invocation. Evidence: `runs/preflight_keypoint_bbox54/unittest_smoke.log`.

Passing smoke checks:

```powershell
python -m unittest discover -s tests -p "test_feature_schema_51_vs_54.py"
python -m unittest discover -s tests -p "test_lstm_action_classifier.py"
python -m unittest discover -s tests -p "test_rtsp_inference*.py"
python -m unittest discover -s tests -p "test_rtsp_event_payload.py"
```

Evidence:

- `runs/preflight_keypoint_bbox54/feature_schema_test.log`
- `runs/preflight_keypoint_bbox54/keypoint51_smoke.log`
- `runs/preflight_keypoint_bbox54/rtsp_smoke.log`
- `runs/preflight_keypoint_bbox54/frame_payload_smoke.log`

54-dim checks:

```powershell
python -c "... LSTMActionModel(input_size=54); x=torch.randn(1,30,54); model(x) ..."
python scripts/inspect_lstm_feature_shape.py --feature-schema keypoint_bbox54 --expected-input-size 54 --sequence-length 30 --sample 1
python scripts/run_rtsp_inference.py --preflight-only --dry-run --detector-mode mock --action-model runs/preflight_keypoint_bbox54/mock_keypoint_bbox54.pt --action-device cpu --sequence-length 30 --sequence-stride 15 --publisher console
```

Hard-negative and evaluation dry-run:

```powershell
python scripts/export_hard_negative_candidates.py --sample --dry-run --output-dir runs/preflight_keypoint_bbox54/candidates
python scripts/evaluate_retraining_manifest_v2.py --dry-run --input-size 54 --feature-schema keypoint_bbox54 --output-dir runs/preflight_keypoint_bbox54/evaluation --report-path runs/preflight_keypoint_bbox54/evaluation/dry_run_eval.md
```

## Remaining Risks Before 54-Dim Deployment

1. Remove or block all `keypoint_bbox54` zero-padding from 51-dim cached arrays.
2. Add `--input-size` and `--feature-schema` to the primary training script, then save schema metadata consistently.
3. Require real bbox-backed 54-dim feature generation for training/evaluation caches.
4. Add interval-aware hard-negative validation before approving any FP candidate.
5. Add `frame_id` to event log JSON used for feedback/retraining.
6. Print `feature_schema_version` in real-time preflight logs.
7. Run full real-data evaluation on the GPU server with resumable output and low-memory fallbacks.

## Next Experiment Priority

1. Fix schema safety first: fail fast on `keypoint_bbox54` when source arrays are 51-dim and bbox data is unavailable.
2. Add hard-negative interval validation and distribution reports.
3. Run baseline `keypoint51` vs true bbox-backed `keypoint_bbox54` on the same fixed test split.
4. Run hard-negative ratio experiments at multiple ratios.
5. Sweep alert policy over threshold + consecutive + cooldown without changing production defaults.
