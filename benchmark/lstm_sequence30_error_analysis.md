# YOLO26n-pose + LSTM Sequence 30 Error Analysis

## Final Baseline

Use `sequence_length=30` as the fixed baseline for the next Faint-detection improvement loop. Keep the existing YOLO26n-pose detector and the current LSTM keypoint input path unchanged:

- detector: `YOLO26n-pose`
- classifier: existing LSTM
- sequence length: `30`
- input feature contract: 17 keypoints x `(x, y, confidence)` = 51 features per frame
- output directory: create a new directory for each run; do not overwrite previous benchmark results

The sequence 30 result should be interpreted as the stability-first baseline. Shorter windows can remain historical ablation data, but FN/FP analysis and retraining candidates should be anchored to sequence 30.

## Current Sequence 30 Result

Current confirmed run:

```text
benchmark/results/lstm_sequence_length_30_yolo26n_cache_fixed_v4/YOLO26n-pose/summary.json
```

Validation split:

| metric | value |
| --- | ---: |
| eval clips | 789 |
| eval Normal sequences | 753 |
| eval Faint sequences | 36 |
| generated sequences | 789 |
| zero sequence clips | 0 |
| keypoint missing rate | 0.178026 |
| accuracy | 0.954373 |
| precision | 0.0 |
| Faint recall | 0.0 |
| F1 | 0.0 |
| false positives | 0 |
| false negatives | 36 |

Confusion matrix:

```text
             predicted Normal   predicted Faint
true Normal        753                 0
true Faint          36                 0
```

Interpretation: this sequence 30 baseline is conservative and currently predicts every validation sample as Normal. The immediate improvement target is not FP reduction, but recovering Faint recall without breaking the existing zero-FP behavior too aggressively.

### Error Cause Diagnosis (Execution Results)
1. **Model Confidence (`faint_prob`)**: The FN samples exhibit extremely low `faint_prob` values (ranging from `0.038` to `0.159`). The model is not just missing the threshold; it is overwhelmingly confident that the faint clips are Normal.
2. **Class Imbalance**: The `summary.json` shows an extreme class imbalance.
   - Train: Normal 3010 vs Faint 144 (~21:1)
   - Eval: Normal 753 vs Faint 36 (~21:1)
   The model is likely collapsing to a local minimum where predicting the majority class (Normal) trivially yields ~95.4% accuracy.
3. **Cache / Pipeline Health**: Diagnostic runs on `train_clips.json` confirm `Total Faint Train Clips: 144` and `Zero Sequence Faints: 0`. The keypoint caching and 30-frame sequence extraction pipeline is perfectly healthy. The "All-Normal" prediction is entirely an algorithmic/learning issue, not a data loading issue.

## FN/FP Extraction

After a sequence 30 run writes `eval_predictions.csv`, export error samples with:

```bash
python scripts/extract_lstm_fp_fn.py \
  --predictions benchmark/results/lstm_sequence_length_30_yolo26n_cache_fixed_v4/YOLO26n-pose/eval_predictions.csv \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --output-dir benchmark/results/lstm_sequence_length_30_fp_fn \
  --sequence-length 30
```

Optional threshold-based audit:

```bash
python scripts/extract_lstm_fp_fn.py \
  --run-dir benchmark/results/lstm_sequence_length_30_yolo26n_cache_fixed_v4 \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --output-dir benchmark/results/lstm_sequence_length_30_fp_fn_threshold_0_4 \
  --sequence-length 30 \
  --threshold 0.4
```

Outputs:

- `false_negatives.csv`: true Faint predicted as Normal
- `false_positives.csv`: true Normal predicted as Faint
- `fp_fn_summary.json`: counts and input paths

Confirmed output:

```text
benchmark/results/lstm_sequence_length_30_fp_fn/false_negatives.csv  # 36 rows
benchmark/results/lstm_sequence_length_30_fp_fn/false_positives.csv  # 0 rows
benchmark/results/lstm_sequence_length_30_fp_fn/fp_fn_summary.json
```

Existing overlay/debug helpers are available in `scripts/run_rtsp_inference.py --overlay-output` and `benchmark/diagnose_fall_candidates.py`, but this step only exports CSV error candidates. Use those helpers later for selected `clip_id` rows after the FN/FP list is reviewed.

## Motion Feature Design

Do not connect motion features to training until the 51-dim keypoint baseline and error CSVs are stable. The next feature draft should preserve the existing 51-dim path and add a separate experimental feature branch or explicit feature flag.

Candidate motion features:

- `center_drop`: vertical movement of body/keypoint center across the sequence
- `torso_angle`: shoulder-to-hip orientation, including angle delta over time
- `bbox_aspect_ratio`: person box width / height per frame
- `bbox_area_change`: relative bbox area change across the sequence
- `keypoint_velocity`: per-keypoint first-order frame-to-frame displacement
- `acceleration`: per-keypoint second-order displacement change
- `avg_keypoint_confidence`: average confidence across visible keypoints
- `missing_keypoint_ratio`: missing or below-threshold keypoint count / total keypoints

Future model families such as GRU, TCN, ST-GCN, and PoseC3D are out of scope for this step. Treat them as later comparison candidates after the sequence 30 LSTM baseline has stable FN/FP evidence.

## Class Imbalance Mitigation (Weighted Loss)

**Status**: 🚧 실행 예정 (Pending Execution on Server)

Based on the extreme 21:1 class imbalance (Normal: 3010, Faint: 144), two loss function adjustments have been implemented to penalize the model heavily for misclassifying the minority `Faint` class.

- **Weighted CrossEntropy (`--loss weighted-ce`)**: Re-weights the standard CrossEntropy Loss inversely proportional to class frequencies.
- **Focal Loss (`--loss focal`)**: Applies `gamma=2.0` down-weighting to easily classified `Normal` samples, forcing the model to focus on the hard-to-learn `Faint` cases.

### Execution Commands (Remote Server)

To run the training with the new Weighted CE loss, without overwriting previous baseline results, run:

```bash
python scripts/run_lstm_sequence_length_comparison.py \
  --output-dir benchmark/results/lstm_sequence_length_30_yolo26n_weighted_loss_v1 \
  --detector-mode cache \
  --keypoint-cache-dir ../ai_fall_experiments/data/keypoints/yolo26n-pose \
  --device cuda:0 \
  --epochs 5 \
  --loss weighted-ce
```

To run with Focal Loss:
```bash
python scripts/run_lstm_sequence_length_comparison.py \
  --output-dir benchmark/results/lstm_sequence_length_30_yolo26n_focal_loss_v1 \
  --detector-mode cache \
  --keypoint-cache-dir ../ai_fall_experiments/data/keypoints/yolo26n-pose \
  --device cuda:0 \
  --epochs 5 \
  --loss focal
```

### Verification & Logging Commands

After training, verify the `summary.json` for Faint recall and F1 scores:
```bash
cat benchmark/results/lstm_sequence_length_30_yolo26n_weighted_loss_v1/sequence_length_30/YOLO26n-pose/summary.json | grep -A 10 '"lstm_metrics"'
```

### Experiment Results (Weighted CE vs Baseline)

| metric | Baseline (CE) | Weighted CE | Focal Loss |
| --- | --- | --- | --- |
| Accuracy | 0.954373 | - | - |
| Precision | 0.0 | - | - |
| Faint recall | 0.0 | - | - |
| F1 score | 0.0 | - | - |
| False Positives | 0 | - | - |
| False Negatives | 36 | - | - |
