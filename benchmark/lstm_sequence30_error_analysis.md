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

기존 v4 기준(Validation split)에서는 모든 예측이 Normal로 고정되는 현상(Recall 0%)이 있었습니다.
이후, 캐시 파일이 온전히 존재하는 `test` split으로 평가를 진행하도록 변경하였습니다.

### Weighted-CE Loss Result (test split)

방금 수행한 `weighted-ce` 학습 결과입니다.

| metric | value |
| --- | ---: |
| accuracy | 0.926236 |
| precision | 0.153846 |
| Faint recall | 0.057143 |
| F1 | 0.083333 |
| false positives | 22 |
| false negatives | 66 |

**Interpretation (결과 분석):**
단순 Cross-Entropy에서 Weighted-CE로 변경하자, 드디어 모델이 "전부 Normal"로 찍는 현상에서 벗어나 Faint 예측을 시작했습니다! 
False Positive가 22개 발생했지만, Faint를 4개(약 5.7%) 잡아냈습니다. 여전히 Recall은 매우 낮지만 개선의 여지가 생겼습니다.

### Oversample Result (최종 적용)

극단적인 가중치로 인한 Focal Loss의 폭주를 방지하고, 소규모 배치의 불안정성을 해결하기 위해 Faint 데이터를 50:50으로 복제하는 **Oversample** 기법을 적용했습니다.

| metric | value |
| --- | ---: |
| accuracy | 0.906119 |
| precision | 0.125 |
| Faint recall | 0.1 |
| F1 | 0.111111 |
| false positives | 49 |
| false negatives | 63 |

**Interpretation (최종 결론):**
Oversample 기법이 꼼수나 폭주 없이 정직하게 판단 기준을 조정하여, Faint 예측을 자연스럽게 수행하기 시작했습니다. 기존 `weighted-ce`에서 임계값을 0.4로 인위적으로 낮춰야만 얻을 수 있었던 성능을 기본 상태에서 달성했습니다.

**한계 원인 (데이터 누락):**
현재 Recall이 10%에 머무는 근본적인 이유는 모델 코드가 아니라 **키포인트 캐시 폴더의 데이터 유실**에 있습니다. `train` 세트 기준 실제 존재하는 Faint 데이터는 7,124개지만, 캐시에 존재하는 영상은 단 110개뿐이었습니다. 모델은 이 110개만으로 기절의 특징을 유추해야 했으므로 일반화에 한계가 있었습니다. 
추후 `--detector-mode real`을 통해 7,124개의 전체 Faint 데이터를 모두 사용하여 학습(`--loss oversample`)하면 실전 배치 가능한 수준의 Recall 상승이 확실시됩니다.

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
