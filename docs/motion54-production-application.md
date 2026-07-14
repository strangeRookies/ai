# keypoint_motion54 LSTM production application

## Problem

Production runtime historically defaulted to **51-dim keypoint** features (`keypoint51`), while offline comparison experiments trained a stronger **54-dim motion** model (`keypoint_motion54`: 51 keypoints + `center_drop` / `velocity` / `torso_angle_norm`).

Without checkpoint schema metadata, a 54-dim weight file could not be loaded safely: runtime either mismatched dimensions or risked silent fallback.

## Selection rationale

Target experiment artifact (GPU):

```text
/home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best.pt
```

Reported comparison metrics for the motion54 lineage (from experiment notes; **not re-measured in this change**):

| Metric | Value |
|---|---|
| Accuracy | 93.45% |
| Precision | 92.80% |
| Recall | 94.20% |
| F1 | 93.49% |
| FP | 81 |
| FN | 66 |

**MODEL_MATCH = `ASSUMED_MATCH_WITH_EVIDENCE`**

Evidence used:

- Path contains `lstm_sequence30_motion_features` + `YOLO26n-pose`
- Feature math is the existing `ai/action/motion_features.append_motion_features()` implementation (PR #18 lineage / current tree)
- Sequence length 30 from experiment name; stride **15** recorded as `INFERRED_FROM_EXPERIMENT_DEFAULT=15` from `DEFAULT_LSTM_SEQUENCE_STRIDE` / `benchmark/compare_lstm_extractors.py` CLI default

Limitation: GPU host was **not reachable** from this worktree during packaging (SSH timeout). Local packaging was validated on a 54-dim Normal/Faint fixture with identical metadata rules; remote `best.pt` must be packaged on the GPU before cutover.

## Feature contract (`keypoint_motion54`)

```text
input shape = (batch, sequence_length, 54)
feature_schema_version = keypoint_motion54
```

| Index | Name | Definition |
|---:|---|---|
| 0–50 | `kp{i}_{x,y,conf}` | 17 keypoints × 3 |
| 51 | `center_drop` | `hip_mid_y[t] - hip_mid_y[t-1]` (0 at t=0) |
| 52 | `velocity` | Euclidean hip-midpoint step (0 at t=0) |
| 53 | `torso_angle_norm` | `(atan2(dy,dx) + π) / (2π)` shoulder→hip |

No timestamp-based velocity/acceleration in this release (`keypoint_motion54_v2` is out of scope).

### Schema separation

| Schema | Dims | Builder |
|---|---:|---|
| `keypoint51` | 51 | keypoints only |
| `keypoint_motion54` | 54 | keypoints + `append_motion_features()` |
| `keypoint_bbox54` | 54 | keypoints + bbox width/height/area norms |

Fail-fast: missing/unknown schema for size 54, schema↔dim mismatch, feature_names length mismatch, ImportError on motion module (no silent 54→51 fallback), packaging of `keypoint_bbox54` as motion54.

## Packaging

Script:

```text
scripts/package_motion54_checkpoint.py
```

Example (GPU):

```bash
python scripts/package_motion54_checkpoint.py \
  --input /home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best.pt \
  --output /home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best_motion54_packaged.pt \
  --sequence-length 30 \
  --dry-run

python scripts/package_motion54_checkpoint.py \
  --input .../best.pt \
  --output .../best_motion54_packaged.pt \
  --sequence-length 30
```

Behavior:

1. Load original dict checkpoint  
2. Require `model_config.input_size == 54`, classes `Normal`/`Faint`  
3. Reject `keypoint_bbox54`  
4. Write **new** file with schema metadata  
5. Reload and verify every `model_state` tensor with `torch.equal`  
6. Never overwrite input  

### Local packaging verification (this PR)

| Item | Value |
|---|---|
| Fixture raw | `runs/motion54_packaging_local/best_raw_motion54_fixture.pt` |
| Packaged | `runs/motion54_packaging_local/best_motion54_packaged.pt` |
| model_state identical | **true** (unit + CLI) |
| original unchanged | **true** |

Remote original/packaged SHA256: fill on GPU after real package (`checkpoint_inspect.json` currently records SSH unreachable).

## Runtime wiring

Workers already accept:

```text
ACTION_MODEL / MODEL_CHECKPOINT_PATH / --action-model
```

See `scripts/rtsp_inference_args.py`, `scripts/run_registered_cameras.py`, `scripts/serve_ai_overlay.py`.

Example env (also in `AI_LOCAL_CONFIG.example.bat`):

```bat
set "ACTION_MODEL=/home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best_motion54_packaged.pt"
set "MODEL_CHECKPOINT_PATH=%ACTION_MODEL%"
```

Tracker production defaults remain:

```text
new_track_thresh=0.30
near_dup=hybrid_kp
```

### Startup / first-inference logs

```text
[lstm-contract] cameraLoginId=... checkpoint=... input_size=54 feature_schema=keypoint_motion54 feature_names_count=54 sequence_length=30 sequence_stride=15 classes=Normal,Faint device=...
[lstm-runtime] cameraLoginId=... trackId=... tensor_shape=(1,30,54) finite=true center_drop_range=... velocity_range=... torso_angle_range=...
```

`[lstm-runtime]` is emitted **once** per classifier instance.

## Regression tests

```text
python -m py_compile ai/action/feature_schema.py ai/action/classifier.py ai/action/lstm_contract.py ai/action/motion_features.py ai/inference/rtsp_runtime.py scripts/package_motion54_checkpoint.py
python -m unittest tests.test_motion54_packaging_and_runtime tests.test_feature_schema_51_vs_54 tests.test_lstm_action_classifier tests.test_session_boundary_contract
git diff --check
```

Covered:

- Motion dim order and first-frame zeros  
- Down / still / horizontal cases  
- Shape (30,51)→(30,54)  
- Packaging weight identity  
- bbox54 packaging reject  
- Schema load contract + `[lstm-contract]` line  

## Rollback

1. Point `ACTION_MODEL` back to previous production checkpoint (prior 51-dim or previous packaged path).  
2. Restart camera workers / overlay processes.  
3. Confirm `[lstm-contract]` shows the previous `input_size` / schema.  

Immediate rollback triggers: load failure, shape ≠ 54, schema mismatch, NaN/Inf, constant probabilities, worker crash loops, MQTT regression.

## GPU apply checklist (remote resume)

```bash
# 1) record current workers + ACTION_MODEL
ps -ef | grep -E 'serve_ai_overlay|run_registered|run_rtsp' | grep -v grep
echo "$ACTION_MODEL"

# 2) pull branch feature/incident-recovery-resume-2
cd /path/to/ai && git fetch && git checkout feature/incident-recovery-resume-2 && git pull

# 3) package
IN=/home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best.pt
OUT=/home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best_motion54_packaged.pt
python scripts/package_motion54_checkpoint.py --input "$IN" --output "$OUT" --sequence-length 30 --dry-run
python scripts/package_motion54_checkpoint.py --input "$IN" --output "$OUT" --sequence-length 30

# 4) export ACTION_MODEL=$OUT and restart workers
# 5) confirm [lstm-contract] + first [lstm-runtime]
# 6) short clip regression (E03_001 segment) without changing tracker thresholds
```

## Alert E2E

**ALERT_FLOW = `NOT_TESTED`** in this worktree (no live MQTT/backend/frontend stack exercised here).

## Final enums

| Enum | Value |
|---|---|
| MODEL_MATCH | `ASSUMED_MATCH_WITH_EVIDENCE` |
| PACKAGING | `PASS` (local fixture + script); GPU original package pending host access |
| RUNTIME | `PASS_WITH_LIMITATIONS` (code path + unit load verified; live multi-camera not run) |
| ALERT_FLOW | `NOT_TESTED` |
| RETRAINING | `NOT_REQUIRED` |

## Known limitations

- GPU `best.pt` not inspected live (SSH timeout).  
- Sequence stride not embedded in legacy checkpoints → recorded as experiment default 15.  
- Full detect→track→MQTT→DB→frontend chain not run here.  
- Motion formulas remain frame-diff based (not timestamp-normalized).  
