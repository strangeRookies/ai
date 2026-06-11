# AI 학습/전처리 설명 정리

이 문서는 현재 repository에서 직접 확인한 코드, 문서, 설정, 결과 파일 기준으로 작성했다. 실제 GPU PC 실행 기준 경로는 다음과 같다.

```bash
cd ~/yolo_training/strange_ai_lstm
```

## 1. 전체 AI 파이프라인 개요

| 단계 | 현재 확인한 구현 | 근거 파일 |
| --- | --- | --- |
| RTSP/영상 입력 | RTSP URL 또는 영상 파일을 `VideoReader`로 읽고 frame 단위로 처리한다. | `scripts/run_rtsp_inference.py`, `ai/streams/video_reader.py` |
| YOLO26n-pose 사람/관절 추출 | 운영 RTSP 경로는 `YoloPoseDetector`가 bbox, confidence, keypoints, keypoint confidence를 반환한다. 기본 YOLO 모델은 `yolo26n-pose.pt`다. | `detector/yolo_pose_detector.py`, `scripts/rtsp_inference_args.py` |
| track_id 처리 | `--tracking-mode supervision` 또는 `TRACKING_MODE=supervision`을 사용하면 `supervision.ByteTrack()`을 사용한다. 기본 `auto`는 `ENABLE_SUPERVISION_POSTPROCESSING=true`일 때 supervision, 아니면 `SimpleTrackAssigner` fallback을 사용한다. | `ai/inference/rtsp_runtime.py`, `tracking/simple_tracker.py`, `ai/postprocess/supervision_postprocessor.py` |
| Sequence Buffer | RTSP 운영 경로는 track별 keypoint/crop sequence buffer를 유지한다. 기본 입력은 keypoints다. | `ai/action/per_track_sequence_buffer.py`, `ai/action/keypoint_sequence_buffer.py`, `ai/action/sequence_buffer.py` |
| LSTM 행동 분류 | `LSTMActionModel`은 PyTorch `nn.LSTM + Linear head` 구조다. `LSTMActionClassifier`가 checkpoint를 로드해 Normal/Faint 확률을 만든다. | `ai/action/classifier.py` |
| Faint/Normal 판단 | `Faint` 확률이 threshold 이상이면 `Faint`, 아니면 `Normal`로 판단한다. 기본 threshold는 `0.3`이다. | `ai/action/classifier.py`, `ai/action/faint_post_processing.py` |
| 이벤트 후처리 | 같은 `camera_id + track_id`에서 Faint가 설정 횟수 이상 연속 발생해야 이벤트를 낸다. 기본은 3회 연속, cooldown 10초다. | `ai/action/faint_post_processing.py` |
| MQTT 이벤트 발행 | `--publisher mqtt`이면 `MqttEventPublisher`가 `safety/events` topic으로 JSON payload를 발행한다. `--dry-run` 또는 console 모드는 콘솔 출력이다. | `ai/publishers/event_publisher.py`, `scripts/run_rtsp_inference.py` |

### 중요한 구분

현재 코드에는 LSTM 입력 방식이 두 갈래로 존재한다.

| 구분 | 입력 feature | 설명 |
| --- | --- | --- |
| `ai/action/train_lstm.py` | 사람 crop 이미지 기반 feature | 학습용 crop을 grayscale resize 후 flatten한다. YOLO pose keypoint 학습이 아니라 person box crop 기반이다. |
| `benchmark/compare_lstm_extractors.py` 및 RTSP 운영 기본 | YOLO pose keypoint 기반 feature | YOLO26n-pose, YOLOv11n-pose 등을 비교하고 keypoint sequence로 LSTM을 학습/평가한다. |

따라서 멘토님께 설명할 때는 “현재 운영 파이프라인은 YOLO26n-pose keypoint sequence를 LSTM에 넣는 구조이고, 별도 `train_lstm.py`는 crop 기반 학습 스크립트로 남아 있다”고 구분해서 말하는 것이 안전하다.

## 2. 데이터 전처리 방식

### 2.1 `ai/action/train_lstm.py` 기준 입력

학습 진입점은 다음이다.

```bash
python -m ai.action.train_lstm --dataset-csv <CSV_PATH>
```

CSV row는 `load_training_rows()`에서 읽는다.

| 컬럼 | 사용 방식 | 근거 |
| --- | --- | --- |
| `video_path` 또는 `clip_path` | 영상 파일 경로. 둘 중 하나가 없으면 row를 건너뛴다. | `ai/action/train_lstm.py:load_training_rows` |
| `clip_id` | 없으면 영상 stem을 사용한다. | `ai/action/train_lstm.py:load_training_rows` |
| `label` | 있으면 `1=Faint`, 그 외 `Normal`로 사용한다. | `row_is_active()`, `row_label_name()` |
| `label_name` | 있으면 표시/메타데이터용 label 이름으로 우선 사용한다. | `row_label_name()` |
| `event_class` | `label_name`이 없을 때 label 이름으로 사용한다. | `row_label_name()` |
| `label_path` 또는 `annotation_path` | JSON event label이 있으면 event frame range를 읽는다. | `load_row_label()`, `ai/labels/event_label_loader.py` |
| `start_frame`, `end_frame` | Faint row에서 event range로 사용한다. | `target_ranges_for_row()` |
| `split` | `--train-split`, `--val-split` 필터에 사용한다. | `load_training_rows()` |

`label_path` JSON은 `metadata.file_name`, `metadata.frame_count`, `annotations.event_class`, `annotations.event_frame`을 읽는다. `event_frame`은 `(start, end)` frame 범위 목록이다.

### 2.2 sequence 생성 방식

`collect_sequences()`는 split별 row를 돌면서 영상을 읽고, target frame range에 해당하는 frame만 처리한다.

1. `VideoReader`로 frame을 읽는다.
2. `detect_person()`으로 사람 bbox를 찾는다.
3. bbox가 있으면 `CropSequenceBuffer`에 frame crop을 넣는다.
4. `sequence_length`만큼 쌓이고, 마지막 emit 이후 `sequence_stride` 이상 진행되면 sequence를 만든다.
5. `crops_to_features()`로 crop sequence를 LSTM 입력 feature로 바꾼다.

`CropSequenceBuffer`는 가장 큰 bbox 하나를 선택하고, 해당 영역을 `resize_size x resize_size`로 resize한다. 이후 `crops_to_features()`는 각 crop을 grayscale로 바꾼 뒤 `feature_size x feature_size`로 줄이고 flatten한다.

### 2.3 fallback full-frame crop

`--fallback-full-frame` 기본값은 enabled다. 사람 bbox가 없을 때:

| 조건 | 동작 |
| --- | --- |
| `fallback_full_frame=True`이고 bbox 없음 | frame 전체를 fallback bbox로 사용한다. `crop_source=fallback_full_frame` |
| `fallback_full_frame=False`이고 bbox 없음 | 해당 frame은 skip된다. |

`preprocess_summary_train.json`, `preprocess_summary_val.json`에는 `fallback_ratio`가 저장된다. `fallback_ratio`가 높다는 것은 사람이 검출되지 않아 전체 frame crop으로 학습된 sequence 비율이 높다는 뜻이다. `detector-mode none`에서 fallback을 켜면 bbox가 항상 없으므로 fallback 비율이 1.0에 가까워질 수 있다. 이 경우 모델이 “사람 crop”이 아니라 “전체 화면”을 많이 학습하므로 신뢰도가 낮아진다.

### 2.4 `detector-mode` 차이

`ai/action/train_lstm.py` 기준:

| 옵션 | 동작 |
| --- | --- |
| `--detector-mode mock` | 고정 mock person box를 생성한다. 빠른 구조 검증용이다. |
| `--detector-mode yolo` | `YoloPersonDetector`가 YOLO person class bbox를 검출한다. bbox가 없으면 `--yolo-retry-conf`로 한 번 더 낮은 confidence 재시도 가능하다. |
| `--detector-mode none` | 항상 bbox 없음으로 처리한다. fallback이 켜져 있으면 full-frame crop만 생성된다. |

운영 RTSP 경로의 `scripts/run_rtsp_inference.py`는 `--detector-mode real|mock`을 사용하고, real일 때 `YoloPoseDetector`를 사용한다. 즉 학습 스크립트의 `yolo`와 RTSP 경로의 `real`은 옵션 이름이 다르다.

### 2.5 `--dry-run-preprocess`

`--dry-run-preprocess`는 `collect_sequences()`까지 실행하고 학습은 하지 않는다. 확인 가능한 것:

- train/val row가 정상적으로 읽히는지
- 영상 decode가 되는지
- detector가 sequence를 만들 수 있는지
- fallback ratio가 높은지
- `preprocess_sequences_*.csv`, `preprocess_clips_*.json`, `preprocess_summary_*.json`이 생성되는지

## 3. 학습 방식

### 3.1 `ai/action/train_lstm.py`

주요 옵션:

| 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `--sequence-length` | 16 | sequence frame 수 |
| `--sequence-stride` | 8 | sequence emit 간격 |
| `--resize-size` | 224 | person crop resize 크기 |
| `--feature-size` | 32 | LSTM feature 변환용 grayscale 크기 |
| `--hidden-size` | 128 | LSTM hidden size |
| `--num-layers` | 1 | LSTM layer 수 |
| `--dropout` | 0.0 | LSTM dropout. 단, PyTorch LSTM은 layer가 1이면 dropout이 0으로 처리된다. |
| `--batch-size` | 32 | batch size |
| `--epochs` | 20 | epoch 수 |
| `--lr` | 0.001 | AdamW learning rate |
| `--device` | auto | `auto`면 CUDA 가능 시 `cuda:0`, 아니면 CPU |

학습 흐름:

1. `args = parser.parse_args()`
2. CUDA/cuDNN 진단 로그 출력
3. train/val CSV row 로드
4. `collect_sequences()`로 `train_x`, `train_y`, `val_x`, `val_y` 생성
5. `TensorDataset(torch.from_numpy(...))` 생성
6. epoch loop에서 LSTM 학습
7. validation accuracy가 가장 높을 때 `best.pt` 저장
8. `history.json` 저장

`best.pt` 저장 기준은 `val_acc > best_acc`다. 저장 내용은 `model_state`, `model_config`, `classes`, `feature_size`, `sequence_length`, `best_val_acc`, `preprocess_summary`다.

현재 `classes`는 `["Normal", "Fall"]`로 저장된다. 반면 운영 classifier의 threshold 로직은 `Faint`와 `Normal` class가 있을 때만 `faint_threshold`를 적용한다. 이 불일치는 현재 코드 기준 확인 필요 항목이다.

### 3.2 CUDA/cuDNN 진단 로그

최근 추가된 `print_cuda_diagnostics()`는 `ai/action/train_lstm.py`와 `strange_ai/ai/action/train_lstm.py`에 있다. 호출 위치는 `main()`에서 `args = parser.parse_args()` 직후, 전처리/학습 시작 전이다.

확인 항목:

- PyTorch version
- CUDA 사용 가능 여부
- torch CUDA version
- cuDNN enabled 여부
- cuDNN version
- CUDA device count
- 각 GPU name/memory/capability
- 현재 CUDA device
- test tensor device
- 사용자가 요청한 `args.device`

GPU PC 실행 예:

```bash
cd ~/yolo_training/strange_ai_lstm

python -m ai.action.train_lstm \
  --dataset-csv "data/retrain_candidates/yolo26n_lstm_error_based/merged/dataset_error_augmented_clips.csv" \
  --dry-run-preprocess \
  --device cuda:0
```

위 CSV 경로는 사용자 요청 예시 경로이며, 현재 로컬 repository에서는 해당 파일이 확인되지 않았다. GPU PC에서 실제 존재 여부 확인이 필요하다.

### 3.3 YOLO26n-pose benchmark/LSTM 학습 경로

YOLO pose keypoint 기반 LSTM 비교/학습은 `benchmark/compare_lstm_extractors.py`가 담당한다. wrapper는 `scripts/run_yolo26n_final_lstm.sh`다.

주요 특징:

- 기본 metadata CSV: `data/splits/final_source_video_split/all.csv`
- 기본 pose model: `yolo26n-pose.pt`
- 기본 비교 모델: `YOLO26n-pose:${POSE_MODEL}`
- mode: `dry-run`, `sequences`, `train`, `audit`
- `audit` mode는 `scripts/audit_lstm_thresholds.py`도 실행한다.
- keypoint feature는 17개 keypoint의 x/y/confidence를 정규화하여 사용한다.

GPU PC 실행 예:

```bash
cd ~/yolo_training/strange_ai_lstm

bash scripts/run_yolo26n_final_lstm.sh dry-run
bash scripts/run_yolo26n_final_lstm.sh train
bash scripts/run_yolo26n_final_lstm.sh audit
```

최종 서비스 검증용으로 chromakey 제외 CSV를 쓰려면:

```bash
cd ~/yolo_training/strange_ai_lstm

METADATA_CSV="data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv" \
bash scripts/run_yolo26n_final_lstm.sh audit
```

## 4. 판단 기준

### 4.1 LSTM 출력

`LSTMActionClassifier.predict()`는 다음 순서로 판단한다.

1. sequence를 feature로 변환한다.
2. LSTM logits를 계산한다.
3. softmax로 class별 확률을 만든다.
4. `probabilities = {"Normal": ..., "Faint": ...}` 형태로 저장한다.
5. `threshold_prediction()`에서 `Faint` 확률이 threshold 이상이면 `Faint`, 아니면 `Normal`로 판단한다.

단, checkpoint의 class 이름에 `Faint`가 없으면 threshold 로직은 적용되지 않고 argmax 결과를 사용한다.

### 4.2 threshold 0.3 / 0.4 / 0.5 의미

`scripts/audit_lstm_thresholds.py`와 `benchmark/compare_lstm_extractors.py`는 threshold별로 Faint 판정 기준을 바꿔본다.

| threshold | 의미 | 일반적 경향 |
| ---: | --- | --- |
| 0.3 | Faint 확률이 30% 이상이면 Faint | 민감하게 잡아 미탐(FN)을 줄이지만 오탐(FP)이 늘 수 있음 |
| 0.4 | 중간 후보 | recall과 F1 균형 후보 |
| 0.5 | 더 보수적인 기준 | 오탐은 줄 수 있지만 미탐이 늘 수 있음 |

`scripts/audit_lstm_thresholds.py`의 추천 기준은 “Faint recall 우선, 그 다음 F1, 그 다음 false positive 수 감소”다.

### 4.3 관제 시스템에서 우선 볼 지표

안전 관제에서는 실신/이상 상황을 놓치지 않는 것이 중요하므로 Faint recall을 우선 본다. 다만 threshold를 너무 낮추면 Normal이 Faint로 많이 잡혀 관제 피로도가 올라가므로 F1-score와 false positive 수를 함께 본다.

| 용어 | 의미 |
| --- | --- |
| TP | 실제 Faint를 Faint로 맞춘 경우 |
| FP | 실제 Normal인데 Faint로 잘못 울린 경우. 오탐 |
| FN | 실제 Faint인데 Normal로 놓친 경우. 미탐 |
| TN | 실제 Normal을 Normal로 맞춘 경우 |

`ai/evaluation/prediction_log.py`와 `ai/evaluation/prediction_metrics.py`에서 FP/FN/TN/TP를 계산한다.

### 4.4 Hard Negative / Faint Reinforcement

현재 코드에서 확인한 정의:

- `hard_negative` 폴더는 평가에서 Normal ground truth로 취급된다.
- `samples/evaluation/normal_basic`, `samples/evaluation/faint`, `samples/evaluation/hard_negative`, `samples/evaluation/rtsp_real` 구조가 문서와 코드에 있다.
- `rtsp_real`은 ground truth가 없으면 metric에서 제외되는 모니터링 데이터로 취급된다.

현재 코드에서 확인 필요:

- Hard Negative 데이터를 자동으로 dataset CSV에 병합하는 스크립트는 현재 root `scripts/`에서 확인하지 못했다.
- Faint Reinforcement라는 명칭의 별도 구현/스크립트는 현재 코드에서 확인하지 못했다.
- `data/retrain_candidates/yolo26n_lstm_error_based/merged/dataset_error_augmented_clips.csv`는 사용자 요청 예시에는 있으나 현재 로컬 checkout에는 없다.

## 5. 성능 개선 과정

### 5.1 현재 repository에서 확인된 baseline/result 파일

| 파일/경로 | 현재 상태 | 의미 |
| --- | --- | --- |
| `benchmark/results/model_benchmark.csv` | 존재 | pose-only benchmark 결과. 현재 sample video가 없어 모든 모델이 skipped |
| `benchmark/results/model_benchmark.md` | 존재 | 위 CSV의 Markdown 보고서 |
| `benchmark/results/lstm_final_11n_vs_26n_workflow.patch` | 존재 | LSTM workflow 관련 patch 파일. 실제 result directory는 아님 |
| `benchmark/results/lstm_final_11n_vs_26n/` | 현재 로컬에는 없음 | `PROJECT_SUMMARY.md`에서 기대 출력 경로로 언급됨 |
| `benchmark/results/lstm_yolo26n_final_split_test_audit/` | 현재 로컬에는 없음 | threshold audit 결과 경로로 문서에 언급됨 |
| `benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/` | 현재 로컬에는 없음 | 기본 ACTION_MODEL 경로로 코드/문서에 언급됨 |
| `runs/action_lstm/` | 현재 로컬에는 비어 있음 | `train_lstm.py` 기본 출력 위치 |

현재 로컬에서 실제 수치가 있는 benchmark artifact는 `model_benchmark.*`뿐이며, 이 결과는 “sample_videos folder is empty”로 전부 skipped다. 따라서 실제 LSTM 성능 수치는 현재 로컬 result 파일로는 검증할 수 없다.

### 5.2 문서에 기록된 성능 수치

`PROJECT_SUMMARY.md`와 `docs/MODEL_BENCHMARK_REPORT.md`에는 다음 내용이 기록되어 있다. 다만 대응하는 raw result directory가 현재 로컬에는 없어 “문서상 기록”으로만 취급한다.

| 항목 | 문서상 기록 | 확인 상태 |
| --- | --- | --- |
| 4-camera 3000-frame test | 각 camera 약 29.7 FPS, YOLO26n-pose 6.1~6.5 ms/frame, LSTM 0.4 ms/frame | 문서상 기록, raw summary 파일 현재 없음 |
| 1000/class benchmark threshold 0.3 | Faint recall 0.784553, F1 0.665661 | 문서상 기록, raw CSV 현재 없음 |
| 1000/class benchmark threshold 0.5 | Faint recall 0.586382, F1 0.617608 | 문서상 기록, raw CSV 현재 없음 |
| repeated-seed mean | Faint recall 0.658198, F1 0.648263 | 문서상 기록, raw JSON 현재 없음 |

### 5.3 FP/FN 분석과 error-augmented dataset

현재 코드에서 확인된 흐름:

1. RTSP inference 중 `--evaluation-log`를 지정하면 prediction JSONL을 저장한다.
2. `build_prediction_log_row()`가 prediction, confidence, ground_truth, result_type, keypoint 품질, consecutive count, cooldown 여부를 기록한다.
3. `scripts/evaluate_prediction_logs.py`가 `samples/evaluation/` 아래의 JSON/JSONL을 읽어 confusion matrix와 threshold sweep을 만든다.
4. `hard_negative` 폴더는 Normal로 평가된다.

현재 구현 없음/확인 필요:

- FP/FN row를 자동으로 골라 `dataset_error_augmented_clips.csv`를 생성하는 스크립트는 현재 checkout에서 찾지 못했다.
- Hard Negative Normal 데이터와 Faint 보강 데이터를 실제로 어떤 CSV에 추가했는지 확인할 raw 파일이 현재 로컬에는 없다.
- error-augmented checkpoint 경로는 `ai/action/faint_post_processing.py`의 기본값과 `docs/RTSP_LSTM_ERROR_AUGMENTED_DEPLOYMENT.md`에 있지만, 해당 `best.pt` 파일은 현재 로컬에는 없다.

### 5.4 fallback_ratio 개선 해석

`train_lstm.py`의 `preprocess_summary_*.json`에는 다음 값이 저장된다.

- `sequences_using_yolo_person_boxes`
- `sequences_using_fallback_full_frame_crops`
- `fallback_ratio`

해석:

- `detector-mode none` + fallback enabled이면 사람 bbox가 없으므로 full-frame crop 비율이 매우 높아진다.
- `detector-mode yolo`로 바꾸면 실제 사람 bbox를 사용할 수 있어 fallback ratio가 낮아지는 것이 정상이다.
- 현재 로컬에는 `preprocess_summary_train.json`, `preprocess_summary_val.json` 실제 결과 파일이 없어 구체 수치는 확인 필요다.

### 5.5 epoch 1 이후 과적합 여부

`history.json`이 있으면 epoch별 `train_loss`, `val_acc`를 보고 과적합 여부를 판단할 수 있다. 현재 로컬에는 실제 `history.json` 결과 파일이 없어 epoch 1 이후 과적합 경향은 확인 필요다.

## 6. 결과 파일 역할

| 파일 | 생성 스크립트 | 역할 | 현재 로컬 존재 |
| --- | --- | --- | --- |
| `history.json` | `ai/action/train_lstm.py`, `benchmark/compare_lstm_extractors.py` | epoch별 loss/validation metric 기록 | 없음 |
| `best.pt` | 학습 스크립트 | 가장 좋은 validation 결과 checkpoint | 없음 |
| `preprocess_summary_train.json` | `ai/action/train_lstm.py` | train split preprocessing 요약, fallback ratio 포함 | 없음 |
| `preprocess_summary_val.json` | `ai/action/train_lstm.py` | val split preprocessing 요약, fallback ratio 포함 | 없음 |
| `preprocess_sequences_*.csv` | `ai/action/train_lstm.py` | 생성된 sequence별 메타데이터 | 없음 |
| `preprocess_clips_*.json` | `ai/action/train_lstm.py` | clip별 preprocessing 결과 | 없음 |
| `summary.json` | `benchmark/compare_lstm_extractors.py`, RTSP inference output | benchmark/run 요약 | 없음 |
| `confusion_matrix.csv` | `benchmark/compare_lstm_extractors.py` | Normal/Faint confusion matrix | 없음 |
| `eval_predictions.csv` | `benchmark/compare_lstm_extractors.py` | sequence별 true/pred label, normal_prob, faint_prob | 없음 |
| `threshold_audit.csv` | `benchmark/compare_lstm_extractors.py`, `scripts/audit_lstm_thresholds.py` | threshold별 recall/F1/precision 등 | 없음 |
| `threshold_audit.json/md` | `scripts/audit_lstm_thresholds.py` | 추천 threshold와 선택 기준 | 없음 |
| `model_benchmark.csv/md` | `benchmark/benchmark_models.py` | pose-only benchmark 결과 | 있음. 단 sample video 없음으로 skipped |

## 7. 앞으로 개선 방향

| 방향 | 왜 필요한가 | 현재 코드 연결점 |
| --- | --- | --- |
| threshold 튜닝 | Faint recall과 오탐 균형을 맞추기 위해 필요 | `scripts/audit_lstm_thresholds.py`, `scripts/evaluate_prediction_logs.py` |
| consecutive count 적용 | 순간 오탐 하나로 이벤트가 나가지 않게 함 | `FaintEventPostProcessor` |
| cooldown 적용 | 같은 사람/카메라에서 반복 알림 방지 | `FaintEventPostProcessor` |
| Hard Negative 추가 수집 | 모델이 헷갈린 Normal을 다시 학습시켜 FP 감소 | `samples/evaluation/hard_negative`, 확인 필요: dataset 병합 스크립트 |
| Faint 보강 데이터 추가 | 미탐되는 Faint 유형을 학습에 추가해 FN 감소 | 확인 필요 |
| 실제 RTSP 환경 테스트 | 녹화/데모 영상과 실제 camera 환경 차이 확인 | `scripts/run_rtsp_inference.py`, `scripts/serve_ai_overlay.py` |
| fallback 비율 감소 | full-frame crop 학습을 줄이고 사람 중심 입력을 늘림 | `preprocess_summary_*.json`, `detector-mode yolo` |
| 오탐/미탐 로그 기반 재학습 루프 | FP/FN 샘플을 다시 학습에 넣어 성능 개선 | `--evaluation-log`, `scripts/evaluate_prediction_logs.py` |
| 이벤트 발행 전 후처리 개선 | 관제 피로도 감소, 중요 이벤트 유지 | threshold + consecutive + cooldown + keypoint 품질 조건 |

## 8. 멘토님께 설명하기 쉬운 요약

이 프로젝트의 AI는 CCTV 또는 RTSP 영상을 frame 단위로 읽고, YOLO26n-pose로 사람과 관절을 찾은 뒤, 같은 사람의 frame들을 sequence로 묶어 LSTM이 Normal/Faint를 판단하는 구조다. 한 frame만 보고 판단하지 않고 여러 frame의 흐름을 보기 때문에 실신처럼 자세가 변하는 상황을 더 안정적으로 판단하려는 목적이다.

운영 단계에서는 LSTM이 Faint 확률을 출력하고, 이 값이 threshold보다 높으면 Faint 후보가 된다. 하지만 바로 MQTT 이벤트를 보내지는 않고, 같은 camera와 track_id에서 Faint가 여러 번 연속으로 나와야 이벤트를 발행한다. 그리고 한 번 이벤트를 낸 뒤에는 cooldown을 적용해 같은 사람에 대한 중복 알림을 줄인다.

성능 개선 방향은 모델이 헷갈린 데이터를 다시 학습에 넣는 방식이다. Normal인데 Faint로 잘못 잡힌 데이터는 Hard Negative로 모으고, 실제 Faint인데 놓친 데이터는 Faint 보강 데이터로 모아 재학습하면 오탐과 미탐을 줄일 수 있다. 현재 repo에는 평가 로그와 threshold sweep 코드가 있으며, error-augmented dataset을 실제로 만든 raw CSV/결과 파일은 현재 로컬에서 확인되지 않아 GPU PC 결과 확인이 필요하다.

## 9. GPU PC 실행 명령어 예시

### 9.1 crop 기반 `train_lstm.py` 전처리 점검

```bash
cd ~/yolo_training/strange_ai_lstm

python -m ai.action.train_lstm \
  --dataset-csv "data/retrain_candidates/yolo26n_lstm_error_based/merged/dataset_error_augmented_clips.csv" \
  --detector-mode yolo \
  --yolo-model yolo26n-pose.pt \
  --device cuda:0 \
  --dry-run-preprocess
```

### 9.2 crop 기반 `train_lstm.py` 학습

```bash
cd ~/yolo_training/strange_ai_lstm

python -m ai.action.train_lstm \
  --dataset-csv "data/retrain_candidates/yolo26n_lstm_error_based/merged/dataset_error_augmented_clips.csv" \
  --detector-mode yolo \
  --yolo-model yolo26n-pose.pt \
  --sequence-length 16 \
  --sequence-stride 8 \
  --feature-size 32 \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.0 \
  --batch-size 32 \
  --epochs 20 \
  --lr 0.001 \
  --device cuda:0 \
  --output-dir runs/action_lstm
```

### 9.3 YOLO26n-pose keypoint 기반 최종 wrapper

```bash
cd ~/yolo_training/strange_ai_lstm

bash scripts/run_yolo26n_final_lstm.sh dry-run
bash scripts/run_yolo26n_final_lstm.sh train
bash scripts/run_yolo26n_final_lstm.sh audit
```

### 9.4 RTSP inference smoke test

```bash
cd ~/yolo_training/strange_ai_lstm

python scripts/run_rtsp_inference.py \
  --rtsp-url "rtsp://localhost:8554/cam1" \
  --camera-id cam_01 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --device 0 \
  --tracking-mode supervision \
  --action-model benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt \
  --action-device 0 \
  --action-threshold 0.3 \
  --min-consecutive-faint 3 \
  --camera-cooldown-seconds 10 \
  --classifier-input keypoints \
  --publisher console \
  --dry-run \
  --debug-every-n 1 \
  --max-frames 120 \
  --output runs/verification/cam_01_summary.json
```

### 9.5 prediction log 평가

```bash
cd ~/yolo_training/strange_ai_lstm

python scripts/evaluate_prediction_logs.py \
  --sample-root samples/evaluation \
  --output runs/evaluation/prediction_metrics.json \
  --sweep-csv runs/evaluation/threshold_sweep.csv \
  --faint-confidence-thresholds 0.3,0.4,0.5 \
  --consecutive-faint-counts 1,2,3 \
  --event-cooldown-seconds 0,10,30 \
  --max-keypoint-missing-rates 0.3,0.5,1.0 \
  --min-avg-keypoint-confs 0.0,0.3,0.5
```

## 10. 발표용 10줄 요약

1. RTSP/영상 frame을 읽고 YOLO26n-pose로 사람 bbox와 관절 keypoint를 추출한다.
2. 같은 사람의 frame을 track_id 기준으로 sequence buffer에 쌓는다.
3. LSTM은 sequence를 보고 Normal/Faint를 분류한다.
4. Faint 확률이 threshold 이상이면 Faint 후보로 본다.
5. 단일 예측만으로 알림을 보내지 않고, 연속 Faint 횟수와 cooldown을 적용한다.
6. MQTT payload에는 event_type, camera_id, track_id, confidence, faint_prob, bbox가 포함된다.
7. 학습 전처리에서는 detector 실패 시 fallback full-frame crop 비율을 확인해야 한다.
8. FP는 Normal을 Faint로 잘못 울린 오탐, FN은 Faint를 놓친 미탐이다.
9. 모델이 헷갈린 데이터를 Hard Negative/Faint 보강 데이터로 다시 넣어 개선하는 방향이다.
10. 현재 로컬에는 일부 최종 result 파일이 없어 GPU PC에서 raw 결과 파일 확인이 필요하다.

## 11. 현재 코드에서 확인 필요/구현 없음

- `data/retrain_candidates/yolo26n_lstm_error_based/merged/dataset_error_augmented_clips.csv`는 현재 로컬 checkout에서 확인되지 않았다.
- `benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/.../best.pt`는 기본 경로로 설정되어 있지만 현재 로컬에는 없다.
- `history.json`, `summary.json`, `threshold_audit.csv`, `confusion_matrix.csv`, `eval_predictions.csv`, `preprocess_summary_train.json`, `preprocess_summary_val.json` 실제 결과 파일은 현재 로컬에 없다.
- FP/FN을 자동으로 dataset CSV에 병합하는 스크립트는 현재 root `scripts/`에서 확인하지 못했다.
- `ai/action/train_lstm.py`의 저장 class가 `["Normal", "Fall"]`인 점과 운영 threshold 로직이 `Faint`를 기대하는 점은 정합성 확인이 필요하다.
