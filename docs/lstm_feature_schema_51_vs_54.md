# LSTM Feature Schema: 51-Dimensional vs 54-Dimensional

이 문서는 본 프로젝트의 LSTM 행동 분류기에서 지원하는 51차원 keypoint feature와 54차원 feature의 구조, 정규화 방식, 호환 정책 및 GPU PC 실행 가이드라인을 제공합니다.

---

## 1. Feature Schema 구조 및 차이

| Feature Schema | Input Size | 차원 구성 (Feature Elements) | 마지막 3차원 정의 (Additional Dims) |
| :--- | :---: | :--- | :--- |
| **`keypoint51`** (기본값) | 51 | 17 keypoints × 3 values (`normalized_x`, `normalized_y`, `confidence`) | 없음 (Base keypoints only) |
| **`keypoint_motion54`** | 54 | 17 keypoints × 3 values + 3 motion values | `center_drop`, `velocity`, `torso_angle_norm` |
| **`keypoint_bbox54`** | 54 | 17 keypoints × 3 values + 3 bbox values | `bbox_width_norm`, `bbox_height_norm`, `bbox_area_norm` |

### 1.1. 51차원 기본 구조 (`keypoint51`)
- COCO 포맷의 17개 관절(keypoints)에 대해 각 관절당 `x`, `y`, `confidence` 총 3개 값을 수집하여 51차원을 형성합니다.
- `x`, `y` 좌표는 프레임의 해상도(width, height)를 기준으로 나누어 `[0, 1]` 범위로 정규화됩니다.

### 1.2. 54차원 모션 구조 (`keypoint_motion54` - 기존 코드 발견)
- 기존 `ai/action/motion_features.py`에 구현되어 있던 54차원 구조입니다.
- **`center_drop`**: 프레임 $t$와 $t-1$ 사이의 골반 중심(hip midpoint) y좌표 변화량 ($y_t - y_{t-1}$).
- **`velocity`**: 골반 중심의 프레임 간 이동 속도(Euclidean distance).
- **`torso_angle_norm`**: 어깨 중심과 골반 중심 사이의 몸통 각도를 `[0, 1]` 범위로 정규화한 값.
- **Fallback**: 첫 번째 프레임($t=0$)이거나 keypoint가 소실된 경우 추가 3차원은 `0.0`으로 대체됩니다.

### 1.3. 54차원 바운딩 박스 구조 (`keypoint_bbox54` - 신규 제안 계약)
- 사람 바운딩 박스 크기 및 비율 변화를 반영하기 위해 추가된 54차원 구조입니다.
- **`bbox_width_norm`** = $bbox\_width / frame\_width$
- **`bbox_height_norm`** = $bbox\_height / frame\_height$
- **`bbox_area_norm`** = $bbox\_width\_norm \times bbox\_height\_norm$
- **Fallback (bbox missing)**: 바운딩 박스가 탐지되지 않았거나 missing인 경우 마지막 3개 값은 **`0.0`**으로 자동 패딩 처리됩니다.

---

## 2. 학습 및 추론 feature 생성 경로

학습과 실시간 추론 시의 feature 생성 순서 및 값 계산 방식의 일관성은 모델 성능에 치명적입니다. 본 프로젝트는 아래의 공통 feature builder 함수들을 사용하여 동일한 feature 순서를 엄격히 강제합니다.

1. **학습 전처리 / 평가 시**:
   - `scripts/evaluate_retraining_manifest_v2.py`가 NPZ 캐시 로드 후 `load_npz_sequences`를 호출하여 `--feature-schema`에 해당하는 차원으로 슬라이싱 또는 모션/패딩 3차원 확장을 진행합니다.
2. **실시간 추론 시**:
   - `serve_ai_overlay.py` 및 RTSP runtime 루프가 `LSTMActionClassifier.predict()`를 호출합니다.
   - 내부적으로 `ai/action/classifier.py`의 `sequence_to_lstm_features`를 실행하여 YOLO pose 탐지 결과로부터 동일한 방식으로 51차원 또는 54차원을 실시간으로 생성합니다.

---

## 3. Checkpoint 호환 및 검증 정책

51차원 모델에 54차원 feature가 주입되거나, 54차원 모델에 51차원 feature가 잘못 연동되어 오작동하는 현상을 방지하기 위해 **엄격한 차원 세이프가드(Safeguard)**가 적용되어 있습니다.

- **Checkpoints 저장 메타데이터**:
  학습 완료 시 모델 `.pt` 체크포인트에 `input_size`, `feature_schema_version`, `feature_names` 정보가 포함되어 저장됩니다.
- **Constructor Safeguard**:
  `LSTMActionClassifier` 로드 시 체크포인트 내부의 `input_size`와 `feature_schema_version` 정보가 다를 경우 `ValueError`를 발생시키고 즉시 기동을 중단합니다.
- **Runtime Safeguard**:
  실시간 추론 `predict()` 단계에서 주입된 feature의 최종 차원이 모델의 `input_size`와 일치하지 않으면 조용히 넘어가거나 패딩하지 않고, `ValueError` 예외를 던지며 프로세스를 안전하게 차단합니다.

---

## 4. GPU PC 실행 가이드라인 (명령어)

로컬 개발 환경에서는 dry-run 또는 mock 모드로 shape와 리포트 형식만 빠르게 검증하고, 대량 데이터(21만 개)에 대한 실제 학습 및 평가는 **GPU 서버**에서 아래 명령어를 복사하여 실행합니다.

### 4.1. 가상환경 활성화 (welabs 계정 예시)
```bash
# 가상환경 진입
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate
```

### 4.2. 51차원 Baseline 모델 학습 및 평가
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline data/metadata/metadata.csv \
  --manifest data/metadata/metadata.csv \
  --feature-schema keypoint51 \
  --input-size 51 \
  --balance-labels \
  --per-class \
  --train-limit 7000 \
  --val-limit 1500 \
  --test-limit 1400 \
  --fixed-test-from-baseline \
  --device cuda \
  --epochs 20 \
  --seed 42 \
  --output-dir runs/evaluation_feature_dim/feature51 \
  --report-path reports/lstm_feature_51_eval.md
```

### 4.3. 54차원 바운딩 박스 모델 학습 및 평가
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline data/metadata/metadata.csv \
  --manifest data/metadata/metadata.csv \
  --feature-schema keypoint_bbox54 \
  --input-size 54 \
  --balance-labels \
  --per-class \
  --train-limit 7000 \
  --val-limit 1500 \
  --test-limit 1400 \
  --fixed-test-from-baseline \
  --device cuda \
  --epochs 20 \
  --seed 42 \
  --output-dir runs/evaluation_feature_dim/feature54 \
  --report-path reports/lstm_feature_54_eval.md
```

### 4.4. 51차원 vs 54차원 성능 수치 비교 리포트 생성
```bash
python scripts/compare_lstm_feature_dims.py \
  --metrics-a runs/evaluation_feature_dim/feature51/metrics.json \
  --metrics-b runs/evaluation_feature_dim/feature54/metrics.json \
  --name-a keypoint51 \
  --name-b keypoint_bbox54 \
  --output-csv runs/evaluation_feature_dim/comparison_51_vs_54.csv \
  --report-path reports/lstm_feature_51_vs_54_eval.md
```

---

## 5. 결과 해석 및 주의사항

> [!WARNING]
> **Mock vs Real 모드 차이**:
> 로컬 환경에서 `--dry-run`으로 생성된 리포트 및 성능 지표는 **검증용 임시 모의 데이터(Plausible Mock)**입니다. 실제 54차원 feature 도입에 따른 정확도 향상 유무는 GPU 서버에서 실 데이터(Cache/NPZ) 기반의 **Real Mode** 실행이 완료되어야 정확히 판단할 수 있습니다.
