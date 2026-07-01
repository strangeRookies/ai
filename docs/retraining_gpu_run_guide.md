# GPU PC LSTM 재학습 및 평가 실행 가이드

이 가이드는 GPU PC 또는 원격 GPU 서버에서 LSTM 쓰러짐 분류 모델의 재학습 및 평가 파이프라인을 실행하는 방법을 설명합니다.

## 1. 로컬 환경 vs GPU PC 환경 비교

| 구분 / 단계 | 로컬 개발 환경 (Local Environment) | GPU PC / 원격 GPU 서버 (GPU PC) |
| :--- | :--- | :--- |
| **주요 목적** | 코드 수정, 파이프라인 작동 확인, Dry-run 검증 | 실제 모델 학습, 최종 평가, 성능 비교 벤치마크 |
| **데이터 소스** | Mock keypoint (누락 시 자동 생성), 샘플 메타데이터 | 실제 추출된 YOLO Pose 키포인트 캐시(NPZ) 및 비디오 클립 |
| **실행 모드** | `--dry-run` 또는 `--sample 100` | 전체 데이터셋 기반의 Real-mode 학습 및 검증 |
| **하드웨어** | CPU / Mock 실행 | NVIDIA GPU (CUDA 활성화 필요) |

> [!WARNING]
> **Mock 모드는 실제 모델 성능 지표가 아닙니다.**
> 로컬 환경에서 `--dry-run`이나 mock fallback 모드로 실행하는 평가는 파이프라인의 정상 동작 여부를 확인하기 위해 무작위 또는 모사된 수치를 생성합니다. **성능 개선 여부를 검증하기 위한 최종 비교 자료로는 반드시 GPU PC에서의 real mode 평가 결과만 사용해야 합니다.**

---

## 2. GPU PC 환경 확인

원격 GPU PC에 SSH로 로그인한 뒤, 작업을 시작하기 전에 PyTorch 및 CUDA 상태를 확인하십시오.

```bash
# 1. 프로젝트 루트 디렉토리로 이동
cd ~/yolo_training/strange_ai_lstm/strange_ai

# 2. 가상환경 접속 (활성화)
source .venv/bin/activate

# 3. Python 버전 확인
python --version

# 4. PyTorch 설치 여부 및 CUDA 사용 가능 상태 확인
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("[ERROR] CUDA is not available on this GPU PC! Check nvidia drivers.")
PY
```

---

## 3. 실행 전 필수 경로 확인

GPU PC에서 실행하기 전에 아래의 디렉토리와 파일들이 정상적으로 존재하는지 확인하십시오:

* **Baseline 메타데이터 CSV:** `../ai_fall_experiments/data/metadata/metadata.csv`
* **V2 재학습용 Manifest:** `data/manifests/training_manifest_v2.csv`
* **Keypoint 캐시 디렉토리:** `../ai_fall_experiments/data/keypoints/yolo26n-pose`
* **비디오 클립 루트 디렉토리:** `/home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos`
* **LSTM 체크포인트 출력 디렉토리:** `runs/action_lstm/`
* **평가 보고서 출력 경로:** `reports/retraining_manifest_v2_eval.md`

---

## 4. 실행 단계 (실제 명령어)

### 단계 4.1. Manifest Leakage 및 승인 여부 검증
생성된 `training_manifest_v2.csv`에 데이터 스플릿 누수(split leakage)나 승인 상태(`review_status=approved`) 위반 항목이 없는지 검사합니다:
```bash
python scripts/check_manifest_leakage.py \
  --manifest data/manifests/training_manifest_v2.csv
```

### 단계 4.2. Sample 기반 빠른 평가 (경로 및 작동 검증)
전체 데이터셋 학습 전, 소량의 데이터(100개 샘플)만 사용하여 파이프라인 경로 및 동작이 정상적인지 우선 검증합니다:
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --sample 100 \
  --output-dir runs/evaluation_sample \
  --report-path reports/retraining_manifest_v2_eval_sample.md
```

### 단계 4.3. 전체 Real-Mode 재학습 및 성능 평가 실행
샘플 평가가 정상적으로 완료되면, 전체 실제 데이터셋에 대해 baseline과 retrained 모델을 동시에 학습하여 최종 성능 비교를 수행합니다.

이 명령어는 다음을 자동으로 수행합니다:
1. 전체 학습 데이터셋에 대한 YOLO Pose 키포인트 캐시 시퀀스를 로드합니다.
2. Baseline 학습 세트와 Retrained 학습 세트 각각에서 두 개의 LSTM 모델을 개별 학습합니다.
3. 학습된 두 모델을 *동일한* 공통 테스트 스플릿(Test split)에 대해 평가합니다.
4. Precision, Recall, F1-score 개선 수치 및 시나리오 태그별 통계 보고서를 생성합니다.

```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --output-dir runs/evaluation_real \
  --report-path reports/retraining_manifest_v2_eval_real.md \
  --device cuda \
  --epochs 20 \
  --batch-size 32 \
  --seed 42
```

---

## 5. 대안: 단계별 개별 학습 및 사후 평가

만약 표준 학습 스크립트로 LSTM 모델을 먼저 학습하여 파일로 저장해 둔 뒤, 저장된 체크포인트 파일만 개별적으로 평가하고 싶다면 아래 단계를 따릅니다.

### 단계 5.1. LSTM 액션 분류 모델 학습 실행
메인 재학습용 스크립트를 사용하여 V2 manifest 데이터셋을 기반으로 LSTM 모델을 개별 학습시킵니다:
```bash
python -m ai.action.train_lstm \
  --dataset-csv data/manifests/training_manifest_v2.csv \
  --detector-mode yolo \
  --yolo-model yolo26n-pose.pt \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --output-dir runs/retraining_manifest_v2
```
*완료되면 `runs/retraining_manifest_v2/best.pt` 가 생성됩니다.*

### 단계 5.2. 저장된 체크포인트(best.pt) 사후 평가
저장된 가중치 체크포인트를 평가 스크립트에 바로 주입하여 baseline 모델과 비교 분석 보고서를 생성합니다:
```bash
python scripts/evaluate_retraining_manifest_v2.py \
  --baseline ../ai_fall_experiments/data/metadata/metadata.csv \
  --retrained data/manifests/training_manifest_v2.csv \
  --checkpoint runs/retraining_manifest_v2/best.pt \
  --output-dir runs/evaluation_retrained_checkpoint \
  --report-path reports/retraining_manifest_v2_checkpoint_eval.md \
  --device cuda
```

---

## 6. 생성되는 결과 파일

* **마크다운 결과 보고서:** `reports/retraining_manifest_v2_eval_real.md` (F1-score, accuracy, 시나리오별 FP/FN 감소 분석 요약)
* **상세 메트릭 JSON:** `runs/evaluation_real/evaluation_summary.json` (baseline 및 retrained 실행에 대한 로우 데이터 로그)

---

## 7. 실행 실패 시 체크리스트

| 발생 에러 / 현상 | 예상 원인 | 해결책 |
| :--- | :--- | :--- |
| `FileNotFoundError` 발생 | 파일 경로 지정 오류 | `--baseline` 및 `--retrained` 인자로 넘긴 파일이 실제 GPU PC 경로에 존재하는지 확인합니다. |
| Keypoint 관련 `RuntimeError` | keypoint cache (`.npz`) 누락 | `python scripts/extract_keypoint_cache.py --metadata-csv <csv> --cache-dir <dir>` 명령을 먼저 실행하여 캐시 데이터를 생성하십시오. |
| `RuntimeError: split_group_id leakage` | 데이터 스플릿 누수 | 동일한 parent_clip_id 또는 split_group_id를 가진 데이터가 train, val, test에 섞여 있습니다. manifest 생성 스크립트를 재조정해야 합니다. |
| `RuntimeError: pending/rejected rows` | 검수되지 않은 데이터 포함 | `review_status`가 `approved`가 아닌 pending, rejected, needs_review 상태의 임시 행이 포함되어 있습니다. `check_manifest_leakage.py`를 실행해 위반 행을 색출하십시오. |
| CUDA out of memory / unavailable | PyTorch의 GPU 메모리 고갈 | `nvidia-smi`로 잔여 메모리를 확인하고, `--batch-size`를 16이나 32로 낮춰 실행합니다. |

---

## 8. 실행 결과 수집 (로컬 PC로 가져오기)

GPU PC에서 모든 평가 리포트 생성이 완료되면, 로컬 개발 환경으로 마크다운 리포트를 가져와 결과를 검토 및 공유할 수 있습니다:

```bash
# 로컬 개발 환경 터미널에서 실행
scp GPU_USER@GPU_HOST:~/yolo_training/strange_ai_lstm/strange_ai/reports/retraining_manifest_v2_eval_real.md ./docs/retraining_gpu_eval_report.md
```

