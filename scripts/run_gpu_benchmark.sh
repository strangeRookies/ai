#!/usr/bin/env bash
# =============================================================================
# GPU PC 수치 비교 실행 스크립트
# run_gpu_benchmark.sh
#
# 목적:
#   keypoint51 vs keypoint_motion54 모델 간 수치 비교,
#   TensorRT vs PyTorch 성능 비교,
#   회귀 테스트 순차 실행
#
# 사용법 (GPU PC에서):
#   cd /home/welabs/yolo_training/strange_ai_lstm
#   source .venv/bin/activate
#   bash scripts/run_gpu_benchmark.sh
#
# 선택적 환경 변수 오버라이드:
#   VIDEO=/다른경로/영상.mp4 bash scripts/run_gpu_benchmark.sh
#   SKIP_REGRESSION=1 bash scripts/run_gpu_benchmark.sh
#   SKIP_COMPARE=1    bash scripts/run_gpu_benchmark.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# -----------------------------------------------------------------------------
# ★ 경로 고정 블록 (GPU PC 절대경로 — 여기만 수정하면 됨)
# -----------------------------------------------------------------------------
PT="${PT:-/home/welabs/yolo_training/strange_ai_lstm/yolo26n-pose.pt}"
ENGINE="${ENGINE:-/home/welabs/yolo_training/strange_ai_lstm/yolo26n-pose.engine}"

ACTION51="${ACTION51:-/home/welabs/yolo_training/strange_ai_lstm/runs/evaluation_feature_dim/feature51/retrained_best.pt}"
ACTION54="${ACTION54:-/home/welabs/yolo_training/strange_ai_lstm/runs/evaluation_feature_dim/feature54/retrained_best.pt}"

VIDEO="${VIDEO:-/home/welabs/yolo_training/strange_ai_lstm/video_pool/faint_01.mp4}"
TRACK_VIDEO="${TRACK_VIDEO:-/home/welabs/yolo_training/strange_ai_lstm/video_pool/cam2.mp4}"

META="${META:-/home/welabs/yolo_training/ai_fall_experiments/data/metadata/metadata.csv}"
EVENT_META="${EVENT_META:-}"
KEYPOINT_CACHE="${KEYPOINT_CACHE:-/home/welabs/yolo_training/strange_ai_lstm/benchmark/results/lstm_extractor_comparison_fast/yolo26n-pose}"
# -----------------------------------------------------------------------------

# 실행 옵션
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
MAX_FRAMES="${MAX_FRAMES:-300}"
DETECTOR_CONF="${DETECTOR_CONF:-0.10}"

SKIP_REGRESSION="${SKIP_REGRESSION:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
SKIP_TRACKING="${SKIP_TRACKING:-0}"

# 출력 디렉토리 (타임스탬프 포함)
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$ROOT/runs/gpu_benchmark_${TS}}"
mkdir -p "$OUT"

# -----------------------------------------------------------------------------
# 유틸 함수
# -----------------------------------------------------------------------------
log()  { echo "[bench] $*"; }
warn() { echo "[bench][WARN] $*" >&2; }
die()  { echo "[bench][ERROR] $*" >&2; exit 1; }

run_logged() {
  local logfile="$1"; shift
  "$@" 2>&1 | tee "$logfile"
  return "${PIPESTATUS[0]}"
}

# -----------------------------------------------------------------------------
# 0) 파일 존재 확인
# -----------------------------------------------------------------------------
log "============================================================"
log "파일 존재 확인"
log "============================================================"

MISSING=0
for FILE in \
  "$PT" \
  "$ENGINE" \
  "$ACTION51" \
  "$ACTION54" \
  "$VIDEO" \
  "$TRACK_VIDEO" \
  "$META"
do
  if [ -e "$FILE" ]; then
    echo "  [OK]      $FILE"
  else
    echo "  [MISSING] $FILE"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  warn "누락 파일 ${MISSING}개 — ENGINE 누락은 허용, 나머지는 확인 필요"
  for REQUIRED in "$PT" "$VIDEO" "$META" "$ACTION51" "$ACTION54"; do
    [ -e "$REQUIRED" ] || die "필수 파일 없음: $REQUIRED"
  done
fi

log "OUT=$OUT"
log "DEVICE=$DEVICE  IMGSZ=$IMGSZ  MAX_FRAMES=$MAX_FRAMES"
log "VIDEO  (고정) = $VIDEO"
log "TRACK_VIDEO (고정) = $TRACK_VIDEO"

# -----------------------------------------------------------------------------
# 1) 회귀 테스트
# -----------------------------------------------------------------------------
if [[ "$SKIP_REGRESSION" != "1" ]]; then
  log "============================================================"
  log "[1/4] 회귀 테스트"
  log "============================================================"

  run_logged "$OUT/00_regression_tests.log" \
    python -m unittest \
      tests.test_standing_faint_block \
      tests.test_motion54_relink_discontinuity \
      tests.test_tracking_ab_replay \
      -v

  log "회귀 테스트 완료 → $OUT/00_regression_tests.log"
else
  log "[1/4] 회귀 테스트 SKIP (SKIP_REGRESSION=1)"
fi

# -----------------------------------------------------------------------------
# 2) Preflight 확인 (PyTorch & TensorRT)
# -----------------------------------------------------------------------------
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  log "============================================================"
  log "[2/4] Preflight 확인"
  log "============================================================"

  run_logged "$OUT/01_preflight_pytorch.log" \
    python -u scripts/run_rtsp_inference.py \
      --preflight-only \
      --detector-mode real \
      --yolo-model "$PT" \
      --device "$DEVICE" \
      --imgsz "$IMGSZ" \
      --detector-conf "$DETECTOR_CONF" \
      --action-model "$ACTION54" \
      --output "$OUT/01_preflight_pytorch.json"
  log "PyTorch preflight OK → $OUT/01_preflight_pytorch.json"

  if [[ -f "$ENGINE" ]]; then
    run_logged "$OUT/01_preflight_tensorrt.log" \
      python -u scripts/run_rtsp_inference.py \
        --preflight-only \
        --detector-mode real \
        --yolo-model "$ENGINE" \
        --device "$DEVICE" \
        --imgsz "$IMGSZ" \
        --detector-conf "$DETECTOR_CONF" \
        --action-model "$ACTION54" \
        --output "$OUT/01_preflight_tensorrt.json"
    log "TensorRT preflight OK → $OUT/01_preflight_tensorrt.json"
  else
    warn "ENGINE 파일 없음, TensorRT preflight 건너뜀: $ENGINE"
  fi
else
  log "[2/4] Preflight SKIP (SKIP_PREFLIGHT=1)"
fi

# -----------------------------------------------------------------------------
# 3) TensorRT vs PyTorch 수치 비교  (★ VIDEO 고정)
# -----------------------------------------------------------------------------
if [[ "$SKIP_COMPARE" != "1" ]]; then
  log "============================================================"
  log "[3/4] TensorRT vs PyTorch 수치 비교 (VIDEO 고정)"
  log "============================================================"

  if [[ -f "$ENGINE" ]]; then
    run_logged "$OUT/02_compare_tensorrt.log" \
      python scripts/compare_tensorrt_candidate.py \
        --model "$PT" \
        --engine "$ENGINE" \
        --video "$VIDEO" \
        --max-frames "$MAX_FRAMES" \
        --imgsz "$IMGSZ" \
        --device "$DEVICE" \
        --output-dir "$OUT/compare_tensorrt"
    log "TensorRT 비교 완료 → $OUT/compare_tensorrt/"
  else
    warn "ENGINE 없음, TensorRT 비교 건너뜀"
  fi

  # keypoint51 vs keypoint_motion54 LSTM 비교 (cache 모드)
  log "keypoint51 vs keypoint_motion54 LSTM 비교..."
  run_logged "$OUT/03_compare_action_models.log" \
    python benchmark/compare_lstm_extractors.py \
      --metadata-csv "$META" \
      --detector-mode cache \
      --keypoint-cache-dir "$KEYPOINT_CACHE" \
      --models "YOLO26n-pose:$PT" \
      --device "$DEVICE" \
      --imgsz "$IMGSZ" \
      --output-dir "$OUT/compare_action_models" \
      --train-split train \
      --eval-split val \
      --dry-run \
      --no-cpu-fallback || warn "LSTM 비교 실패 (비치명적)"
  log "LSTM 비교 완료 → $OUT/compare_action_models/"
else
  log "[3/4] 수치 비교 SKIP (SKIP_COMPARE=1)"
fi

# -----------------------------------------------------------------------------
# 4) 트래킹 A/B 리플레이  (★ TRACK_VIDEO 고정)
# -----------------------------------------------------------------------------
if [[ "$SKIP_TRACKING" != "1" ]]; then
  log "============================================================"
  log "[4/4] 트래킹 A/B 리플레이 (TRACK_VIDEO 고정)"
  log "============================================================"

  if [[ -f "$TRACK_VIDEO" ]]; then
    run_logged "$OUT/04_tracking_replay.log" \
      python scripts/replay_tracking_from_cache.py \
        --video "$TRACK_VIDEO" \
        --output-dir "$OUT/tracking_replay" \
        --device "$DEVICE" || warn "트래킹 리플레이 실패 (비치명적)"
    log "트래킹 리플레이 완료 → $OUT/tracking_replay/"
  else
    warn "TRACK_VIDEO 파일 없음, 트래킹 리플레이 건너뜀: $TRACK_VIDEO"
  fi
else
  log "[4/4] 트래킹 SKIP (SKIP_TRACKING=1)"
fi

# -----------------------------------------------------------------------------
# 최종 요약
# -----------------------------------------------------------------------------
log "============================================================"
log "모든 단계 완료"
log "결과 디렉토리: $OUT"
log "============================================================"

for f in "$OUT"/*.json; do
  [[ -e "$f" ]] || continue
  python - "$f" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
    keys = ["runtime", "model_path", "frames_processed",
            "avg_yolo_inference_ms", "effective_fps", "events_generated"]
    brief = {k: d.get(k) for k in keys if k in d}
    print(f"  [{p.name}] {brief}")
except Exception as e:
    print(f"  [{p.name}] 파싱 오류: {e}")
PY
done
