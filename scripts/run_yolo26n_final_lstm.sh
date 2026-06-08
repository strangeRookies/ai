#!/usr/bin/env bash
# =========================================================================================
# YOLO26n-pose LSTM Final Run Wrapper
# 
# [최종 서비스 검증/시연용 실행 예시]
# 최종 서비스 검증 및 시연 시에는 크로마키(그린스크린) 영상이 제외된 CSV를 사용해야 합니다.
# 
#   METADATA_CSV="data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv" \
#   bash scripts/run_yolo26n_final_lstm.sh audit
# 
# [기존 ffmpeg 송출 프로세스 종료 후 재실행 절차]
# 만약 백그라운드에 RTSP 송출(ffmpeg)이나 Overlay 서버가 돌고 있다면 충돌(포트/메모리)을 방지하기 위해 
# 프로세스 정리 후 재실행하는 것을 권장합니다.
#   1) fuser -k 8010/tcp 2>/dev/null || true
#      pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
#   2) pkill -9 ffmpeg 2>/dev/null || true
#   3) 본 스크립트 등 목적에 맞는 스크립트 재실행
# =========================================================================================
set -euo pipefail

MODE="${1:-dry-run}"
SERVICE_VALIDATION_CSV="data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
METADATA_CSV="${METADATA_CSV:-data/splits/final_source_video_split/all.csv}"
POSE_MODEL="${POSE_MODEL:-yolo26n-pose.pt}"
MODEL_SPEC="${MODEL_SPEC:-YOLO26n-pose:${POSE_MODEL}}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-10}"
REPEAT_SEEDS="${REPEAT_SEEDS:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DRYRUN_ROWS_PER_CLASS="${DRYRUN_ROWS_PER_CLASS:-2}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-benchmark/results/lstm_yolo26n_final_split}"
DRYRUN_OUTPUT_DIR="${DRYRUN_OUTPUT_DIR:-benchmark/results/lstm_yolo26n_final_split_dryrun}"

echo "============================================================"
echo "[INFO] Starting YOLO26n-pose LSTM Run Wrapper"
echo "[INFO] Current METADATA_CSV: ${METADATA_CSV}"

if [[ "${METADATA_CSV}" == *"all.csv"* ]]; then
  echo "[WARNING] You are using the default all.csv metadata."
  echo "[WARNING] Chromakey (green screen) videos may be included in this dataset!"
  echo "[INFO] For final service validation/demo, please use:"
  echo "[INFO] METADATA_CSV=${SERVICE_VALIDATION_CSV}"
fi
echo "============================================================"

case "${MODE}" in
  dry-run)
    python benchmark/compare_lstm_extractors.py \
      --metadata-csv "${METADATA_CSV}" \
      --detector-mode real \
      --models "${MODEL_SPEC}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --output-dir "${DRYRUN_OUTPUT_DIR}" \
      --train-split train \
      --eval-split val \
      --max-rows-per-split "${DRYRUN_ROWS_PER_CLASS}" \
      --max-frames "${MAX_FRAMES}" \
      --epochs 1 \
      --dry-run \
      --no-cpu-fallback
    ;;
  sequences)
    python benchmark/compare_lstm_extractors.py \
      --metadata-csv "${METADATA_CSV}" \
      --detector-mode real \
      --models "${MODEL_SPEC}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --output-dir "${FINAL_OUTPUT_DIR}" \
      --train-split train \
      --eval-split val \
      --max-frames "${MAX_FRAMES}" \
      --epochs 1 \
      --dry-run \
      --no-cpu-fallback
    ;;
  train)
    python benchmark/compare_lstm_extractors.py \
      --metadata-csv "${METADATA_CSV}" \
      --detector-mode real \
      --models "${MODEL_SPEC}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --output-dir "${FINAL_OUTPUT_DIR}" \
      --train-split train \
      --eval-split val \
      --max-frames "${MAX_FRAMES}" \
      --epochs "${EPOCHS}" \
      --repeat-seeds "${REPEAT_SEEDS}" \
      --no-cpu-fallback
    ;;
  test-audit|audit)
    TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${FINAL_OUTPUT_DIR}_test_audit}"
    python benchmark/compare_lstm_extractors.py \
      --metadata-csv "${METADATA_CSV}" \
      --detector-mode real \
      --models "${MODEL_SPEC}" \
      --device "${DEVICE}" \
      --imgsz "${IMGSZ}" \
      --output-dir "${TEST_OUTPUT_DIR}" \
      --train-split train \
      --eval-split test \
      --max-frames "${MAX_FRAMES}" \
      --epochs "${EPOCHS}" \
      --repeat-seeds "${REPEAT_SEEDS}" \
      --audit-thresholds 0.3,0.4,0.5,0.6,0.7 \
      --no-cpu-fallback
    python scripts/audit_lstm_thresholds.py \
      --predictions "${TEST_OUTPUT_DIR}/YOLO26n-pose/eval_predictions.csv" \
      --output-dir "${TEST_OUTPUT_DIR}/threshold_audit"
    ;;
  *)
    echo "Usage: bash scripts/run_yolo26n_final_lstm.sh [dry-run|sequences|train|audit]" >&2
    exit 2
    ;;
esac
