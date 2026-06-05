#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
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
