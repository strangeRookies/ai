#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV="${DATASET_CSV:-../ai_fall_experiments/data/metadata/metadata.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
DETECTOR_MODE="${DETECTOR_MODE:-yolo}"
DEVICE="${DEVICE:-auto}"
YOLO_CONF="${YOLO_CONF:-0.15}"
YOLO_RETRY_CONF="${YOLO_RETRY_CONF:-0.10}"
YOLO_IOU="${YOLO_IOU:-0.5}"
IMGSZ="${IMGSZ:-640}"
FALLBACK_FULL_FRAME="${FALLBACK_FULL_FRAME:-true}"
EPOCHS="${EPOCHS:-20}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-30}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-15}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
FEATURE_SIZE="${FEATURE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MAX_ROWS_PER_SPLIT="${MAX_ROWS_PER_SPLIT:-0}"

MODELS=(
  yolov8n.pt
  yolo11n.pt
)

mkdir -p runs/action_lstm

FALLBACK_FLAG="--fallback-full-frame"
if [ "${FALLBACK_FULL_FRAME}" = "false" ] || [ "${FALLBACK_FULL_FRAME}" = "0" ]; then
  FALLBACK_FLAG="--no-fallback-full-frame"
fi

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME="${MODEL%.pt}"
  OUT_DIR="runs/action_lstm/${MODEL_NAME}"
  echo "[lstm-compare] training model=${MODEL} output=${OUT_DIR}"
  python -m ai.action.train_lstm \
    --dataset-csv "${DATASET_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --detector-mode "${DETECTOR_MODE}" \
    --yolo-model "${MODEL}" \
    --yolo-conf "${YOLO_CONF}" \
    --yolo-retry-conf "${YOLO_RETRY_CONF}" \
    --yolo-iou "${YOLO_IOU}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --resize-size "${RESIZE_SIZE}" \
    --feature-size "${FEATURE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-frames "${MAX_FRAMES}" \
    --max-rows-per-split "${MAX_ROWS_PER_SPLIT}" \
    "${FALLBACK_FLAG}" \
    --output-dir "${OUT_DIR}"
done

python -m ai.action.summarize_lstm_runs --runs-dir runs/action_lstm
echo "[lstm-compare] done"
