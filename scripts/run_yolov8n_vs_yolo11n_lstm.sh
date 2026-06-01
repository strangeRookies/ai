#!/usr/bin/env bash
set -euo pipefail

DATASET_CSV="${DATASET_CSV:-../ai_fall_experiments/data/metadata/metadata.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
DETECTOR_MODE="${DETECTOR_MODE:-yolo}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-20}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-8}"
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
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --resize-size "${RESIZE_SIZE}" \
    --feature-size "${FEATURE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-frames "${MAX_FRAMES}" \
    --max-rows-per-split "${MAX_ROWS_PER_SPLIT}" \
    --output-dir "${OUT_DIR}"
done

python -m ai.action.summarize_lstm_runs --runs-dir runs/action_lstm
echo "[lstm-compare] done"
