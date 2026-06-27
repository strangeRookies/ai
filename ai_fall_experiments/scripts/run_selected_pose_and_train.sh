#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_selected_pose_and_train.sh <pose-model.pt>" >&2
  echo "Example: bash scripts/run_selected_pose_and_train.sh yolo11s-pose.pt" >&2
  exit 2
fi

POSE_MODEL="$1"
CONFIG="${CONFIG:-configs/fall_lstm.yaml}"
EXPERIMENTS="${EXPERIMENTS:-indoor_outdoor all_domains partial_chromakey}"
MODEL_NAME="$(basename "${POSE_MODEL}" .pt)"
KEYPOINT_DIR="${KEYPOINT_DIR:-data/keypoints/${MODEL_NAME}}"
RUN_ROOT="${RUN_ROOT:-runs/selected_pose/${MODEL_NAME}}"
EXTRACT_LOG="${RUN_ROOT}/extract_keypoints.log"
METRICS_JSON="${RUN_ROOT}/pose_metrics.json"
METRICS_CSV="${RUN_ROOT}/pose_metrics.csv"
TRAIN_LOG="${RUN_ROOT}/train_all.log"
METADATA_CSV="${METADATA_CSV:-data/metadata/metadata.csv}"
MODEL_METADATA_CSV="${RUN_ROOT}/metadata.csv"

mkdir -p "${RUN_ROOT}"

log() {
  echo "$(date -Is) $*" | tee -a "${TRAIN_LOG}"
}

log "[run] pose_model=${POSE_MODEL}"
log "[run] config=${CONFIG}"
log "[run] keypoint_dir=${KEYPOINT_DIR}"
log "[run] experiments=${EXPERIMENTS}"

if [ ! -f "${METADATA_CSV}" ]; then
  log "[run] missing metadata=${METADATA_CSV}. Finish prepare_clips.py first."
  exit 1
fi

log "[run] extracting keypoints and updating metadata"
python scripts/extract_keypoints_yolo_pose.py \
  --config "${CONFIG}" \
  --pose-model "${POSE_MODEL}" \
  --output-dir "${KEYPOINT_DIR}" \
  --metrics-output "${METRICS_JSON}" \
  --metrics-csv-output "${METRICS_CSV}" \
  --overwrite \
  > "${EXTRACT_LOG}" 2>&1
log "[run] extract_done log=${EXTRACT_LOG} metrics=${METRICS_JSON}"
cp "${METADATA_CSV}" "${MODEL_METADATA_CSV}"
log "[run] metadata_snapshot=${MODEL_METADATA_CSV}"

for EXPERIMENT in ${EXPERIMENTS}; do
  EXP_LOG="${RUN_ROOT}/train_${EXPERIMENT}.log"
  EXP_RUN_DIR="runs/lstm/${MODEL_NAME}/${EXPERIMENT}"
  log "[run] training experiment=${EXPERIMENT}"
  python scripts/train_lstm.py \
    --config "${CONFIG}" \
    --experiment "${EXPERIMENT}" \
    --metadata "${MODEL_METADATA_CSV}" \
    --run-dir "${EXP_RUN_DIR}" \
    > "${EXP_LOG}" 2>&1
  log "[run] train_done experiment=${EXPERIMENT} log=${EXP_LOG}"

  CHECKPOINT="${EXP_RUN_DIR}/best.pt"
  if [ -f "${CHECKPOINT}" ]; then
    EVAL_LOG="${RUN_ROOT}/eval_${EXPERIMENT}.log"
    log "[run] evaluating experiment=${EXPERIMENT}"
    python scripts/eval_lstm.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --metadata "${MODEL_METADATA_CSV}" \
      > "${EVAL_LOG}" 2>&1
    log "[run] eval_done experiment=${EXPERIMENT} log=${EVAL_LOG}"
  else
    log "[run] skip_eval experiment=${EXPERIMENT} missing_checkpoint=${CHECKPOINT}"
  fi
done

SUMMARY_LOG="${RUN_ROOT}/summary.log"
python scripts/summarize_results.py --runs runs/lstm > "${SUMMARY_LOG}" 2>&1
log "[run] summary_done log=${SUMMARY_LOG}"
log "[run] done"
