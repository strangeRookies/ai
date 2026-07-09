#!/usr/bin/env bash
set -euo pipefail

MAX_FRAMES="${MAX_FRAMES:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/verification}"
YOLO_MODEL="${YOLO_MODEL:-yolo26n-pose.pt}"
ACTION_MODEL="${ACTION_MODEL:-benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt}"
DEVICE="${DEVICE:-0}"
ACTION_DEVICE="${ACTION_DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
DETECTOR_CONF="${DETECTOR_CONF:-0.10}"
ACTION_THRESHOLD="${ACTION_THRESHOLD:-0.3}"
MIN_CONSECUTIVE_FAINT="${MIN_CONSECUTIVE_FAINT:-3}"
CAMERA_COOLDOWN_SECONDS="${CAMERA_COOLDOWN_SECONDS:-10}"

mkdir -p "$OUTPUT_DIR"

model_label="$(basename "$YOLO_MODEL")"
model_label="${model_label%.*}"
model_label="${model_label//[^A-Za-z0-9_-]/_}"
suffix="_${model_label}_${MAX_FRAMES}"

for idx in 1 2 3 4; do
  camera_id="$(printf "cam_%02d" "$idx")"
  rtsp_url="rtsp://localhost:8554/${camera_id}"
  output="${OUTPUT_DIR}/${camera_id}_rtsp_metrics${suffix}.json"
  echo "[4cam-metrics] ${camera_id} ${rtsp_url} -> ${output}"
  python -u scripts/run_rtsp_inference.py \
    --rtsp-url "$rtsp_url" \
    --camera-id "$camera_id" \
    --camera-login-id "$camera_id" \
    --max-frames "$MAX_FRAMES" \
    --detector-mode real \
    --yolo-model "$YOLO_MODEL" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --detector-conf "$DETECTOR_CONF" \
    --action-model "$ACTION_MODEL" \
    --action-device "$ACTION_DEVICE" \
    --action-threshold "$ACTION_THRESHOLD" \
    --min-consecutive-faint "$MIN_CONSECUTIVE_FAINT" \
    --camera-cooldown-seconds "$CAMERA_COOLDOWN_SECONDS" \
    --classifier-input keypoints \
    --dry-run \
    --output "$output"
done
