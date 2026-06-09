#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/fall_lstm.yaml}"
EXPERIMENTS="${EXPERIMENTS:-indoor_outdoor all_domains partial_chromakey}"

for MODEL in yolov8n-pose.pt yolo11n-pose.pt; do
  EXPERIMENTS="${EXPERIMENTS}" CONFIG="${CONFIG}" bash scripts/run_selected_pose_and_train.sh "${MODEL}"
done

python scripts/summarize_results.py --runs runs/lstm
