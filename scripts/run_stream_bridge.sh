#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export CAMERA_1_RTSP_URL="${CAMERA_1_RTSP_URL:-rtsp://localhost:8554/cam1}"
export CAMERA_2_RTSP_URL="${CAMERA_2_RTSP_URL:-rtsp://localhost:8554/cam2}"
export CAMERA_3_RTSP_URL="${CAMERA_3_RTSP_URL:-rtsp://localhost:8554/cam3}"
export CAMERA_4_RTSP_URL="${CAMERA_4_RTSP_URL:-rtsp://localhost:8554/cam4}"
export MJPEG_HOST="${MJPEG_HOST:-0.0.0.0}"
export MJPEG_PORT="${MJPEG_PORT:-8000}"
export MJPEG_FPS="${MJPEG_FPS:-8}"

exec python serve_mjpeg.py
