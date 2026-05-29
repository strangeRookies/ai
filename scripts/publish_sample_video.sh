#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <video-path> [cam1|cam2|cam3|cam4]" >&2
  exit 1
fi

VIDEO_PATH="$1"
CAMERA_PATH="${2:-cam1}"
RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://localhost:8554}"

exec ffmpeg -re -stream_loop -1 \
  -i "$VIDEO_PATH" \
  -an \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -f rtsp \
  "$RTSP_BASE_URL/$CAMERA_PATH"
