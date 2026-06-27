#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-/dev/video0}"
CAMERA_PATH="${2:-cam1}"
RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://localhost:8554}"
VIDEO_SIZE="${VIDEO_SIZE:-1280x720}"
FRAMERATE="${FRAMERATE:-30}"

exec ffmpeg \
  -f v4l2 \
  -framerate "$FRAMERATE" \
  -video_size "$VIDEO_SIZE" \
  -i "$DEVICE" \
  -an \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -f rtsp \
  "$RTSP_BASE_URL/$CAMERA_PATH"
