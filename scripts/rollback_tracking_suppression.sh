#!/bin/bash
# Rollback near-dup suppression to pre-adoption production values.
# Does NOT kill mediamtx / simulated RTSP publishers.
set -euo pipefail

REMOTE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REMOTE_ROOT"

export NEAR_DUP_SUPPRESS_MODE=none
export SIMPLE_TRACK_NEW_TRACK_THRESH=0.25

# Clear any per-camera canary overrides
if [ -f .venv/bin/python ]; then
  PYTHONPATH="$REMOTE_ROOT" .venv/bin/python - <<'PY'
from ai.tracking_canary import clear_canary_config
clear_canary_config()
print("[rollback] cleared canary_config.json")
PY
fi

echo "[rollback] NEAR_DUP_SUPPRESS_MODE=$NEAR_DUP_SUPPRESS_MODE"
echo "[rollback] SIMPLE_TRACK_NEW_TRACK_THRESH=$SIMPLE_TRACK_NEW_TRACK_THRESH"

# Restart only AI runner + overlay workers (keep publisher/ffmpeg/mediamtx)
pkill -f 'scripts/run_registered_cameras.py' || true
pkill -f 'scripts/serve_ai_overlay.py' || true
sleep 2

if [ -f scripts/start_ai_stable.sh ]; then
  echo "[rollback] Re-start via start_ai_stable is NOT automatic (would respawn publisher)."
  echo "[rollback] Restart runner only with rolled-back env:"
fi

echo "[rollback] Starting run_registered_cameras with rollback env..."
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

nohup env VIDEO_DOMAIN= \
  NEAR_DUP_SUPPRESS_MODE=none \
  SIMPLE_TRACK_NEW_TRACK_THRESH=0.25 \
  NEAR_DUP_SORT_BY_CONF=1 \
  python scripts/run_registered_cameras.py \
    --backend-base-url "${BACKEND_URL:-http://127.0.0.1:18080}" \
    --rtsp-base-url "${RTSP_BASE:-rtsp://127.0.0.1:8554}" \
    --video-pool "${VIDEO_POOL_DIR:-$REMOTE_ROOT/video_pool}" \
    --overlay-report-enabled \
    --detector-mode real \
    --yolo-model "${YOLO_MODEL:-yolo26n-pose.pt}" \
    --action-model "${ACTION_MODEL:-runs/evaluation_manifest_v2_bbox54_balanced/retrained_best.pt}" \
    --publisher mqtt \
    --mqtt-host "${MQTT_HOST:-15.165.248.37}" \
    --mqtt-port "${MQTT_PORT:-1883}" \
    --mqtt-topic safety/events \
    --skip-simulated-ffmpeg \
    --tracking-stability-fallback \
    --mjpeg-enabled \
    --mjpeg-fps 15 \
    --mjpeg-width 1280 \
    --mjpeg-height 720 \
    --frame-rate 30 \
    --domain '' \
    > ai_runner_rollback.log 2>&1 </dev/null &

echo "[rollback] runner pid=$!"
echo "[rollback] Verify: grep nearDupSuppressMode runs/registered_cameras/runner*.log | tail"
