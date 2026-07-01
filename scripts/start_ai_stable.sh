#!/bin/bash
set -e

# Set variables
REMOTE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MQTT_HOST="${1:-15.165.248.37}"
MQTT_PORT="${2:-1883}"

cd "$REMOTE_ROOT"

echo "[start_ai_stable] Starting MediaMTX..."
if docker ps --filter 'name=^mediamtx$' --format '{{.Names}}' | grep -q '^mediamtx$'; then
    echo "MediaMTX is already running."
else
    nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null &
fi

sleep 3

echo "[start_ai_stable] Activating virtual environment..."
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "[start_ai_stable][error] .venv/bin/activate not found!"
    exit 1
fi

echo "[start_ai_stable] Starting start_simulated_rtsp_from_folder.py..."
nohup python scripts/start_simulated_rtsp_from_folder.py \
    --video-dir /home/$USER/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
    --backend-url http://127.0.0.1:18080 \
    --rtsp-host 127.0.0.1 \
    --rtsp-port 8554 \
    --poll-interval 30 \
    --ffmpeg-mode copy > publisher.log 2>&1 </dev/null &

sleep 8

echo "[start_ai_stable] Starting run_registered_cameras.py..."
nohup python scripts/run_registered_cameras.py \
    --backend-base-url http://127.0.0.1:18080 \
    --rtsp-base-url rtsp://127.0.0.1:8554 \
    --video-pool /home/$USER/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
    --overlay-report-enabled \
    --detector-mode real \
    --yolo-model yolo26n-pose.pt \
    --publisher mqtt \
    --mqtt-host "$MQTT_HOST" \
    --mqtt-port "$MQTT_PORT" \
    --mqtt-topic safety/events \
    --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null &

echo "[start_ai_stable] All processes spawned in background."
