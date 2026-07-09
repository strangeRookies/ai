#!/bin/bash
set -e

# Set variables
REMOTE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MQTT_HOST="${1:-15.165.248.37}"
MQTT_PORT="${2:-1883}"
DEFAULT_ACTION_MODEL="runs/evaluation_manifest_v2_bbox54_balanced/retrained_best.pt"
ACTION_MODEL="${3:-${ACTION_MODEL:-$DEFAULT_ACTION_MODEL}}"
DEFAULT_YOLO_MODEL="yolo26n-pose.pt"
TENSORRT_YOLO_MODEL="yolo26n-pose.engine"
USE_TENSORRT="${USE_TENSORRT:-false}"
if [ "$USE_TENSORRT" = "true" ] || [ "$USE_TENSORRT" = "1" ]; then
    YOLO_MODEL="${YOLO_MODEL_PATH:-${YOLO_MODEL:-$TENSORRT_YOLO_MODEL}}"
else
    YOLO_MODEL="${YOLO_MODEL_PATH:-${YOLO_MODEL:-$DEFAULT_YOLO_MODEL}}"
fi

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

if [ -f .env ]; then
    echo "[start_ai_stable] Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

if [ ! -f "$ACTION_MODEL" ]; then
    echo "[start_ai_stable][error] ACTION_MODEL checkpoint not found: $ACTION_MODEL"
    exit 1
fi

if [ ! -f "$YOLO_MODEL" ]; then
    if [ "$YOLO_MODEL" = "$TENSORRT_YOLO_MODEL" ] && [ -f "$DEFAULT_YOLO_MODEL" ]; then
        echo "[start_ai_stable][warning] TensorRT requested but engine not found: $YOLO_MODEL"
        echo "[start_ai_stable][warning] Falling back to Torch model: $DEFAULT_YOLO_MODEL"
        YOLO_MODEL="$DEFAULT_YOLO_MODEL"
    else
        echo "[start_ai_stable][error] YOLO_MODEL not found: $YOLO_MODEL"
        exit 1
    fi
fi

echo "[start_ai_stable] Using ACTION_MODEL=$ACTION_MODEL"
echo "[start_ai_stable] Using YOLO_MODEL=$YOLO_MODEL"

WEBRTC_SYNC_ARGS=()
if [ "${AI_WEBRTC_SYNC_ENABLED:-false}" = "true" ] || [ "${AI_WEBRTC_SYNC_ENABLED:-false}" = "1" ]; then
    WEBRTC_SYNC_ARGS=(
        --webrtc-sync-enabled
        --webrtc-sync-host "${AI_WEBRTC_SYNC_HOST:-0.0.0.0}"
        --webrtc-sync-base-port "${AI_WEBRTC_SYNC_BASE_PORT:-8090}"
    )
fi
echo "[start_ai_stable] Starting start_simulated_rtsp_from_folder.py..."
nohup python scripts/start_simulated_rtsp_from_folder.py \
    --video-dir /home/$USER/yolo_training/ai_fall_experiments/data/raw \
    --backend-url http://127.0.0.1:18080 \
    --rtsp-host 127.0.0.1 \
    --rtsp-port 8554 \
    --poll-interval 30 \
    --ffmpeg-mode auto \
    --domain outside > publisher.log 2>&1 </dev/null &

sleep 8

echo "[start_ai_stable] Starting run_registered_cameras.py..."
nohup python scripts/run_registered_cameras.py \
    --backend-base-url http://127.0.0.1:18080 \
    --rtsp-base-url rtsp://127.0.0.1:8554 \
    --video-pool /home/$USER/yolo_training/ai_fall_experiments/data/raw \
    --overlay-report-enabled \
    --detector-mode real \
    --yolo-model "$YOLO_MODEL" \
    --action-model "$ACTION_MODEL" \
    --publisher mqtt \
    --mqtt-host "$MQTT_HOST" \
    --mqtt-port "$MQTT_PORT" \
    --mqtt-topic safety/events \
    "${WEBRTC_SYNC_ARGS[@]}" \
    --skip-simulated-ffmpeg \
    --domain outside > ai_runner.log 2>&1 </dev/null &

echo "[start_ai_stable] All processes spawned in background."
