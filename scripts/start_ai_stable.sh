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

# Tracking association production defaults (offline A/B + cam_03 canary, 2026-07).
# Rollback: scripts/rollback_tracking_suppression.sh (none + 0.25).
export NEAR_DUP_SUPPRESS_MODE="${NEAR_DUP_SUPPRESS_MODE:-hybrid_kp}"
export SIMPLE_TRACK_NEW_TRACK_THRESH="${SIMPLE_TRACK_NEW_TRACK_THRESH:-0.30}"
export NEAR_DUP_SORT_BY_CONF="${NEAR_DUP_SORT_BY_CONF:-1}"
echo "[start_ai_stable] Tracking defaults: NEAR_DUP_SUPPRESS_MODE=$NEAR_DUP_SUPPRESS_MODE SIMPLE_TRACK_NEW_TRACK_THRESH=$SIMPLE_TRACK_NEW_TRACK_THRESH"

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
VIDEO_POOL_DIR="${VIDEO_POOL_DIR:-$REMOTE_ROOT/video_pool}"
if [ ! -d "$VIDEO_POOL_DIR" ]; then
    echo "[start_ai_stable][error] VIDEO_POOL_DIR not found: $VIDEO_POOL_DIR"
    exit 1
fi
POOL_MP4_COUNT=$(find "$VIDEO_POOL_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)
echo "[start_ai_stable] Using VIDEO_POOL_DIR=$VIDEO_POOL_DIR (mp4_count=$POOL_MP4_COUNT)"
if [ "$POOL_MP4_COUNT" -lt 1 ]; then
    echo "[start_ai_stable][error] No mp4 files in VIDEO_POOL_DIR"
    exit 1
fi

echo "[start_ai_stable] Starting start_simulated_rtsp_from_folder.py..."
# Use project video_pool with distinct demo clips. Do NOT use --domain outside here:
# that filter previously collapsed onto outdoor_swoon chromakey symlinks and caused
# multiple cameras to publish the same content.
# Empty --domain disables DEFAULT_STREAM_DOMAIN=outside path substring filtering so
# flat names like faint_01.mp4 / fall_01.mp4 under video_pool are accepted.
nohup env VIDEO_DOMAIN= python scripts/start_simulated_rtsp_from_folder.py \
    --video-dir "$VIDEO_POOL_DIR" \
    --backend-url http://safety-backend-alb-eks-1607216893.ap-northeast-2.elb.amazonaws.com \
    --rtsp-host 127.0.0.1 \
    --rtsp-port 8554 \
    --poll-interval 15 \
    --ffmpeg-mode auto \
    --domain '' \
    > publisher.log 2>&1 </dev/null &

sleep 8

echo "[start_ai_stable] Starting run_registered_cameras.py..."
nohup env VIDEO_DOMAIN= \
    NEAR_DUP_SUPPRESS_MODE="${NEAR_DUP_SUPPRESS_MODE}" \
    SIMPLE_TRACK_NEW_TRACK_THRESH="${SIMPLE_TRACK_NEW_TRACK_THRESH}" \
    NEAR_DUP_SORT_BY_CONF="${NEAR_DUP_SORT_BY_CONF}" \
    python scripts/run_registered_cameras.py \
    --backend-base-url http://safety-backend-alb-eks-1607216893.ap-northeast-2.elb.amazonaws.com \
    --rtsp-base-url rtsp://127.0.0.1:8554 \
    --video-pool "$VIDEO_POOL_DIR" \
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
    --domain '' \
    > ai_runner.log 2>&1 </dev/null &

echo "[start_ai_stable] All processes spawned in background."
