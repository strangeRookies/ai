#!/usr/bin/env bash
# Ubuntu/GPU PC: Queue Lag 2-cam AB + optional MQTT E2E (Bash only).
# Usage:
#   cd /home/welabs/yolo_training/strange_ai_lstm  # or repo root
#   source .venv/bin/activate
#   export ROOT="$PWD"
#   bash scripts/run_queue_lag_and_mqtt_e2e.sh queue   # step 2 only
#   bash scripts/run_queue_lag_and_mqtt_e2e.sh e2e     # step 4 only
#   bash scripts/run_queue_lag_and_mqtt_e2e.sh all

set -Eeuo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${1:-queue}"

log() { printf '[wiki-metrics] %s\n' "$*"; }

run_queue_ab() {
  export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/wiki_metrics/$(date +%Y%m%d_%H%M%S)/08_queue_latency}"
  mkdir -p "$OUTPUT_DIR"
  log "OUTPUT_DIR=$OUTPUT_DIR"

  CAMERA_IDS="${CAMERA_IDS:-cam_03,cam_04}" \
  PT="${PT:-$ROOT/yolo26n-pose.pt}" \
  ENGINE="${ENGINE:-$ROOT/yolo26n-pose.engine}" \
  ACTION_MODEL="${ACTION_MODEL:-$ROOT/runs/evaluation_feature_dim/feature54/retrained_best.pt}" \
  MANAGE_STREAMS="${MANAGE_STREAMS:-never}" \
  MAX_FRAMES="${MAX_FRAMES:-1800}" \
  FRAME_QUEUE_MAXSIZE="${FRAME_QUEUE_MAXSIZE:-3}" \
  bash "$ROOT/scripts/run_2cam_rtsp_ab_metrics.sh"

  python3 - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for backend in ("pytorch", "tensorrt"):
    print(f"\n===== {backend} =====")
    sub = root / backend
    if not sub.is_dir():
        continue
    for path in sorted(sub.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(path.name)
        for key in (
            "avg_queue_lag_ms",
            "p50_queue_lag_ms",
            "p95_queue_lag_ms",
            "max_queue_lag_ms",
            "max_queue_depth",
            "latest_dropped_frame_count",
        ):
            print(f"  {key}: {data.get(key)}")
PY
}

run_mqtt_e2e() {
  export E2E_OUT="${E2E_OUT:-$ROOT/runs/wiki_metrics/$(date +%Y%m%d_%H%M%S)/09_alert_e2e}"
  mkdir -p "$E2E_OUT"
  log "E2E_OUT=$E2E_OUT"

  log "Existing workers:"
  pgrep -af "run_registered_cameras.py|run_rtsp_inference.py" || true

  if [[ "${KILL_EXISTING_WORKERS:-0}" == "1" ]]; then
    pkill -TERM -f "scripts/run_registered_cameras.py" || true
    pkill -TERM -f "scripts/run_rtsp_inference.py" || true
    sleep 3
  fi

  python3 -u "$ROOT/scripts/measure_mqtt_alert_latency.py" \
    --host "${MQTT_HOST:-127.0.0.1}" \
    --port "${MQTT_PORT:-1883}" \
    --topic "${MQTT_EVENT_TOPIC:-event}" \
    --duration-seconds "${PROBE_DURATION_SECONDS:-600}" \
    --max-events "${PROBE_MAX_EVENTS:-50}" \
    --output-dir "$E2E_OUT" \
    > "$E2E_OUT/mqtt_probe.log" 2>&1 &
  PROBE_PID=$!
  log "MQTT probe PID=$PROBE_PID"

  CAM_LIST="${E2E_CAMERAS:-cam_03 cam_04}"
  for CAM in $CAM_LIST; do
    python3 -u "$ROOT/scripts/run_rtsp_inference.py" \
      --rtsp-url "${RTSP_BASE_URL:-rtsp://127.0.0.1:8554}/$CAM" \
      --camera-id "$CAM" \
      --camera-login-id "$CAM" \
      --max-frames "${E2E_MAX_FRAMES:-5400}" \
      --frame-queue-maxsize "${FRAME_QUEUE_MAXSIZE:-3}" \
      --detector-mode real \
      --yolo-model "${ENGINE:-$ROOT/yolo26n-pose.engine}" \
      --device "${DEVICE:-0}" \
      --imgsz "${IMGSZ:-640}" \
      --detector-conf "${DETECTOR_CONF:-0.15}" \
      --action-model "${ACTION_MODEL:-$ROOT/runs/evaluation_feature_dim/feature54/retrained_best.pt}" \
      --action-device "${ACTION_DEVICE:-0}" \
      --classifier-input keypoints \
      --sequence-length "${SEQUENCE_LENGTH:-30}" \
      --sequence-stride "${SEQUENCE_STRIDE:-15}" \
      --faint-threshold "${FAINT_THRESHOLD:-0.5}" \
      --min-consecutive-faint "${MIN_CONSECUTIVE_FAINT:-2}" \
      --camera-cooldown-seconds 0 \
      --publisher mqtt \
      --mqtt-host "${MQTT_HOST:-127.0.0.1}" \
      --mqtt-port "${MQTT_PORT:-1883}" \
      --mqtt-event-topic "${MQTT_EVENT_TOPIC:-event}" \
      --output "$E2E_OUT/${CAM}_summary.json" \
      > "$E2E_OUT/${CAM}.log" 2>&1 &
  done
  wait

  kill -TERM "$PROBE_PID" 2>/dev/null || true
  wait "$PROBE_PID" 2>/dev/null || true

  if [[ -f "$E2E_OUT/alert_latency_summary.json" ]]; then
    cat "$E2E_OUT/alert_latency_summary.json"
  fi

  python3 - "$E2E_OUT" <<'PY'
import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("cam_*_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print("\n", path.name)
    for key in (
        "avg_queue_lag_ms",
        "p50_queue_lag_ms",
        "p95_queue_lag_ms",
        "max_queue_lag_ms",
        "latest_dropped_frame_count",
        "avg_yolo_inference_ms",
        "p95_yolo_inference_ms",
    ):
        print(f"{key}: {data.get(key)}")
PY
}

case "$MODE" in
  queue) run_queue_ab ;;
  e2e) run_mqtt_e2e ;;
  all) run_queue_ab; run_mqtt_e2e ;;
  *)
    echo "Usage: $0 [queue|e2e|all]" >&2
    exit 1
    ;;
esac