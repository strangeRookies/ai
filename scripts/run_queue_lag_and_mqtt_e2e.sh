#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Experiment 08 – Frame Queue Lag & MQTT E2E Latency
#
# 서브커맨드:
#   queue  – LatestFrameQueue drop/lag 측정 (MQTT dry-run)
#   mqtt   – MQTT 브로커 실제 발행 E2E 레이턴시 측정
#   both   – queue → mqtt 순서로 실행 (기본값)
#
# 사용 예:
#   CAMERA_IDS="cam_03,cam_04" \
#   PT="$ROOT/yolo26n-pose.pt" \
#   ENGINE="$ROOT/yolo26n-pose.engine" \
#   ACTION_MODEL="$ROOT/runs/evaluation_feature_dim/feature54/retrained_best.pt" \
#   MANAGE_STREAMS=never \
#   MAX_FRAMES=1800 \
#   FRAME_QUEUE_MAXSIZE=3 \
#   bash scripts/run_queue_lag_and_mqtt_e2e.sh queue
# ============================================================

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUBCOMMAND="${1:-both}"

# ── 카메라 / RTSP ────────────────────────────────────────────
CAMERA_IDS="${CAMERA_IDS:-cam_03,cam_04}"
RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://127.0.0.1:8554}"

# ── 모델 경로 ────────────────────────────────────────────────
PT="${PT:-$ROOT/yolo26n-pose.pt}"
ENGINE="${ENGINE:-$ROOT/yolo26n-pose.engine}"
ACTION_MODEL="${ACTION_MODEL:-$ROOT/runs/evaluation_feature_dim/feature54/retrained_best.pt}"

# ── 추론 파라미터 ─────────────────────────────────────────────
MAX_FRAMES="${MAX_FRAMES:-1800}"
DEVICE="${DEVICE:-0}"
ACTION_DEVICE="${ACTION_DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
DETECTOR_CONF="${DETECTOR_CONF:-0.15}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-30}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-15}"
FAINT_THRESHOLD="${FAINT_THRESHOLD:-0.5}"
MIN_CONSECUTIVE_FAINT="${MIN_CONSECUTIVE_FAINT:-2}"
CAMERA_COOLDOWN_SECONDS="${CAMERA_COOLDOWN_SECONDS:-0}"

# ── Queue 실험 파라미터 ──────────────────────────────────────
FRAME_QUEUE_MAXSIZE="${FRAME_QUEUE_MAXSIZE:-3}"
INFER_WARMUP_FRAMES="${INFER_WARMUP_FRAMES:-50}"
INFER_METRICS_EVERY_N="${INFER_METRICS_EVERY_N:-100}"

# ── MQTT E2E 파라미터 ────────────────────────────────────────
MQTT_HOST="${MQTT_HOST:-127.0.0.1}"
MQTT_PORT="${MQTT_PORT:-1883}"
MQTT_TOPIC_PREFIX="${MQTT_TOPIC_PREFIX:-shields}"
MQTT_USERNAME="${MQTT_USERNAME:-}"
MQTT_PASSWORD="${MQTT_PASSWORD:-}"

# ── Stream 관리 ──────────────────────────────────────────────
MANAGE_STREAMS="${MANAGE_STREAMS:-auto}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/video_pool}"
CHROMAKEY_VIDEO_DIR="${CHROMAKEY_VIDEO_DIR:-}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8080}"
PUBLISHER_FFMPEG_MODE="${PUBLISHER_FFMPEG_MODE:-auto}"
PUBLISHER_POLL_INTERVAL="${PUBLISHER_POLL_INTERVAL:-2}"
STREAM_READY_TIMEOUT="${STREAM_READY_TIMEOUT:-60}"

# ── 출력 ─────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/wiki_metrics/$RUN_ID/08_queue_latency}"

mkdir -p "$OUTPUT_DIR"

# ── 카메라 배열 파싱 ─────────────────────────────────────────
IFS=',' read -r -a CAMERAS <<< "$CAMERA_IDS"

if [[ "${#CAMERAS[@]}" -lt 1 ]]; then
  echo "[ERROR] CAMERA_IDS가 비어 있습니다."
  exit 1
fi

for i in "${!CAMERAS[@]}"; do
  CAMERAS[$i]="$(echo "${CAMERAS[$i]}" | xargs)"
  if [[ -z "${CAMERAS[$i]}" ]]; then
    echo "[ERROR] 빈 camera ID가 포함되어 있습니다: $CAMERA_IDS"
    exit 1
  fi
done

# ── 상태 변수 ─────────────────────────────────────────────────
PUBLISHER_PID=""
PUBLISHER_STARTED_BY_SCRIPT=0
declare -a ACTIVE_PIDS=()

# ── 유틸 함수 ─────────────────────────────────────────────────
log()  { printf '[q08] %s\n' "$*"; }
warn() { printf '[q08][WARN] %s\n' "$*" >&2; }
die()  { printf '[q08][ERROR] %s\n' "$*" >&2; exit 1; }

terminate_pid() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
      sleep 0.25
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup_workers() {
  [[ "${#ACTIVE_PIDS[@]}" -eq 0 ]] && return
  log "Stopping workers..."
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  cleanup_workers
  if [[ "$PUBLISHER_STARTED_BY_SCRIPT" -eq 1 ]] && [[ -n "$PUBLISHER_PID" ]]; then
    log "Stopping simulated RTSP publisher pid=$PUBLISHER_PID"
    terminate_pid "$PUBLISHER_PID"
  fi
  log "Cleanup done. exit_code=$exit_code"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

camera_url() {
  printf '%s/%s' "${RTSP_BASE_URL%/}" "$1"
}

stream_ready() {
  timeout 8 ffprobe \
    -v error -rtsp_transport tcp \
    -select_streams v:0 \
    -show_entries stream=codec_name \
    -of default=noprint_wrappers=1 \
    "$(camera_url "$1")" >/dev/null 2>&1
}

all_streams_ready() {
  local cam
  for cam in "${CAMERAS[@]}"; do
    stream_ready "$cam" || return 1
  done
  return 0
}

wait_for_streams() {
  local deadline=$(( SECONDS + STREAM_READY_TIMEOUT ))
  while (( SECONDS < deadline )); do
    local ok=0
    for cam in "${CAMERAS[@]}"; do
      stream_ready "$cam" && ok=$(( ok + 1 ))
    done
    log "RTSP readiness: ${ok}/${#CAMERAS[@]}"
    if [[ "$ok" -eq "${#CAMERAS[@]}" ]]; then
      for cam in "${CAMERAS[@]}"; do log "READY $(camera_url "$cam")"; done
      return 0
    fi
    sleep 2
  done
  die "RTSP streams were not ready within ${STREAM_READY_TIMEOUT}s"
}

start_simulated_publisher() {
  if pgrep -af "start_simulated_rtsp_from_folder.py" \
      | grep -v "pgrep -af" >/dev/null 2>&1; then
    die "기존 simulated RTSP publisher가 실행 중입니다. MANAGE_STREAMS=never 또는 기존 publisher 종료 후 재시도하세요."
  fi

  local -a args=(
    python -u scripts/start_simulated_rtsp_from_folder.py
    --video-dir "$VIDEO_DIR"
    --backend-url "$BACKEND_URL"
    --rtsp-host "127.0.0.1"
    --rtsp-port "8554"
    --poll-interval "$PUBLISHER_POLL_INTERVAL"
    --ffmpeg-mode "$PUBLISHER_FFMPEG_MODE"
    --loop
  )
  [[ -n "$CHROMAKEY_VIDEO_DIR" ]] && args+=(--chromakey-video-dir "$CHROMAKEY_VIDEO_DIR")

  log "Starting simulated RTSP publisher → $OUTPUT_DIR/publisher.log"
  ( cd "$ROOT"; exec "${args[@]}" ) >"$OUTPUT_DIR/publisher.log" 2>&1 &
  PUBLISHER_PID=$!
  PUBLISHER_STARTED_BY_SCRIPT=1
  sleep 2

  kill -0 "$PUBLISHER_PID" 2>/dev/null || {
    cat "$OUTPUT_DIR/publisher.log"
    die "simulated RTSP publisher exited during startup."
  }
  log "Publisher pid=$PUBLISHER_PID"
}

# ── preflight ─────────────────────────────────────────────────
preflight() {
  require_command python
  require_command ffprobe
  require_command timeout
  require_file "$PT"
  require_file "$ENGINE"
  require_file "$ACTION_MODEL"
  require_file "$ROOT/scripts/run_rtsp_inference.py"

  {
    echo "run_id=$RUN_ID"
    echo "subcommand=$SUBCOMMAND"
    echo "root=$ROOT"
    echo "camera_ids=$CAMERA_IDS"
    echo "pt=$PT"
    echo "engine=$ENGINE"
    echo "action_model=$ACTION_MODEL"
    echo "max_frames=$MAX_FRAMES"
    echo "frame_queue_maxsize=$FRAME_QUEUE_MAXSIZE"
    echo "manage_streams=$MANAGE_STREAMS"
    echo "mqtt_host=$MQTT_HOST"
    echo "mqtt_port=$MQTT_PORT"
    echo "git_sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  } | tee "$OUTPUT_DIR/experiment_config.txt"
}

# ── 스트림 준비 ───────────────────────────────────────────────
prepare_streams() {
  case "$MANAGE_STREAMS" in
    never)  log "Using existing RTSP streams (MANAGE_STREAMS=never)" ;;
    always) start_simulated_publisher ;;
    auto)
      if all_streams_ready; then
        log "All RTSP streams already ready; skipping publisher start."
      else
        log "Some streams not ready; starting publisher."
        start_simulated_publisher
      fi
      ;;
    *) die "MANAGE_STREAMS must be auto, always, or never" ;;
  esac
  wait_for_streams
}

# ── 동시 Worker 실행 ──────────────────────────────────────────
run_concurrent_workers() {
  local label="$1"
  local model_path="$2"
  shift 2
  local extra_args=("$@")

  local out_dir="$OUTPUT_DIR/$label"
  mkdir -p "$out_dir"

  log "Starting concurrent workers label=$label cameras=${CAMERAS[*]}"
  local start_ns
  start_ns="$(date +%s%N)"
  ACTIVE_PIDS=()

  for cam in "${CAMERAS[@]}"; do
    local rtsp_url summary_path prediction_path log_path pid
    rtsp_url="$(camera_url "$cam")"
    summary_path="$out_dir/${cam}_summary.json"
    prediction_path="$out_dir/${cam}_predictions.jsonl"
    log_path="$out_dir/${cam}.log"

    (
      cd "$ROOT"
      exec python -u scripts/run_rtsp_inference.py \
        --rtsp-url "$rtsp_url" \
        --camera-id "${cam}_${label}" \
        --camera-login-id "${cam}_${label}" \
        --max-frames "$MAX_FRAMES" \
        --detector-mode real \
        --yolo-model "$model_path" \
        --device "$DEVICE" \
        --imgsz "$IMGSZ" \
        --detector-conf "$DETECTOR_CONF" \
        --action-model "$ACTION_MODEL" \
        --action-device "$ACTION_DEVICE" \
        --classifier-input keypoints \
        --sequence-length "$SEQUENCE_LENGTH" \
        --sequence-stride "$SEQUENCE_STRIDE" \
        --faint-threshold "$FAINT_THRESHOLD" \
        --min-consecutive-faint "$MIN_CONSECUTIVE_FAINT" \
        --camera-cooldown-seconds "$CAMERA_COOLDOWN_SECONDS" \
        --frame-queue-maxsize "$FRAME_QUEUE_MAXSIZE" \
        --infer-warmup-frames "$INFER_WARMUP_FRAMES" \
        --infer-metrics-every-n "$INFER_METRICS_EVERY_N" \
        --evaluation-log "$prediction_path" \
        --output "$summary_path" \
        "${extra_args[@]}"
    ) >"$log_path" 2>&1 &

    pid=$!
    ACTIVE_PIDS+=("$pid")
    log "Started label=$label camera=$cam pid=$pid"
  done

  local failed=0
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" || failed=1
  done
  ACTIVE_PIDS=()

  local end_ns
  end_ns="$(date +%s%N)"
  python3 -c "print(f'{($end_ns - $start_ns) / 1_000_000_000:.6f}')" \
    > "$out_dir/wall_seconds.txt"

  if [[ "$failed" -ne 0 ]]; then
    for cam in "${CAMERAS[@]}"; do
      echo "===== $out_dir/${cam}.log ====="
      tail -n 80 "$out_dir/${cam}.log" || true
    done
    die "One or more $label workers failed."
  fi

  for cam in "${CAMERAS[@]}"; do
    [[ -f "$out_dir/${cam}_summary.json" ]] \
      || die "Missing summary: $out_dir/${cam}_summary.json"
  done

  local wall
  wall="$(cat "$out_dir/wall_seconds.txt")"
  log "Completed label=$label wall_seconds=$wall"
}

# ── 실험 A: Queue lag (dry-run) ───────────────────────────────
run_queue_experiment() {
  log "========================================================"
  log "Queue Lag Experiment (dry-run, maxsize=$FRAME_QUEUE_MAXSIZE)"
  log "========================================================"
  run_concurrent_workers "queue_pt"  "$PT"     --dry-run
  sleep 3
  run_concurrent_workers "queue_trt" "$ENGINE" --dry-run
}

# ── 실험 B: MQTT E2E ──────────────────────────────────────────
run_mqtt_experiment() {
  log "========================================================"
  log "MQTT E2E Latency Experiment (live publish)"
  log "========================================================"

  local -a mqtt_args=(
    --mqtt-host "$MQTT_HOST"
    --mqtt-port "$MQTT_PORT"
    --mqtt-topic-prefix "$MQTT_TOPIC_PREFIX"
  )
  [[ -n "$MQTT_USERNAME" ]] && mqtt_args+=(--mqtt-username "$MQTT_USERNAME")
  [[ -n "$MQTT_PASSWORD" ]] && mqtt_args+=(--mqtt-password "$MQTT_PASSWORD")

  run_concurrent_workers "mqtt_pt"  "$PT"     "${mqtt_args[@]}"
  sleep 3
  run_concurrent_workers "mqtt_trt" "$ENGINE" "${mqtt_args[@]}"
}

# ── 보고서 생성 ───────────────────────────────────────────────
build_report() {
  python3 - "$OUTPUT_DIR" "$SUBCOMMAND" "${CAMERAS[@]}" <<'PY'
from __future__ import annotations
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
subcommand = sys.argv[2]
camera_ids = sys.argv[3:]

if subcommand == "queue":
    pairs = [("queue_pt", "PT dry-run"), ("queue_trt", "TRT dry-run")]
elif subcommand == "mqtt":
    pairs = [("mqtt_pt", "PT+MQTT"), ("mqtt_trt", "TRT+MQTT")]
else:
    pairs = [
        ("queue_pt", "PT dry-run"), ("queue_trt", "TRT dry-run"),
        ("mqtt_pt", "PT+MQTT"),    ("mqtt_trt", "TRT+MQTT"),
    ]

rows: list[dict] = []
aggregates: dict[str, dict] = {}

for label, desc in pairs:
    d = root / label
    if not d.exists():
        print(f"[WARN] {d} not found, skipping")
        continue
    wall_txt = d / "wall_seconds.txt"
    wall = float(wall_txt.read_text().strip()) if wall_txt.exists() else 0.0
    summaries: list[dict] = []
    for cam in camera_ids:
        sp = d / f"{cam}_summary.json"
        if not sp.exists():
            print(f"[WARN] {sp} missing")
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        summaries.append(s)
        rows.append({
            "experiment": desc, "label": label, "camera_id": cam,
            "source_mode": s.get("source_mode", ""),
            "frame_queue_maxsize": int(s.get("frame_queue_maxsize") or 0),
            "frames_processed": int(s.get("frames_processed") or 0),
            "effective_fps": float(s.get("effective_fps") or 0),
            "avg_yolo_ms": float(s.get("avg_yolo_inference_ms") or 0),
            "p50_yolo_ms": float(s.get("p50_yolo_inference_ms") or 0),
            "p95_yolo_ms": float(s.get("p95_yolo_inference_ms") or 0),
            "avg_lstm_ms": float(s.get("avg_lstm_inference_ms") or 0),
            "avg_total_ms": float(s.get("avg_total_frame_ms") or 0),
            "latest_ai_latency_ms": float(s.get("latest_ai_latency_ms") or 0),
            "latest_publish_latency_ms": float(s.get("latest_publish_latency_ms") or 0),
            "dropped_frames": int(s.get("latest_dropped_frame_count") or 0),
            "events_generated": int(s.get("events_generated") or 0),
            "events_publish_succeeded": int(s.get("events_publish_succeeded") or 0),
            "events_publish_failed": int(s.get("events_publish_failed") or 0),
            "alert_delivery_result": str(s.get("alert_delivery_result") or ""),
        })
    if not summaries:
        continue
    total = max(sum(int(s.get("frames_processed") or 0) for s in summaries), 1)
    total_drop = sum(int(s.get("latest_dropped_frame_count") or 0) for s in summaries)
    total_events = sum(int(s.get("events_generated") or 0) for s in summaries)
    total_ok = sum(int(s.get("events_publish_succeeded") or 0) for s in summaries)
    total_fail = sum(int(s.get("events_publish_failed") or 0) for s in summaries)
    w_yolo = sum(float(s.get("avg_yolo_inference_ms") or 0) * int(s.get("frames_processed") or 0) for s in summaries) / total
    w_total = sum(float(s.get("avg_total_frame_ms") or 0) * int(s.get("frames_processed") or 0) for s in summaries) / total
    worst_p95 = max(float(s.get("p95_yolo_inference_ms") or 0) for s in summaries)
    aggregates[label] = {
        "label": label, "description": desc, "camera_count": len(summaries),
        "total_frames": total, "wall_seconds": wall,
        "aggregate_fps": total / max(wall, 1e-9),
        "weighted_avg_yolo_ms": w_yolo, "worst_camera_p95_yolo_ms": worst_p95,
        "weighted_avg_total_ms": w_total,
        "total_dropped_frames": total_drop,
        "drop_rate_pct": total_drop / total * 100.0,
        "total_events": total_events, "total_publish_ok": total_ok,
        "total_publish_fail": total_fail,
    }

if rows:
    csv_path = root / "per_camera_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

agg_path = root / "aggregate_results.json"
agg_path.write_text(json.dumps(aggregates, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Experiment 08 – Frame Queue Lag & MQTT E2E Latency", "",
    f"- Cameras: `{', '.join(camera_ids)}`",
    f"- Subcommand: `{subcommand}`", "",
    "## Aggregate Results", "",
    "| Experiment | Frames | Agg FPS | YOLO avg ms | YOLO p95 ms | Total avg ms | Drops | Drop% | Events | Pub OK | Pub Fail |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for agg in aggregates.values():
    lines.append(
        f"| {agg['description']} | {int(agg['total_frames'])} "
        f"| {float(agg['aggregate_fps']):.2f} "
        f"| {float(agg['weighted_avg_yolo_ms']):.2f} "
        f"| {float(agg['worst_camera_p95_yolo_ms']):.2f} "
        f"| {float(agg['weighted_avg_total_ms']):.2f} "
        f"| {int(agg['total_dropped_frames'])} "
        f"| {float(agg['drop_rate_pct']):.2f}% "
        f"| {int(agg['total_events'])} "
        f"| {int(agg['total_publish_ok'])} "
        f"| {int(agg['total_publish_fail'])} |"
    )
lines += [
    "", "## Per-Camera Results", "",
    "| Experiment | Camera | FPS | YOLO avg | YOLO p95 | Total avg | AI lat ms | Pub lat ms | Drops | Delivery |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    lines.append(
        f"| {r['experiment']} | {r['camera_id']} "
        f"| {float(r['effective_fps']):.2f} | {float(r['avg_yolo_ms']):.2f} "
        f"| {float(r['p95_yolo_ms']):.2f} | {float(r['avg_total_ms']):.2f} "
        f"| {float(r['latest_ai_latency_ms']):.2f} "
        f"| {float(r['latest_publish_latency_ms']):.2f} "
        f"| {int(r['dropped_frames'])} | {r['alert_delivery_result']} |"
    )
if "queue_pt" in aggregates and "queue_trt" in aggregates:
    pt = aggregates["queue_pt"]; trt = aggregates["queue_trt"]
    def pct(a, b): return (b - a) / max(abs(a), 1e-9) * 100.0
    lines += [
        "", "## Queue Experiment: PT vs TRT", "",
        "| Metric | PT dry-run | TRT dry-run | Change |",
        "|---|---:|---:|---:|",
        f"| Aggregate FPS | {float(pt['aggregate_fps']):.2f} | {float(trt['aggregate_fps']):.2f} | {pct(pt['aggregate_fps'], trt['aggregate_fps']):+.2f}% |",
        f"| YOLO avg ms | {float(pt['weighted_avg_yolo_ms']):.2f} | {float(trt['weighted_avg_yolo_ms']):.2f} | {pct(pt['weighted_avg_yolo_ms'], trt['weighted_avg_yolo_ms']):+.2f}% |",
        f"| YOLO p95 ms | {float(pt['worst_camera_p95_yolo_ms']):.2f} | {float(trt['worst_camera_p95_yolo_ms']):.2f} | {pct(pt['worst_camera_p95_yolo_ms'], trt['worst_camera_p95_yolo_ms']):+.2f}% |",
        f"| Dropped frames | {int(pt['total_dropped_frames'])} | {int(trt['total_dropped_frames'])} | {int(trt['total_dropped_frames']) - int(pt['total_dropped_frames']):+d} |",
        f"| Drop rate | {float(pt['drop_rate_pct']):.2f}% | {float(trt['drop_rate_pct']):.2f}% | {pct(pt['drop_rate_pct'], trt['drop_rate_pct']):+.2f}% |",
    ]
lines += ["", f"Saved to: `{root}`", ""]
report = root / "queue_lag_and_mqtt_e2e_report.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved: {report}")
print(f"Saved: {agg_path}")
PY
}

# ── main ──────────────────────────────────────────────────────
main() {
  cd "$ROOT"
  log "Subcommand: $SUBCOMMAND"
  preflight
  prepare_streams

  case "$SUBCOMMAND" in
    queue) run_queue_experiment ;;
    mqtt)  run_mqtt_experiment ;;
    both)  run_queue_experiment; sleep 5; run_mqtt_experiment ;;
    *)     die "Unknown subcommand: '$SUBCOMMAND'. Use: queue | mqtt | both" ;;
  esac

  build_report
  log "Experiment 08 complete: $OUTPUT_DIR"
}

main "$@"