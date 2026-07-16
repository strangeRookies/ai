#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Two-camera concurrent PyTorch vs TensorRT RTSP benchmark
#
# 동작:
# 1. 필요 시 simulated RTSP publisher 실행
# 2. 지정한 카메라 2대 RTSP 준비 확인
# 3. PyTorch Worker 2개 동시 실행
# 4. TensorRT Worker 2개 동시 실행
# 5. 결과 비교 보고서 생성
# 6. 스크립트가 시작한 프로세스만 종료
# ============================================================

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CAMERA_IDS="${CAMERA_IDS:-cam_04,cam_05}"
RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://127.0.0.1:8554}"

PT="${PT:-$ROOT/yolo26n-pose.pt}"
ENGINE="${ENGINE:-$ROOT/yolo26n-pose.engine}"
ACTION_MODEL="${ACTION_MODEL:-$ROOT/runs/evaluation_feature_dim/feature54/retrained_best.pt}"

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

FRAME_QUEUE_MAXSIZE="${FRAME_QUEUE_MAXSIZE:-3}"
INFER_WARMUP_FRAMES="${INFER_WARMUP_FRAMES:-50}"
INFER_METRICS_EVERY_N="${INFER_METRICS_EVERY_N:-100}"

# auto:
#   두 RTSP가 이미 열려 있으면 기존 스트림 사용
#   하나라도 닫혀 있으면 simulated publisher 실행
#
# always:
#   publisher를 무조건 직접 실행
#   기존 publisher가 실행 중이면 lock 충돌 가능
#
# never:
#   RTSP를 직접 켜지 않고 기존 스트림만 사용
MANAGE_STREAMS="${MANAGE_STREAMS:-auto}"

VIDEO_DIR="${VIDEO_DIR:-$ROOT/video_pool}"
CHROMAKEY_VIDEO_DIR="${CHROMAKEY_VIDEO_DIR:-}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8080}"
PUBLISHER_FFMPEG_MODE="${PUBLISHER_FFMPEG_MODE:-auto}"
PUBLISHER_POLL_INTERVAL="${PUBLISHER_POLL_INTERVAL:-2}"
STREAM_READY_TIMEOUT="${STREAM_READY_TIMEOUT:-60}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/wiki_metrics/$RUN_ID/07_2cam_concurrent}"

mkdir -p "$OUTPUT_DIR"

IFS=',' read -r -a CAMERAS <<< "$CAMERA_IDS"

if [[ "${#CAMERAS[@]}" -ne 2 ]]; then
  echo "[ERROR] CAMERA_IDS에는 정확히 2개를 지정해야 합니다."
  echo "예: CAMERA_IDS=cam_04,cam_05"
  exit 1
fi

for i in "${!CAMERAS[@]}"; do
  CAMERAS[$i]="$(echo "${CAMERAS[$i]}" | xargs)"
  if [[ -z "${CAMERAS[$i]}" ]]; then
    echo "[ERROR] 빈 camera ID가 포함되어 있습니다: $CAMERA_IDS"
    exit 1
  fi
done

PUBLISHER_PID=""
PUBLISHER_STARTED_BY_SCRIPT=0
declare -a ACTIVE_BENCHMARK_PIDS=()

log() {
  printf '[2cam-ab] %s\n' "$*"
}

terminate_pid() {
  local pid="${1:-}"

  [[ -z "$pid" ]] && return 0

  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true

    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
      fi
      sleep 0.25
    done

    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup_benchmark_workers() {
  if [[ "${#ACTIVE_BENCHMARK_PIDS[@]}" -eq 0 ]]; then
    return
  fi

  log "Stopping benchmark workers..."

  for pid in "${ACTIVE_BENCHMARK_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  for pid in "${ACTIVE_BENCHMARK_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  ACTIVE_BENCHMARK_PIDS=()
}

cleanup() {
  local exit_code=$?

  trap - EXIT INT TERM

  cleanup_benchmark_workers

  if [[ "$PUBLISHER_STARTED_BY_SCRIPT" -eq 1 ]] && [[ -n "$PUBLISHER_PID" ]]; then
    log "Stopping simulated RTSP publisher pid=$PUBLISHER_PID"
    terminate_pid "$PUBLISHER_PID"
  fi

  log "Cleanup complete. exit_code=$exit_code"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_file() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file not found: $path"
    exit 1
  fi
}

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $command_name"
    exit 1
  fi
}

camera_url() {
  local camera_id="$1"
  printf '%s/%s' "${RTSP_BASE_URL%/}" "$camera_id"
}

stream_ready() {
  local camera_id="$1"
  local url

  url="$(camera_url "$camera_id")"

  timeout 8 ffprobe \
    -v error \
    -rtsp_transport tcp \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate \
    -of default=noprint_wrappers=1 \
    "$url" >/dev/null 2>&1
}

all_streams_ready() {
  local camera_id

  for camera_id in "${CAMERAS[@]}"; do
    if ! stream_ready "$camera_id"; then
      return 1
    fi
  done

  return 0
}

wait_for_streams() {
  local deadline
  local camera_id

  deadline=$((SECONDS + STREAM_READY_TIMEOUT))

  while (( SECONDS < deadline )); do
    local ready_count=0

    for camera_id in "${CAMERAS[@]}"; do
      if stream_ready "$camera_id"; then
        ready_count=$((ready_count + 1))
      fi
    done

    log "RTSP readiness: ${ready_count}/${#CAMERAS[@]}"

    if [[ "$ready_count" -eq "${#CAMERAS[@]}" ]]; then
      for camera_id in "${CAMERAS[@]}"; do
        log "READY $(camera_url "$camera_id")"
      done
      return 0
    fi

    sleep 2
  done

  echo "[ERROR] RTSP streams were not ready within ${STREAM_READY_TIMEOUT}s"

  for camera_id in "${CAMERAS[@]}"; do
    if stream_ready "$camera_id"; then
      echo "[READY] $(camera_url "$camera_id")"
    else
      echo "[NOT READY] $(camera_url "$camera_id")"
    fi
  done

  return 1
}

start_simulated_publisher() {
  local -a command_args

  if pgrep -af "start_simulated_rtsp_from_folder.py" \
      | grep -v "pgrep -af" >/dev/null 2>&1; then
    echo "[ERROR] 기존 simulated RTSP publisher가 실행 중입니다."
    echo "MANAGE_STREAMS=never로 기존 스트림을 사용하거나,"
    echo "기존 publisher를 종료한 후 다시 실행하세요."
    pgrep -af "start_simulated_rtsp_from_folder.py" || true
    exit 1
  fi

  command_args=(
    python -u scripts/start_simulated_rtsp_from_folder.py
    --video-dir "$VIDEO_DIR"
    --backend-url "$BACKEND_URL"
    --rtsp-host "127.0.0.1"
    --rtsp-port "8554"
    --poll-interval "$PUBLISHER_POLL_INTERVAL"
    --ffmpeg-mode "$PUBLISHER_FFMPEG_MODE"
    --loop
  )

  if [[ -n "$CHROMAKEY_VIDEO_DIR" ]]; then
    command_args+=(--chromakey-video-dir "$CHROMAKEY_VIDEO_DIR")
  fi

  log "Starting simulated RTSP publisher"
  log "Publisher log: $OUTPUT_DIR/simulated_rtsp_publisher.log"

  (
    cd "$ROOT"
    exec "${command_args[@]}"
  ) >"$OUTPUT_DIR/simulated_rtsp_publisher.log" 2>&1 &

  PUBLISHER_PID=$!
  PUBLISHER_STARTED_BY_SCRIPT=1

  sleep 2

  if ! kill -0 "$PUBLISHER_PID" 2>/dev/null; then
    echo "[ERROR] simulated RTSP publisher exited during startup."
    cat "$OUTPUT_DIR/simulated_rtsp_publisher.log"
    exit 1
  fi

  log "Publisher started pid=$PUBLISHER_PID"
}

preflight() {
  require_command python
  require_command ffprobe
  require_command timeout

  require_file "$PT"
  require_file "$ENGINE"
  require_file "$ACTION_MODEL"
  require_file "$ROOT/scripts/run_rtsp_inference.py"

  # max_frames가 RTSP를 offline 모드로 바꾸는 옛 코드인지 확인.
  # 이 조건이 남아 있으면 bounded live queue 실험이 아니므로 중단.
  if grep -Fq \
    'or getattr(args, "max_frames", 0) > 0' \
    "$ROOT/scripts/run_rtsp_inference.py"; then
    echo "[ERROR] run_rtsp_inference.py가 --max-frames RTSP 입력을 offline으로 처리합니다."
    echo "이 상태에서는 frame queue drop/latency 실험이 유효하지 않습니다."
    echo "앞서 적용한 offline/live source 판별 수정을 먼저 반영하세요."
    exit 1
  fi

  {
    echo "run_id=$RUN_ID"
    echo "root=$ROOT"
    echo "camera_ids=$CAMERA_IDS"
    echo "pt=$PT"
    echo "engine=$ENGINE"
    echo "action_model=$ACTION_MODEL"
    echo "max_frames=$MAX_FRAMES"
    echo "frame_queue_maxsize=$FRAME_QUEUE_MAXSIZE"
    echo "manage_streams=$MANAGE_STREAMS"
    echo "git_sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  } | tee "$OUTPUT_DIR/experiment_config.txt"
}

prepare_streams() {
  case "$MANAGE_STREAMS" in
    never)
      log "Using existing RTSP streams"
      ;;

    always)
      start_simulated_publisher
      ;;

    auto)
      if all_streams_ready; then
        log "Both RTSP streams already exist; publisher will not be started."
      else
        log "One or more streams are unavailable; starting publisher."
        start_simulated_publisher
      fi
      ;;

    *)
      echo "[ERROR] MANAGE_STREAMS must be auto, always, or never"
      exit 1
      ;;
  esac

  wait_for_streams
}

run_backend() {
  local backend_label="$1"
  local model_path="$2"

  local backend_dir="$OUTPUT_DIR/$backend_label"
  local start_ns
  local end_ns
  local wall_seconds
  local camera_id
  local pid
  local failed=0

  mkdir -p "$backend_dir"

  log "Starting concurrent backend=$backend_label cameras=${CAMERAS[*]}"

  start_ns="$(date +%s%N)"
  ACTIVE_BENCHMARK_PIDS=()

  for camera_id in "${CAMERAS[@]}"; do
    local rtsp_url
    local summary_path
    local prediction_path
    local log_path

    rtsp_url="$(camera_url "$camera_id")"
    summary_path="$backend_dir/${camera_id}_summary.json"
    prediction_path="$backend_dir/${camera_id}_predictions.jsonl"
    log_path="$backend_dir/${camera_id}.log"

    (
      cd "$ROOT"

      exec python -u scripts/run_rtsp_inference.py \
        --rtsp-url "$rtsp_url" \
        --camera-id "${camera_id}_${backend_label}" \
        --camera-login-id "${camera_id}_${backend_label}" \
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
        --dry-run \
        --output "$summary_path"
    ) >"$log_path" 2>&1 &

    pid=$!
    ACTIVE_BENCHMARK_PIDS+=("$pid")

    log "Started backend=$backend_label camera=$camera_id pid=$pid"
  done

  for pid in "${ACTIVE_BENCHMARK_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  ACTIVE_BENCHMARK_PIDS=()

  end_ns="$(date +%s%N)"

  wall_seconds="$(
    python - "$start_ns" "$end_ns" <<'PY'
import sys

start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])

print(f"{(end_ns - start_ns) / 1_000_000_000:.6f}")
PY
  )"

  echo "$wall_seconds" > "$backend_dir/wall_seconds.txt"

  if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] One or more $backend_label workers failed."

    for camera_id in "${CAMERAS[@]}"; do
      echo "===== $backend_dir/${camera_id}.log ====="
      tail -n 80 "$backend_dir/${camera_id}.log" || true
    done

    exit 1
  fi

  for camera_id in "${CAMERAS[@]}"; do
    if [[ ! -f "$backend_dir/${camera_id}_summary.json" ]]; then
      echo "[ERROR] Missing summary: $backend_dir/${camera_id}_summary.json"
      exit 1
    fi
  done

  log "Completed backend=$backend_label wall_seconds=$wall_seconds"
}

build_report() {
  python - "$OUTPUT_DIR" "${CAMERAS[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
camera_ids = sys.argv[2:]
backend_names = ("pytorch", "tensorrt")

rows: list[dict[str, object]] = []
backend_aggregates: dict[str, dict[str, float | int | str]] = {}

for backend in backend_names:
    backend_dir = root / backend
    wall_seconds = float((backend_dir / "wall_seconds.txt").read_text().strip())

    summaries: list[dict] = []

    for camera_id in camera_ids:
        summary_path = backend_dir / f"{camera_id}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)

        rows.append(
            {
                "backend": backend,
                "camera_id": camera_id,
                "actual_backend": summary.get("backend") or summary.get("runtime"),
                "frames_processed": int(summary.get("frames_processed") or 0),
                "effective_fps": float(summary.get("effective_fps") or 0.0),
                "avg_yolo_inference_ms": float(summary.get("avg_yolo_inference_ms") or 0.0),
                "p50_yolo_inference_ms": float(summary.get("p50_yolo_inference_ms") or 0.0),
                "p95_yolo_inference_ms": float(summary.get("p95_yolo_inference_ms") or 0.0),
                "avg_lstm_inference_ms": float(summary.get("avg_lstm_inference_ms") or 0.0),
                "avg_total_frame_ms": float(summary.get("avg_total_frame_ms") or 0.0),
                "dropped_frames": int(summary.get("latest_dropped_frame_count") or 0),
                "events_generated": int(summary.get("events_generated") or 0),
            }
        )

    total_frames = sum(int(s.get("frames_processed") or 0) for s in summaries)
    total_dropped = sum(int(s.get("latest_dropped_frame_count") or 0) for s in summaries)
    total_events = sum(int(s.get("events_generated") or 0) for s in summaries)

    weighted_yolo = (
        sum(
            float(s.get("avg_yolo_inference_ms") or 0.0)
            * int(s.get("frames_processed") or 0)
            for s in summaries
        )
        / max(total_frames, 1)
    )

    weighted_total = (
        sum(
            float(s.get("avg_total_frame_ms") or 0.0)
            * int(s.get("frames_processed") or 0)
            for s in summaries
        )
        / max(total_frames, 1)
    )

    backend_aggregates[backend] = {
        "backend": backend,
        "camera_count": len(camera_ids),
        "total_frames": total_frames,
        "wall_seconds": wall_seconds,
        "aggregate_fps": total_frames / max(wall_seconds, 1e-9),
        "weighted_avg_yolo_ms": weighted_yolo,
        "worst_camera_p95_yolo_ms": max(
            float(s.get("p95_yolo_inference_ms") or 0.0)
            for s in summaries
        ),
        "weighted_avg_total_frame_ms": weighted_total,
        "total_dropped_frames": total_dropped,
        "total_events": total_events,
        "actual_backends": ",".join(
            str(s.get("backend") or s.get("runtime"))
            for s in summaries
        ),
    }

csv_path = root / "per_camera_results.csv"

with csv_path.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

aggregate_path = root / "aggregate_results.json"
aggregate_path.write_text(
    json.dumps(backend_aggregates, indent=2, ensure_ascii=False),
    encoding="utf-8",
)


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


pt = backend_aggregates["pytorch"]
trt = backend_aggregates["tensorrt"]

fps_change = pct_change(float(pt["aggregate_fps"]), float(trt["aggregate_fps"]))
avg_yolo_change = pct_change(
    float(pt["weighted_avg_yolo_ms"]),
    float(trt["weighted_avg_yolo_ms"]),
)
p95_change = pct_change(
    float(pt["worst_camera_p95_yolo_ms"]),
    float(trt["worst_camera_p95_yolo_ms"]),
)
total_change = pct_change(
    float(pt["weighted_avg_total_frame_ms"]),
    float(trt["weighted_avg_total_frame_ms"]),
)

lines = [
    "# Two-Camera Concurrent RTSP Benchmark",
    "",
    f"- Cameras: `{', '.join(camera_ids)}`",
    f"- Frames per camera/backend: `{rows[0]['frames_processed']}`",
    "",
    "## Aggregate Results",
    "",
    "| Metric | PyTorch | TensorRT | Change |",
    "|---|---:|---:|---:|",
    (
        f"| Aggregate throughput FPS | "
        f"{float(pt['aggregate_fps']):.3f} | "
        f"{float(trt['aggregate_fps']):.3f} | "
        f"{fps_change:+.2f}% |"
    ),
    (
        f"| Weighted YOLO avg ms | "
        f"{float(pt['weighted_avg_yolo_ms']):.3f} | "
        f"{float(trt['weighted_avg_yolo_ms']):.3f} | "
        f"{avg_yolo_change:+.2f}% |"
    ),
    (
        f"| Worst-camera YOLO p95 ms | "
        f"{float(pt['worst_camera_p95_yolo_ms']):.3f} | "
        f"{float(trt['worst_camera_p95_yolo_ms']):.3f} | "
        f"{p95_change:+.2f}% |"
    ),
    (
        f"| Weighted total-frame avg ms | "
        f"{float(pt['weighted_avg_total_frame_ms']):.3f} | "
        f"{float(trt['weighted_avg_total_frame_ms']):.3f} | "
        f"{total_change:+.2f}% |"
    ),
    (
        f"| Dropped frames | "
        f"{int(pt['total_dropped_frames'])} | "
        f"{int(trt['total_dropped_frames'])} | "
        f"{int(trt['total_dropped_frames']) - int(pt['total_dropped_frames']):+d} |"
    ),
    (
        f"| Events generated | "
        f"{int(pt['total_events'])} | "
        f"{int(trt['total_events'])} | "
        f"{int(trt['total_events']) - int(pt['total_events']):+d} |"
    ),
    "",
    "## Per-Camera Results",
    "",
    "| Backend | Camera | FPS | YOLO avg ms | YOLO p95 ms | Total avg ms | Drops |",
    "|---|---|---:|---:|---:|---:|---:|",
]

for row in rows:
    lines.append(
        f"| {row['backend']} | {row['camera_id']} | "
        f"{float(row['effective_fps']):.3f} | "
        f"{float(row['avg_yolo_inference_ms']):.3f} | "
        f"{float(row['p95_yolo_inference_ms']):.3f} | "
        f"{float(row['avg_total_frame_ms']):.3f} | "
        f"{int(row['dropped_frames'])} |"
    )

lines.extend(
    [
        "",
        "## Verification",
        "",
        f"- PyTorch actual backends: `{pt['actual_backends']}`",
        f"- TensorRT actual backends: `{trt['actual_backends']}`",
        "- This is a concurrent two-camera RTSP benchmark.",
        "- Offline single-video throughput and live RTSP throughput are separate metrics.",
        "",
    ]
)

report_path = root / "two_camera_comparison.md"
report_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved: {report_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {aggregate_path}")
PY
}

main() {
  cd "$ROOT"

  preflight
  prepare_streams

  run_backend "pytorch" "$PT"

  # GPU 메모리와 CUDA context가 정리될 시간을 둔다.
  sleep 5

  run_backend "tensorrt" "$ENGINE"

  build_report

  log "Experiment complete: $OUTPUT_DIR"
}

main "$@"
