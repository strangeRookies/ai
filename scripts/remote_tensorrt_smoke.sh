#!/usr/bin/env bash
# GPU PC TensorRT validation + PyTorch fallback smoke.
# Usage (on GPU PC):
#   cd /home/welabs/yolo_training/strange_ai_lstm
#   source .venv/bin/activate
#   bash scripts/remote_tensorrt_smoke.sh
#
# Optional env:
#   PT_MODEL, ENGINE_MODEL, ACTION_MODEL, DEVICE, ACTION_DEVICE
#   OUTPUT_DIR, MAX_FRAMES, VIDEO, IMGSZ, DETECTOR_CONF
#   SKIP_UNIT=1, SKIP_FALLBACK=1, SKIP_PREFLIGHT=0, SKIP_COMPARE=1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PT_MODEL="${PT_MODEL:-yolo26n-pose.pt}"
ENGINE_MODEL="${ENGINE_MODEL:-yolo26n-pose.engine}"
DEFAULT_ACTION_MODEL="runs/evaluation_manifest_v2_bbox54_balanced/retrained_best.pt"
LEGACY_ACTION_MODEL="benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt"
if [[ -z "${ACTION_MODEL:-}" ]]; then
  if [[ -f "$DEFAULT_ACTION_MODEL" ]]; then
    ACTION_MODEL="$DEFAULT_ACTION_MODEL"
  elif [[ -f "$LEGACY_ACTION_MODEL" ]]; then
    ACTION_MODEL="$LEGACY_ACTION_MODEL"
  else
    ACTION_MODEL="$DEFAULT_ACTION_MODEL"
  fi
fi
DEVICE="${DEVICE:-0}"
ACTION_DEVICE="${ACTION_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/verification_tensorrt_smoke}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VIDEO="${VIDEO:-}"
IMGSZ="${IMGSZ:-640}"
DETECTOR_CONF="${DETECTOR_CONF:-0.10}"
SKIP_UNIT="${SKIP_UNIT:-0}"
SKIP_FALLBACK="${SKIP_FALLBACK:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-1}"

log() {
  echo "[tensorrt-smoke] $*"
}

die() {
  echo "[tensorrt-smoke][error] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="${2:-file}"
  [[ -f "$path" ]] || die "${label} not found: $path"
}

assert_summary_field() {
  local json_path="$1"
  local expect_runtime="$2"
  local require_engine_validation="${3:-0}"
  python - "$json_path" "$expect_runtime" "$require_engine_validation" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expect = sys.argv[2]
require_ev = sys.argv[3] == "1"
data = json.loads(path.read_text(encoding="utf-8"))
runtime = data.get("runtime")
engine_validation = data.get("engine_validation")
model_path = data.get("model_path")
print(
    f"[tensorrt-smoke] summary {path.name}: runtime={runtime} "
    f"model_path={model_path} engine_validation={engine_validation}"
)
if expect == "tensorrt_or_fallback":
    if runtime not in {"tensorrt", "pytorch_fallback"}:
        raise SystemExit(f"expected runtime tensorrt|pytorch_fallback, got {runtime!r}")
elif runtime != expect:
    raise SystemExit(f"expected runtime={expect!r}, got {runtime!r}")
if require_ev:
    if not isinstance(engine_validation, dict):
        raise SystemExit("engine_validation missing or not a dict")
    if "ok" not in engine_validation:
        raise SystemExit("engine_validation.ok missing")
if expect == "pytorch_fallback":
    if not isinstance(engine_validation, dict) or engine_validation.get("ok") is not False:
        raise SystemExit("expected engine_validation.ok == false for pytorch_fallback")
    if not str(model_path or "").lower().endswith(".pt"):
        raise SystemExit(f"expected .pt model_path for fallback, got {model_path!r}")
PY
}

print_summary_brief() {
  local json_path="$1"
  [[ -f "$json_path" ]] || return 0
  python - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
s = json.loads(path.read_text(encoding="utf-8"))
keys = [
    "runtime",
    "model_path",
    "engine_validation",
    "frames_processed",
    "avg_yolo_inference_ms",
    "effective_fps",
    "events_generated",
]
brief = {k: s.get(k) for k in keys if k in s}
print(f"[tensorrt-smoke] {path}: {brief}")
PY
}

run_preflight() {
  local yolo_model="$1"
  local output_path="$2"
  python -u scripts/run_rtsp_inference.py \
    --preflight-only \
    --detector-mode real \
    --yolo-model "$yolo_model" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --detector-conf "$DETECTOR_CONF" \
    --action-model "$ACTION_MODEL" \
    --action-device "$ACTION_DEVICE" \
    --output "$output_path"
}

run_offline() {
  local yolo_model="$1"
  local output_path="$2"
  local camera_id="${3:-cam_offline}"
  python -u scripts/run_rtsp_inference.py \
    --rtsp-url "$VIDEO" \
    --camera-id "$camera_id" \
    --camera-login-id "$camera_id" \
    --max-frames "$MAX_FRAMES" \
    --detector-mode real \
    --yolo-model "$yolo_model" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --detector-conf "$DETECTOR_CONF" \
    --action-model "$ACTION_MODEL" \
    --action-device "$ACTION_DEVICE" \
    --classifier-input keypoints \
    --dry-run \
    --output "$output_path"
}

mkdir -p "$OUTPUT_DIR"
log "root=$ROOT"
log "PT_MODEL=$PT_MODEL"
log "ENGINE_MODEL=$ENGINE_MODEL"
log "ACTION_MODEL=$ACTION_MODEL"
log "DEVICE=$DEVICE OUTPUT_DIR=$OUTPUT_DIR MAX_FRAMES=$MAX_FRAMES"

require_file "$ACTION_MODEL" "ACTION_MODEL"
require_file "$PT_MODEL" "PT_MODEL"

# --- 1) unit tests ---
if [[ "$SKIP_UNIT" != "1" ]]; then
  log "running unit tests: tests/test_tensorrt_runtime.py"
  python -m pytest tests/test_tensorrt_runtime.py -q
  log "unit tests OK"
else
  log "skipping unit tests (SKIP_UNIT=1)"
fi

# --- 2) preflight pytorch ---
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  PT_PREFLIGHT="$OUTPUT_DIR/preflight_pytorch.json"
  log "preflight pytorch -> $PT_PREFLIGHT"
  run_preflight "$PT_MODEL" "$PT_PREFLIGHT"
  assert_summary_field "$PT_PREFLIGHT" "pytorch" 0
else
  log "skipping preflight (SKIP_PREFLIGHT=1)"
fi

# --- 3) preflight tensorrt (optional if engine missing) ---
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  if [[ -f "$ENGINE_MODEL" ]]; then
    TRT_PREFLIGHT="$OUTPUT_DIR/preflight_tensorrt.json"
    log "preflight tensorrt engine -> $TRT_PREFLIGHT"
    run_preflight "$ENGINE_MODEL" "$TRT_PREFLIGHT"
    assert_summary_field "$TRT_PREFLIGHT" "tensorrt_or_fallback" 1
    runtime_val="$(python -c "import json; print(json.load(open(r'''$TRT_PREFLIGHT''', encoding='utf-8')).get('runtime'))")"
    if [[ "$runtime_val" == "tensorrt" ]]; then
      log "TensorRT path selected (runtime=tensorrt)"
    else
      log "WARNING: engine requested but runtime=$runtime_val (fallback allowed on this host)"
    fi
  else
    log "WARNING: ENGINE_MODEL not found, skipping TensorRT preflight: $ENGINE_MODEL"
  fi
fi

# --- 4) intentional broken-engine fallback ---
if [[ "$SKIP_FALLBACK" != "1" ]]; then
  TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorrt_smoke.XXXXXX")"
  cleanup() { rm -rf "$TMP_DIR"; }
  trap cleanup EXIT

  BROKEN_ENGINE="$TMP_DIR/yolo26n-pose.engine"
  FALLBACK_PT="$TMP_DIR/yolo26n-pose.pt"
  : >"$BROKEN_ENGINE"
  cp "$PT_MODEL" "$FALLBACK_PT"
  FALLBACK_OUT="$OUTPUT_DIR/preflight_broken_engine_fallback.json"
  log "preflight broken engine (0-byte) -> $FALLBACK_OUT"
  run_preflight "$BROKEN_ENGINE" "$FALLBACK_OUT"
  assert_summary_field "$FALLBACK_OUT" "pytorch_fallback" 1
  log "broken-engine fallback OK"
else
  log "skipping broken-engine fallback (SKIP_FALLBACK=1)"
fi

# --- 5) optional offline video dry-run ---
if [[ -n "$VIDEO" && "$MAX_FRAMES" -gt 0 ]]; then
  require_file "$VIDEO" "VIDEO"
  log "offline inference frames=$MAX_FRAMES video=$VIDEO"
  run_offline "$PT_MODEL" "$OUTPUT_DIR/offline_pytorch.json" "cam_offline_pt"
  print_summary_brief "$OUTPUT_DIR/offline_pytorch.json"
  if [[ -f "$ENGINE_MODEL" ]]; then
    run_offline "$ENGINE_MODEL" "$OUTPUT_DIR/offline_tensorrt.json" "cam_offline_trt"
    print_summary_brief "$OUTPUT_DIR/offline_tensorrt.json"
  fi
else
  log "skipping offline inference (set VIDEO=... and MAX_FRAMES>0 to enable)"
fi

# --- 6) optional TensorRT candidate compare ---
if [[ "$SKIP_COMPARE" != "1" ]]; then
  if [[ -n "$VIDEO" && -f "$VIDEO" && -f "$ENGINE_MODEL" ]]; then
    log "running compare_tensorrt_candidate.py"
    python scripts/compare_tensorrt_candidate.py \
      --model "$PT_MODEL" \
      --engine "$ENGINE_MODEL" \
      --video "$VIDEO" \
      --max-frames "${COMPARE_MAX_FRAMES:-300}" \
      --imgsz "$IMGSZ" || log "WARNING: compare_tensorrt_candidate failed (non-fatal)"
  else
    log "skipping compare (need VIDEO + ENGINE_MODEL)"
  fi
else
  log "skipping compare (SKIP_COMPARE=1)"
fi

# --- 7) final brief ---
log "outputs in $OUTPUT_DIR"
for f in "$OUTPUT_DIR"/*.json; do
  [[ -e "$f" ]] || continue
  print_summary_brief "$f"
done

log "SMOKE PASS"
