#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/mingw64/bin:/mingw32/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$(cd "$AI_DIR/../ai_fall_experiments" && pwd 2>/dev/null || true)}"
CSV_PATH="${CSV_PATH:-$AI_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv}"
RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://localhost:8554}"
PLAYLIST_DIR="$AI_DIR/runs/playlists"
LOG_DIR="$AI_DIR/runs/rtsp_publisher_logs"
BAD_KEYWORD_REGEX="croki|크로마키|chroma|chromakey|green_screen|studio|chm|indoor_chromakey|inside_croki"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PKILL_BIN="${PKILL_BIN:-pkill}"

if [[ ! -f "$CSV_PATH" && -f "$EXPERIMENTS_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv" ]]; then
  CSV_PATH="$EXPERIMENTS_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: non-chromakey CSV not found: $CSV_PATH" >&2
  echo "Generate it first with scripts/audit_chromakey_split.py." >&2
  exit 1
fi

mkdir -p "$PLAYLIST_DIR" "$LOG_DIR"

echo "Using non-chromakey CSV: $CSV_PATH"
echo "Using experiments dataset root: ${EXPERIMENTS_DIR:-<not set>}"
echo "RTSP base URL: $RTSP_BASE_URL"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 or python is required to read $CSV_PATH" >&2
  exit 1
fi

mapfile -t SELECTED_ROWS < <(
  "$PYTHON_BIN" - "$CSV_PATH" "$EXPERIMENTS_DIR" <<'PY'
import csv
import os
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
experiments_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
bad = re.compile(r"(croki|크로마키|chroma|chromakey|green_screen|studio|chm|indoor_chromakey|inside_croki)", re.I)
selected = {"INDOOR": [], "OUTDOOR": []}
seen = {"INDOOR": set(), "OUTDOOR": set()}
stats = {
    "rows": 0,
    "missing_path": 0,
    "missing_file": 0,
    "excluded_keyword": 0,
    "excluded_domain": 0,
    "duplicates": 0,
}


def choose_path(row: dict[str, str]) -> str:
    return (row.get("source_video") or row.get("video_path") or row.get("clip_path") or "").strip()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if experiments_dir:
        joined = experiments_dir / candidate
        if joined.exists():
            return joined
    return (Path.cwd() / candidate).resolve()


with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
    for row in csv.DictReader(fp):
        stats["rows"] += 1
        domain = (row.get("domain") or "").strip()
        if domain == "indoor_background":
            bucket = "INDOOR"
        elif domain == "outdoor":
            bucket = "OUTDOOR"
        else:
            stats["excluded_domain"] += 1
            continue

        raw_path = choose_path(row)
        if not raw_path:
            stats["missing_path"] += 1
            continue
        resolved = resolve(raw_path)
        text = str(resolved)
        if bad.search(text):
            stats["excluded_keyword"] += 1
            continue
        if not resolved.exists():
            stats["missing_file"] += 1
            continue
        key = str(resolved)
        if key in seen[bucket]:
            stats["duplicates"] += 1
            continue
        seen[bucket].add(key)
        selected[bucket].append(key)

for key, value in stats.items():
    print(f"[csv-select] {key}={value}", file=sys.stderr)
for bucket in ("INDOOR", "OUTDOOR"):
    print(f"[csv-select] selected_{bucket.lower()}={len(selected[bucket])}", file=sys.stderr)
    for path in selected[bucket][:3]:
        print(f"{bucket}|{path}")
PY
)

INDOOR_SOURCE_VIDEOS=()
OUTDOOR_SOURCE_VIDEOS=()
for row in "${SELECTED_ROWS[@]}"; do
  category="${row%%|*}"
  path="${row#*|}"
  case "$category" in
    INDOOR) INDOOR_SOURCE_VIDEOS+=("$path") ;;
    OUTDOOR) OUTDOOR_SOURCE_VIDEOS+=("$path") ;;
  esac
done

if [[ ${#INDOOR_SOURCE_VIDEOS[@]} -lt 3 ]]; then
  echo "ERROR: Need 3 indoor_background source_video files, found ${#INDOOR_SOURCE_VIDEOS[@]}." >&2
  exit 1
fi
if [[ ${#OUTDOOR_SOURCE_VIDEOS[@]} -lt 3 ]]; then
  echo "ERROR: Need 3 outdoor source_video files, found ${#OUTDOOR_SOURCE_VIDEOS[@]}." >&2
  exit 1
fi

INDOOR_SOURCE_VIDEOS=("${INDOOR_SOURCE_VIDEOS[@]:0:3}")
OUTDOOR_SOURCE_VIDEOS=("${OUTDOOR_SOURCE_VIDEOS[@]:0:3}")

CAM1_VIDS=("${INDOOR_SOURCE_VIDEOS[@]}")
CAM2_VIDS=("${INDOOR_SOURCE_VIDEOS[@]}")
CAM3_VIDS=("${OUTDOOR_SOURCE_VIDEOS[@]}")
CAM4_VIDS=("${OUTDOOR_SOURCE_VIDEOS[@]}")

print_list() {
  local title="$1"
  shift
  echo "$title"
  for path in "$@"; do
    echo "  $path"
  done
}

print_list "selected indoor source videos:" "${INDOOR_SOURCE_VIDEOS[@]}"
print_list "selected outdoor source videos:" "${OUTDOOR_SOURCE_VIDEOS[@]}"
print_list "cam1 playlist files:" "${CAM1_VIDS[@]}"
print_list "cam2 playlist files:" "${CAM2_VIDS[@]}"
print_list "cam3 playlist files:" "${CAM3_VIDS[@]}"
print_list "cam4 playlist files:" "${CAM4_VIDS[@]}"

write_playlist() {
  local playlist_file="$1"
  shift
  rm -f "$playlist_file"
  for path in "$@"; do
    printf "file '%s'\n" "$path" >> "$playlist_file"
  done
  echo "Created playlist: $playlist_file"
}

write_playlist "$PLAYLIST_DIR/cam1.txt" "${CAM1_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam2.txt" "${CAM2_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam3.txt" "${CAM3_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam4.txt" "${CAM4_VIDS[@]}"

validate_playlist_keywords() {
  if grep -RniE "$BAD_KEYWORD_REGEX" "$PLAYLIST_DIR"; then
    echo "ERROR: Chromakey-like keyword detected in generated playlists." >&2
    exit 1
  fi
}

validate_playlist_membership() {
  local playlist="$1"
  local expected="$2"
  local line path
  while IFS= read -r line; do
    path="${line#file \'}"
    path="${path%\'}"
    case "$expected" in
      indoor_background)
        printf '%s\n' "${INDOOR_SOURCE_VIDEOS[@]}" | grep -Fx -- "$path" >/dev/null || {
          echo "ERROR: $playlist contains non-indoor source_video: $path" >&2
          exit 1
        }
        ;;
      outdoor)
        printf '%s\n' "${OUTDOOR_SOURCE_VIDEOS[@]}" | grep -Fx -- "$path" >/dev/null || {
          echo "ERROR: $playlist contains non-outdoor source_video: $path" >&2
          exit 1
        }
        ;;
    esac
  done < "$playlist"
}

validate_playlist_membership "$PLAYLIST_DIR/cam1.txt" indoor_background
validate_playlist_membership "$PLAYLIST_DIR/cam2.txt" indoor_background
validate_playlist_membership "$PLAYLIST_DIR/cam3.txt" outdoor
validate_playlist_membership "$PLAYLIST_DIR/cam4.txt" outdoor
validate_playlist_keywords
echo "Playlist validation passed."

echo "Verification commands:"
echo "  cat runs/playlists/cam1.txt"
echo "  cat runs/playlists/cam2.txt"
echo "  cat runs/playlists/cam3.txt"
echo "  cat runs/playlists/cam4.txt"
echo "  grep -RniE \"$BAD_KEYWORD_REGEX\" runs/playlists"
echo "  ps aux | grep ffmpeg | grep -v grep"

echo "Killing existing RTSP ffmpeg publishers for cam1~cam4..."
"$PKILL_BIN" -9 -f "ffmpeg.*rtsp://.*cam[1-4]" || true

start_publisher() {
  local cam_name="$1"
  local playlist_path="$2"
  echo "Starting publisher for $cam_name with $playlist_path"
  nohup "$FFMPEG_BIN" -re -f concat -safe 0 -stream_loop -1 -i "$playlist_path" \
    -an -vf "scale=640:-2,format=yuv420p" -r 15 \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 15 -bf 0 \
    -f rtsp "$RTSP_BASE_URL/$cam_name" > "$LOG_DIR/${cam_name}.log" 2>&1 &
  echo "$cam_name started. Log: $LOG_DIR/${cam_name}.log"
}

start_publisher "cam1" "$PLAYLIST_DIR/cam1.txt"
start_publisher "cam2" "$PLAYLIST_DIR/cam2.txt"
start_publisher "cam3" "$PLAYLIST_DIR/cam3.txt"
start_publisher "cam4" "$PLAYLIST_DIR/cam4.txt"

echo "All RTSP camera publishers have been started in background."
