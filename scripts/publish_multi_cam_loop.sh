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
DEMO_INDOOR_ALLOWLIST="$AI_DIR/runs/demo_indoor_videos.txt"
DEMO_OUTDOOR_ALLOWLIST="$AI_DIR/runs/demo_outdoor_videos.txt"
BAD_KEYWORD_REGEX="croki|크로마키|chroma|chromakey|green_screen|studio|chm"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PKILL_BIN="${PKILL_BIN:-pkill}"

if [[ ! -f "$CSV_PATH" && -f "$EXPERIMENTS_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv" ]]; then
  CSV_PATH="$EXPERIMENTS_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
fi

mkdir -p "$PLAYLIST_DIR" "$LOG_DIR"

echo "Demo indoor allowlist: $DEMO_INDOOR_ALLOWLIST"
echo "Demo outdoor allowlist: $DEMO_OUTDOOR_ALLOWLIST"
echo "Fallback non-chromakey CSV: $CSV_PATH"
echo "Using experiments dataset root: ${EXPERIMENTS_DIR:-<not set>}"
echo "RTSP base URL: $RTSP_BASE_URL"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 or python is required to read demo allowlists or $CSV_PATH" >&2
  exit 1
fi

INDOOR_SOURCE_VIDEOS=()
OUTDOOR_SOURCE_VIDEOS=()

load_allowlist() {
  local allowlist_path="$1"
  local label="$2"
  local -n target_array="$3"

  if [[ ! -f "$allowlist_path" ]]; then
    return 1
  fi

  mapfile -t target_array < <(
    "$PYTHON_BIN" - "$allowlist_path" "$AI_DIR" <<'PY'
import sys
from pathlib import Path

allowlist = Path(sys.argv[1])
ai_dir = Path(sys.argv[2])
for raw_line in allowlist.read_text(encoding="utf-8-sig").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    path = Path(line).expanduser()
    if not path.is_absolute():
        path = ai_dir / path
    print(str(path.resolve()))
PY
  )

  if [[ ${#target_array[@]} -ne 3 ]]; then
    echo "ERROR: $label allowlist must contain exactly 3 mp4 paths, found ${#target_array[@]}: $allowlist_path" >&2
    exit 1
  fi

  local path
  for path in "${target_array[@]}"; do
    if [[ "$path" != *.mp4 ]]; then
      echo "ERROR: $label allowlist contains a non-mp4 path: $path" >&2
      exit 1
    fi
    if [[ ! -f "$path" ]]; then
      echo "ERROR: $label allowlist path does not exist: $path" >&2
      exit 1
    fi
    if grep -Eiq "$BAD_KEYWORD_REGEX" <<<"$path"; then
      echo "ERROR: $label allowlist path contains a chromakey-like keyword: $path" >&2
      exit 1
    fi
  done

  echo "Using $label allowlist: $allowlist_path"
  return 0
}

select_from_csv() {
  local indoor_needed="$1"
  local outdoor_needed="$2"

  if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: non-chromakey CSV not found: $CSV_PATH" >&2
    echo "Generate it first with scripts/audit_chromakey_split.py, or create runs/demo_indoor_videos.txt and runs/demo_outdoor_videos.txt." >&2
    exit 1
  fi

  mapfile -t SELECTED_ROWS < <(
    "$PYTHON_BIN" - "$CSV_PATH" "$EXPERIMENTS_DIR" "$indoor_needed" "$outdoor_needed" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
experiments_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
limits = {"INDOOR": int(sys.argv[3]), "OUTDOOR": int(sys.argv[4])}
bad = re.compile(r"(croki|크로마키|chroma|chromakey|green_screen|studio|chm)", re.I)
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
        if limits[bucket] <= 0 or len(selected[bucket]) >= limits[bucket]:
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
        if all(len(selected[name]) >= limits[name] for name in selected):
            break

for key, value in stats.items():
    print(f"[csv-select] {key}={value}", file=sys.stderr)
for bucket in ("INDOOR", "OUTDOOR"):
    print(f"[csv-select] selected_{bucket.lower()}={len(selected[bucket])}", file=sys.stderr)
    for path in selected[bucket]:
        print(f"{bucket}|{path}")
PY
  )

  local row category path
  for row in "${SELECTED_ROWS[@]}"; do
    category="${row%%|*}"
    path="${row#*|}"
    case "$category" in
      INDOOR) INDOOR_SOURCE_VIDEOS+=("$path") ;;
      OUTDOOR) OUTDOOR_SOURCE_VIDEOS+=("$path") ;;
    esac
  done
}

load_allowlist "$DEMO_INDOOR_ALLOWLIST" "indoor demo" INDOOR_SOURCE_VIDEOS || true
load_allowlist "$DEMO_OUTDOOR_ALLOWLIST" "outdoor demo" OUTDOOR_SOURCE_VIDEOS || true

if [[ ${#INDOOR_SOURCE_VIDEOS[@]} -lt 3 || ${#OUTDOOR_SOURCE_VIDEOS[@]} -lt 3 ]]; then
  echo "One or more demo allowlists are missing; falling back to CSV automatic selection for missing camera groups."
  select_from_csv "$((3 - ${#INDOOR_SOURCE_VIDEOS[@]}))" "$((3 - ${#OUTDOOR_SOURCE_VIDEOS[@]}))"
fi

if [[ ${#INDOOR_SOURCE_VIDEOS[@]} -lt 3 ]]; then
  echo "ERROR: Need 3 indoor demo source videos, found ${#INDOOR_SOURCE_VIDEOS[@]}." >&2
  exit 1
fi
if [[ ${#OUTDOOR_SOURCE_VIDEOS[@]} -lt 3 ]]; then
  echo "ERROR: Need 3 outdoor demo source videos, found ${#OUTDOOR_SOURCE_VIDEOS[@]}." >&2
  exit 1
fi

INDOOR_SOURCE_VIDEOS=("${INDOOR_SOURCE_VIDEOS[@]:0:3}")
OUTDOOR_SOURCE_VIDEOS=("${OUTDOOR_SOURCE_VIDEOS[@]:0:3}")

CAM1_VIDS=()
CAM2_VIDS=()
CAM3_VIDS=()
CAM4_VIDS=()

split_by_index() {
  local source_name="$1"
  local even_target_name="$2"
  local odd_target_name="$3"
  local -n source_array="$source_name"
  local -n even_target="$even_target_name"
  local -n odd_target="$odd_target_name"
  local index

  for index in "${!source_array[@]}"; do
    if (( index % 2 == 0 )); then
      even_target+=("${source_array[$index]}")
    else
      odd_target+=("${source_array[$index]}")
    fi
  done
}

split_by_index INDOOR_SOURCE_VIDEOS CAM1_VIDS CAM2_VIDS
split_by_index OUTDOOR_SOURCE_VIDEOS CAM3_VIDS CAM4_VIDS

if [[ ${#CAM1_VIDS[@]} -eq 0 || ${#CAM2_VIDS[@]} -eq 0 ]]; then
  echo "ERROR: indoor demo candidates must produce non-empty cam1 and cam2 playlists." >&2
  exit 1
fi
if [[ ${#CAM3_VIDS[@]} -eq 0 || ${#CAM4_VIDS[@]} -eq 0 ]]; then
  echo "ERROR: outdoor demo candidates must produce non-empty cam3 and cam4 playlists." >&2
  exit 1
fi

print_list() {
  local title="$1"
  shift
  echo "$title"
  for path in "$@"; do
    echo "  $path"
  done
}

write_playlist() {
  local playlist_file="$1"
  shift
  rm -f "$playlist_file"
  for path in "$@"; do
    printf "file '%s'\n" "$path" >> "$playlist_file"
  done
  echo "Created playlist: $playlist_file"
}

validate_playlist_keywords() {
  local playlist
  for playlist in "$@"; do
    if grep -niE "$BAD_KEYWORD_REGEX" "$playlist"; then
      echo "ERROR: Chromakey-like keyword detected in generated playlist: $playlist" >&2
      exit 1
    fi
  done
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
          echo "ERROR: $playlist contains non-indoor demo source video: $path" >&2
          exit 1
        }
        ;;
      outdoor)
        printf '%s\n' "${OUTDOOR_SOURCE_VIDEOS[@]}" | grep -Fx -- "$path" >/dev/null || {
          echo "ERROR: $playlist contains non-outdoor demo source video: $path" >&2
          exit 1
        }
        ;;
    esac
  done < "$playlist"
}

assert_playlists_differ() {
  local left_label="$1"
  local left_playlist="$2"
  local right_label="$3"
  local right_playlist="$4"

  if cmp -s "$left_playlist" "$right_playlist"; then
    echo "ERROR: $left_label and $right_label playlists are identical; cameras must split candidate videos." >&2
    exit 1
  fi

  echo "Playlist comparison: $left_label and $right_label are different."
}

print_playlist_file() {
  local title="$1"
  local playlist="$2"
  echo "$title: $playlist"
  sed 's/^/  /' "$playlist"
}

print_list "selected indoor demo source videos:" "${INDOOR_SOURCE_VIDEOS[@]}"
print_list "selected outdoor demo source videos:" "${OUTDOOR_SOURCE_VIDEOS[@]}"
print_list "cam1 playlist files:" "${CAM1_VIDS[@]}"
print_list "cam2 playlist files:" "${CAM2_VIDS[@]}"
print_list "cam3 playlist files:" "${CAM3_VIDS[@]}"
print_list "cam4 playlist files:" "${CAM4_VIDS[@]}"

write_playlist "$PLAYLIST_DIR/cam1.txt" "${CAM1_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam2.txt" "${CAM2_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam3.txt" "${CAM3_VIDS[@]}"
write_playlist "$PLAYLIST_DIR/cam4.txt" "${CAM4_VIDS[@]}"

validate_playlist_membership "$PLAYLIST_DIR/cam1.txt" indoor_background
validate_playlist_membership "$PLAYLIST_DIR/cam2.txt" indoor_background
validate_playlist_membership "$PLAYLIST_DIR/cam3.txt" outdoor
validate_playlist_membership "$PLAYLIST_DIR/cam4.txt" outdoor
validate_playlist_keywords \
  "$PLAYLIST_DIR/cam1.txt" \
  "$PLAYLIST_DIR/cam2.txt" \
  "$PLAYLIST_DIR/cam3.txt" \
  "$PLAYLIST_DIR/cam4.txt"
assert_playlists_differ "cam1" "$PLAYLIST_DIR/cam1.txt" "cam2" "$PLAYLIST_DIR/cam2.txt"
assert_playlists_differ "cam3" "$PLAYLIST_DIR/cam3.txt" "cam4" "$PLAYLIST_DIR/cam4.txt"
echo "Playlist validation passed."

echo "Playlist files to publish:"
print_playlist_file "cam1" "$PLAYLIST_DIR/cam1.txt"
print_playlist_file "cam2" "$PLAYLIST_DIR/cam2.txt"
print_playlist_file "cam3" "$PLAYLIST_DIR/cam3.txt"
print_playlist_file "cam4" "$PLAYLIST_DIR/cam4.txt"

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
  local publisher_pid="$!"
  echo "$cam_name started with PID $publisher_pid. Log: $LOG_DIR/${cam_name}.log"
}

start_publisher "cam1" "$PLAYLIST_DIR/cam1.txt"
start_publisher "cam2" "$PLAYLIST_DIR/cam2.txt"
start_publisher "cam3" "$PLAYLIST_DIR/cam3.txt"
start_publisher "cam4" "$PLAYLIST_DIR/cam4.txt"

echo "All RTSP camera publishers have been started in background."
