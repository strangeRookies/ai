#!/usr/bin/env bash
# publish_multi_cam_loop.sh
# 
# 실내, 야외 비디오 중 크로마키(그린스크린) 영상을 제외하고 
# 1~4번 카메라 채널에 각각 무한 반복 재생 플레이리스트로 연동하여 RTSP 송출하는 스크립트입니다.

set -euo pipefail

# 스크립트 실행 위치 기준 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ai_fall_experiments 디렉토리 경로 탐색 (strange_ai 기준 상위 디렉토리에 존재)
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$(cd "$AI_DIR/../ai_fall_experiments" && pwd 2>/dev/null || echo "")}"

if [[ -z "$EXPERIMENTS_DIR" || ! -d "$EXPERIMENTS_DIR" ]]; then
  # 만약 위 경로에 없으면, 한 단계 더 위나 형제 디렉토리 탐색
  EXPERIMENTS_DIR="$(cd "$AI_DIR/.." && pwd)/ai_fall_experiments"
fi

echo "Using experiments dataset from: $EXPERIMENTS_DIR"

RTSP_BASE_URL="${RTSP_BASE_URL:-rtsp://localhost:8554}"
LOG_DIR="$AI_DIR/runs/rtsp_publisher_logs"
mkdir -p "$LOG_DIR"

# 1. 비디오 파일 로드 및 크로마키(green screen) 영상 필터링 & 셔플링
CSV_PATH=""
CANDIDATES=(
  "data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
  "../ai_fall_experiments/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
  "$EXPERIMENTS_DIR/data/splits/final_source_video_split/chromakey_audit/test_non_chromakey.csv"
)

for cand in "${CANDIDATES[@]}"; do
  if [[ -f "$cand" ]]; then
    CSV_PATH="$cand"
    break
  fi
done

INDOOR_VIDEOS=()
OUTDOOR_VIDEOS=()

if [[ -n "$CSV_PATH" ]]; then
  echo "Loading non-chromakey videos from metadata CSV: $CSV_PATH"
  
  # 실내 비디오 로드 (indoor_background)
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      # 상대 경로로 기재된 경우 절대 경로로 조율
      if [[ "$line" != /* && "$line" != ~* ]]; then
        line="$EXPERIMENTS_DIR/$line"
      fi
      INDOOR_VIDEOS+=("$line")
    fi
  done < <(python3 -c "
import csv
vids = set()
with open('$CSV_PATH', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        domain = r.get('domain', '')
        path = r.get('source_video') or r.get('video_path') or r.get('clip_path')
        if domain == 'indoor_background' and path:
            vids.add(path)
for v in sorted(vids):
    print(v)
" 2>/dev/null || python -c "
import csv
vids = set()
with open('$CSV_PATH', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        domain = r.get('domain', '')
        path = r.get('source_video') or r.get('video_path') or r.get('clip_path')
        if domain == 'indoor_background' and path:
            vids.add(path)
for v in sorted(vids):
    print(v)
" | shuf || true)

  # 야외 비디오 로드 (outdoor)
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      if [[ "$line" != /* && "$line" != ~* ]]; then
        line="$EXPERIMENTS_DIR/$line"
      fi
      OUTDOOR_VIDEOS+=("$line")
    fi
  done < <(python3 -c "
import csv
vids = set()
with open('$CSV_PATH', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        domain = r.get('domain', '')
        path = r.get('source_video') or r.get('video_path') or r.get('clip_path')
        if domain == 'outdoor' and path:
            vids.add(path)
for v in sorted(vids):
    print(v)
" 2>/dev/null || python -c "
import csv
vids = set()
with open('$CSV_PATH', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        domain = r.get('domain', '')
        path = r.get('source_video') or r.get('video_path') or r.get('clip_path')
        if domain == 'outdoor' and path:
            vids.add(path)
for v in sorted(vids):
    print(v)
" | shuf || true)

else
  echo "No test_non_chromakey.csv found. Falling back to directory filtering..."
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      INDOOR_VIDEOS+=("$line")
    fi
  done < <(find "$EXPERIMENTS_DIR/data/raw/indoor_background" -name "*.mp4" 2>/dev/null | grep -v -i -E "chroma|green|screen|studio|key|chm" | shuf || true)

  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      OUTDOOR_VIDEOS+=("$line")
    fi
  done < <(find "$EXPERIMENTS_DIR/data/raw/outdoor" -name "*.mp4" 2>/dev/null | grep -v -i -E "chroma|green|screen|studio|key|chm" | shuf || true)
fi

echo "Found ${#INDOOR_VIDEOS[@]} indoor background videos (filtered)."
echo "Found ${#OUTDOOR_VIDEOS[@]} outdoor videos (filtered)."

echo "=== Indoor Video List (First 10) ==="
for i in $(seq 0 $(( ${#INDOOR_VIDEOS[@]} < 10 ? ${#INDOOR_VIDEOS[@]} - 1 : 9 ))); do
  if [[ -n "${INDOOR_VIDEOS[i]:-}" ]]; then
    echo "  - $(basename "${INDOOR_VIDEOS[i]}")"
  fi
done

echo "=== Outdoor Video List (First 10) ==="
for i in $(seq 0 $(( ${#OUTDOOR_VIDEOS[@]} < 10 ? ${#OUTDOOR_VIDEOS[@]} - 1 : 9 ))); do
  if [[ -n "${OUTDOOR_VIDEOS[i]:-}" ]]; then
    echo "  - $(basename "${OUTDOOR_VIDEOS[i]}")"
  fi
done

if [[ ${#INDOOR_VIDEOS[@]} -eq 0 || ${#OUTDOOR_VIDEOS[@]} -eq 0 ]]; then
  echo "Error: No matching videos found. Check paths under $EXPERIMENTS_DIR" >&2
  exit 1
fi

# 2. 플레이리스트 생성 함수
create_playlist() {
  local playlist_file="$1"
  shift
  local vids=("$@")
  
  rm -f "$playlist_file"
  for vid in "${vids[@]}"; do
    # ffmpeg concat을 위한 absolute path 매핑
    local abs_path
    abs_path="$(cd "$(dirname "$vid")" && pwd)/$(basename "$vid")"
    echo "file '$abs_path'" >> "$playlist_file"
  done
  echo "Created playlist: $playlist_file"
}

# cam1, cam2 -> indoor 비디오들을 분할해서 적절히 섞어 배분
# cam3, cam4 -> outdoor 비디오들을 분할해서 적절히 섞어 배분
# 갯수가 적을 수 있으므로 round-robin 배분

CAM1_VIDS=()
CAM2_VIDS=()
for i in "${!INDOOR_VIDEOS[@]}"; do
  if (( i % 2 == 0 )); then
    CAM1_VIDS+=("${INDOOR_VIDEOS[i]}")
  else
    CAM2_VIDS+=("${INDOOR_VIDEOS[i]}")
  fi
done
# 홀수개인 경우 대응
if [[ ${#CAM2_VIDS[@]} -eq 0 ]]; then
  CAM2_VIDS+=("${INDOOR_VIDEOS[0]}")
fi

CAM3_VIDS=()
CAM4_VIDS=()
for i in "${!OUTDOOR_VIDEOS[@]}"; do
  if (( i % 2 == 0 )); then
    CAM3_VIDS+=("${OUTDOOR_VIDEOS[i]}")
  else
    CAM4_VIDS+=("${OUTDOOR_VIDEOS[i]}")
  fi
done
# 홀수개인 경우 대응
if [[ ${#CAM4_VIDS[@]} -eq 0 ]]; then
  CAM4_VIDS+=("${OUTDOOR_VIDEOS[0]}")
fi

# 임시 playlist 파일 디렉토리
PLAYLIST_DIR="$AI_DIR/runs/playlists"
mkdir -p "$PLAYLIST_DIR"

create_playlist "$PLAYLIST_DIR/cam1.txt" "${CAM1_VIDS[@]}"
create_playlist "$PLAYLIST_DIR/cam2.txt" "${CAM2_VIDS[@]}"
create_playlist "$PLAYLIST_DIR/cam3.txt" "${CAM3_VIDS[@]}"
create_playlist "$PLAYLIST_DIR/cam4.txt" "${CAM4_VIDS[@]}"

# 기존 ffmpeg 프로세스 정리
echo "Killing any existing RTSP ffmpeg publishers..."
pkill -9 -f "ffmpeg.*rtsp://.*cam[1-4]" || true
pkill -9 -f "ffmpeg" || true
pkill -9 ffmpeg || true

# 3. ffmpeg concat 무한루프 송출 실행
start_publisher() {
  local cam_name="$1"
  local playlist_path="$2"
  
  echo "Starting publisher for $cam_name..."
  # -safe 0: 절대경로 사용 허용
  # -stream_loop -1: 플레이리스트 무한반복
  # -re: 실시간 속도로 읽기
  # -c:v libx264 -preset ultrafast -tune zerolatency: 저지연 인코딩
  nohup ffmpeg -re -f concat -safe 0 -stream_loop -1 -i "$playlist_path" \
    -an -vf "scale=640:-2,format=yuv420p" -r 15 \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 15 -bf 0 \
    -f rtsp \
    "$RTSP_BASE_URL/$cam_name" > "$LOG_DIR/${cam_name}.log" 2>&1 &
    
  echo "$cam_name started. Log at $LOG_DIR/${cam_name}.log"
}

start_publisher "cam1" "$PLAYLIST_DIR/cam1.txt"
start_publisher "cam2" "$PLAYLIST_DIR/cam2.txt"
start_publisher "cam3" "$PLAYLIST_DIR/cam3.txt"
start_publisher "cam4" "$PLAYLIST_DIR/cam4.txt"

echo "All RTSP camera publishers have been started in background."
echo "Use 'ps aux | grep ffmpeg' to monitor them."
