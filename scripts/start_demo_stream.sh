#!/bin/bash
# scripts/start_demo_stream.sh

# 영상들이 있는 디렉토리 (기본값: video_pool)
VIDEO_DIR=${1:-"video_pool"}

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Directory $VIDEO_DIR does not exist."
    echo "Usage: bash scripts/start_demo_stream.sh [path/to/video_dir]"
    exit 1
fi

mkdir -p runs/rtsp_publisher_logs

echo "기존에 실행 중인 시뮬레이션 스트림을 종료합니다..."
pkill -f "tools/demo_streamer.py" 2>/dev/null || true
pkill -9 ffmpeg 2>/dev/null || true
sleep 1

# 폴더 내 mp4 파일 목록 배열로 가져오기
shopt -s nullglob
files=("$VIDEO_DIR"/*.mp4)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "Error: No mp4 files found in $VIDEO_DIR"
    exit 1
fi

echo "'$VIDEO_DIR' 폴더 내 영상을 사용하여 4채널 송출을 시작합니다..."

for i in 1 2 3 4; do
    # 영상이 4개보다 적으면 처음부터 다시 반복 할당
    idx=$(( (i - 1) % ${#files[@]} ))
    video_path="${files[$idx]}"
    
    echo "[cam$i] 할당 영상: $video_path"
    nohup python -u tools/demo_streamer.py --video "$video_path" --rtsp-url "rtsp://localhost:8554/cam$i" > "runs/rtsp_publisher_logs/cam$i.log" 2>&1 &
done

echo "=================================================="
echo "✅ 4채널 수동 영상 송출이 백그라운드에서 시작되었습니다."
echo "로그 확인: tail -f runs/rtsp_publisher_logs/cam*.log"
echo "종료 시: bash scripts/stop_demo_stream.sh"
echo "=================================================="
