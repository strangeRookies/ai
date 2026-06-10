#!/bin/bash
# scripts/stop_demo_stream.sh

echo "=================================================="
echo "🛑 테스트용 시뮬레이션 영상 송출 프로세스를 종료합니다..."
echo "=================================================="

# demo_streamer.py 프로세스 종료
pkill -f "tools/demo_streamer.py" 2>/dev/null || true

# 실행된 ffmpeg 프로세스 강제 종료
pkill -9 ffmpeg 2>/dev/null || true

echo "✅ 모든 영상 송출이 중단되었습니다."
