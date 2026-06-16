@echo off
chcp 65001 >nul
echo ========================================================
echo AI 환경 원클릭 실행 (GPU PC 동기화 및 백그라운드 프로세스 시작)
echo ========================================================

echo.
echo [1/3] GPU PC에 접속하여 기존 프로세스 종료 및 최신 코드 업데이트를 진행합니다...
ssh welabs@58.127.241.84 "cd /home/welabs/yolo_training/strange_ai_lstm && git fetch origin && git checkout codex/ai-worker-flow-improvements && git pull origin codex/ai-worker-flow-improvements && fuser -k 8010/tcp 2>/dev/null || true && fuser -k 8011/tcp 2>/dev/null || true && fuser -k 8012/tcp 2>/dev/null || true && fuser -k 8013/tcp 2>/dev/null || true && pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true && docker rm -f mediamtx 2>/dev/null || true"

echo.
echo [2/3] GPU PC에서 AI 시스템(MediaMTX, RTSP Publisher, AI Runner)을 백그라운드로 실행합니다...
ssh welabs@58.127.241.84 "cd /home/welabs/yolo_training/strange_ai_lstm && ( nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null & ) && source .venv/bin/activate && ( nohup python scripts/start_simulated_rtsp_from_folder.py --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --backend-url http://localhost:8080 --rtsp-host 127.0.0.1 --rtsp-port 8554 --poll-interval 30 > publisher.log 2>&1 </dev/null & ) && ( nohup python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:8080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-base-port 8010 --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host 54.116.37.232 --mqtt-port 1883 --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & )"

echo.
echo [3/3] 포트 포워딩(SSH 터널링)을 시작합니다.
echo ※ 주의: 이 창을 닫으면 프론트에서 영상을 볼 수 없습니다. 개발하는 동안 계속 켜두세요!
ssh -N -L 8888:127.0.0.1:8888 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 8080:127.0.0.1:8080 welabs@58.127.241.84
