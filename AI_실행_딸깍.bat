@echo off
echo ========================================================
echo Starting AI Environment (GPU PC Sync and Background Jobs)
echo ========================================================

echo.
echo [1/3] Connecting to GPU PC to stop old processes and pull latest code...
ssh welabs@58.127.241.84 "cd /home/welabs/yolo_training/strange_ai_lstm && git stash && git fetch origin && git checkout develop && git pull origin develop && pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true && pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true && pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true && pkill -f 'rtsp://127.0.0.1:8554' 2>/dev/null || true && fuser -k 8010/tcp 2>/dev/null || true && fuser -k 8011/tcp 2>/dev/null || true && fuser -k 8012/tcp 2>/dev/null || true && fuser -k 8013/tcp 2>/dev/null || true && docker rm -f mediamtx 2>/dev/null || true"

echo.
echo [2/3] Spawning AI systems (MediaMTX, RTSP Publisher, AI Runner) on GPU PC...
ssh welabs@58.127.241.84 "cd /home/welabs/yolo_training/strange_ai_lstm && ( nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null & ) && source .venv/bin/activate && ( nohup python scripts/start_simulated_rtsp_from_folder.py --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --backend-url http://localhost:8080 --rtsp-host 127.0.0.1 --rtsp-port 8554 --poll-interval 30 --ffmpeg-mode copy > publisher.log 2>&1 </dev/null & ) && ( nohup python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:8080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-base-port 8010 --overlay-report-enabled --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host 15.165.248.37 --mqtt-port 1883 --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & )"

echo.
echo [3/3] Starting SSH Port Forwarding tunnel...
echo Keep this window open to access HLS streams and AI overlays on localhost.
ssh -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 8080:127.0.0.1:8080 welabs@58.127.241.84
