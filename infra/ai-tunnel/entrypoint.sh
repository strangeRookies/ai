#!/bin/bash
set -e

# 대상 원격 GPU PC 정보
REMOTE_USER="welabs"
REMOTE_HOST="58.127.241.84"
REMOTE_DEST="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
MUX_SOCKET="/tmp/ssh_mux"

echo "========================================================"
echo "Starting AI Environment Tunnel via Docker Container"
echo "========================================================"

# SSH Key 존재 여부 확인 및 권한 강제 조정
if [ -f /root/.ssh/id_rsa ] || [ -f /root/.ssh/id_ed25519 ]; then
  # SSH 키 파일 권한 소유자 읽기 전용으로 조정 (SSH 보안 요구사항)
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/id_* 2>/dev/null || true
  echo " -> SSH Key found. Establishing master connection..."
else
  echo " -> No SSH Key found. Password authentication will be used."
fi

# 1. 마스터 SSH 연결(MUX) 생성 - 패스워드를 1회만 입력하기 위함
echo ""
echo "[1/4] Establishing Master SSH Connection (Enter password if prompted)..."
ssh $SSH_OPTS -M -S "$MUX_SOCKET" -fN $REMOTE_DEST

# MUX 소켓을 경유해 SSH 명령을 내리는 래퍼 함수
run_ssh() {
  ssh $SSH_OPTS -S "$MUX_SOCKET" $REMOTE_DEST "$@"
}

echo ""
echo "[2/4] Connecting to GPU PC to stop old processes and pull latest code..."
run_ssh "bash -c 'cd /home/welabs/yolo_training/strange_ai_lstm && \
  git fetch origin && \
  git checkout codex/ai-worker-flow-improvements && \
  git pull origin codex/ai-worker-flow-improvements && \
  (pkill -f \"[s]cripts/run_registered_cameras.py\" || true) && \
  (pkill -f \"[s]cripts/start_simulated_rtsp_from_folder.py\" || true) && \
  (pkill -f \"[s]cripts/serve_ai_overlay.py\" || true) && \
  (pkill -f \"[r]tsp://127.0.0.1:8554\" || true) && \
  (fuser -k 8010/tcp || true) && \
  (fuser -k 8080/tcp || true) && \
  (fuser -k 8888/tcp || true) && \
  (fuser -k 8889/tcp || true) && \
  (fuser -k 8189/tcp || true) && \
  (docker rm -f mediamtx || true)'"

echo ""
echo "[3/4] Spawning AI systems (MediaMTX, RTSP Publisher, AI Runner) on GPU PC..."
run_ssh "bash -c 'cd /home/welabs/yolo_training/strange_ai_lstm && \
  ( nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null & ) && \
  source .venv/bin/activate && \
  ( nohup python scripts/start_simulated_rtsp_from_folder.py --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --backend-url http://localhost:8080 --rtsp-host 127.0.0.1 --rtsp-port 8554 --poll-interval 30 > publisher.log 2>&1 </dev/null & ) && \
  ( nohup python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:8080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-base-port 8010 --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host 15.165.248.37 --mqtt-port 1883 --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & )'"

# 컨테이너 종료(SIGTERM/SIGINT) 시 원격 프로세스 정리 및 마스터 세션 해제 핸들러
cleanup() {
  echo ""
  echo "========================================================"
  echo "Stopping AI Environment Tunnel (SIGTERM received)"
  echo "========================================================"
  echo "[1/3] Terminating background processes on GPU PC..."
  run_ssh "pkill -f 'scripts/run_registered_cameras.py' 2>/dev/null || true; \
    pkill -f 'scripts/start_simulated_rtsp_from_folder.py' 2>/dev/null || true; \
    pkill -f 'scripts/serve_ai_overlay.py' 2>/dev/null || true; \
    pkill -f 'rtsp://127.0.0.1:8554' 2>/dev/null || true; \
    fuser -k 8010/tcp 2>/dev/null || true; \
    fuser -k 8011/tcp 2>/dev/null || true; \
    fuser -k 8012/tcp 2>/dev/null || true; \
    fuser -k 8013/tcp 2>/dev/null || true; \
    docker rm -f mediamtx 2>/dev/null || true; \
    echo 'Remote processes terminated.'"
  
  echo "[2/3] Closing Master SSH connection..."
  ssh $SSH_OPTS -O exit -S "$MUX_SOCKET" $REMOTE_DEST 2>/dev/null || true
  
  echo "[3/3] Exiting tunnel container."
  exit 0
}

# 시그널 트랩 등록
trap cleanup SIGTERM SIGINT

echo ""
echo "[4/4] Starting SSH Port Forwarding tunnel..."
echo "HLS streams & AI overlays are forwarded to localhost."
echo "Keep this container running. Press Ctrl+C or run 'docker compose down' to stop."

# 마스터 세션을 경유해 포트포워딩 터널을 실행
ssh $SSH_OPTS -S "$MUX_SOCKET" -N \
  -L 0.0.0.0:8888:127.0.0.1:8888 \
  -L 0.0.0.0:8010:127.0.0.1:8010 \
  -L 0.0.0.0:8011:127.0.0.1:8011 \
  -L 0.0.0.0:8012:127.0.0.1:8012 \
  -L 0.0.0.0:8013:127.0.0.1:8013 \
  -L 0.0.0.0:8189:127.0.0.1:8189 \
  -R 8080:host.docker.internal:8080 \
  $REMOTE_DEST &

SSH_PID=$!

# SSH 백그라운드 프로세스가 끝날 때까지 대기 (시그널 수신 대기)
wait $SSH_PID
