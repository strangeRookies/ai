# Front Stream / AI Overlay 실행 가이드

작업 경로는 반드시 `/home/welabs/yolo_training/strange_ai_lstm` 기준입니다. 카메라 경로와 ID는 `cam_01`, `cam_02`, `cam_03`, `cam_04` 규칙을 유지합니다.

## 0. GPU PC 접속 및 최신 코드 업데이트 (필수)

ssh welabs@58.127.241.84

GPU PC에 접속하여 변경된 코드를 최신화하고 불필요한 기존 프로세스를 종료합니다.

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements

# 백그라운드 프로세스 강제 종료 (포트 충돌 방지)
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
docker stop mediamtx 2>/dev/null || true
```

## 포트 역할

- `8554`: MediaMTX RTSP
- `8888`: MediaMTX raw HLS
- `8010~8013`: AI overlay MJPEG

프론트 기본값은 AI 박스가 보이는 overlay 모드입니다.

```env
VITE_STREAM_MODE=overlay
VITE_HLS_BASE_URL=http://localhost:8888
VITE_OVERLAY_BASE_URL=http://localhost:8010
```

raw HLS 모드는 `http://localhost:8888/{cameraLoginId}/index.m3u8`를 사용합니다.

overlay MJPEG 모드는 포트를 카메라 번호에 맞춰 사용합니다.

- `cam_01 -> http://localhost:8010`
- `cam_02 -> http://localhost:8011`
- `cam_03 -> http://localhost:8012`
- `cam_04 -> http://localhost:8013`

## 1. GPU PC에서 MediaMTX 실행

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
bash scripts/run_rtsp_server.sh
```

## 2. GPU PC에서 folder-based RTSP publisher 실행

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate
python scripts/start_simulated_rtsp_from_folder.py \
  --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --backend-url http://localhost:8080 \
  --rtsp-host 127.0.0.1 \
  --rtsp-port 8554 \
  --poll-interval 30
```

송출 확인:

```bash
curl -L http://127.0.0.1:8888/cam_01/index.m3u8
```

정상이라면 `#EXTM3U`가 출력됩니다.

## 3. GPU PC에서 AI runner 실행

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate
python scripts/run_registered_cameras.py \
  --backend-base-url "http://127.0.0.1:8080" \
  --rtsp-base-url "rtsp://127.0.0.1:8554" \
  --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --overlay-base-port 8010 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --publisher mqtt \
  --mqtt-host "54.116.37.232" \
  --mqtt-port 1883 \
  --mqtt-topic "safety/events" \
  --skip-simulated-ffmpeg
```

## 4. Windows PC에서 SSH 터널링 실행

GPU PC IP 직접 접근이 timeout이면 Windows 브라우저에서는 GPU PC IP 대신 `localhost`를 사용합니다.

```powershell
ssh -N -L 8888:127.0.0.1:8888 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 welabs@58.127.241.84

```

## 5. 브라우저 확인

- RAW HLS: `http://localhost:8888/cam_01/index.m3u8`
- AI Overlay: `http://localhost:8010`

## 6. 프론트 확인

overlay 모드:

```env
VITE_STREAM_MODE=overlay
VITE_HLS_BASE_URL=http://localhost:8888
VITE_OVERLAY_BASE_URL=http://localhost:8010
```

raw 모드:

```env
VITE_STREAM_MODE=raw
VITE_HLS_BASE_URL=http://localhost:8888
VITE_OVERLAY_BASE_URL=http://localhost:8010
```

검증 기준:

- `VITE_STREAM_MODE=raw`에서 원본 HLS 영상이 표시됩니다.
- `VITE_STREAM_MODE=overlay`에서 AI 박스가 포함된 MJPEG 영상이 표시됩니다.
- 프론트 콘솔에 최종 stream URL이 출력됩니다.
