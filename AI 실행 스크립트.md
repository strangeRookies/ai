# GPU PC 최신 AI 코드 Pull + RTSP/Overlay 재실행 명령어

> 기준 경로: `~/yolo_training/strange_ai_lstm`  
> 브랜치: `codex/ai-worker-flow-improvements`  
> 목적: GPU PC에서 최신 AI 코드를 받고, 기존 RTSP/Overlay 프로세스를 정리한 뒤, RTSP 4채널과 AI Overlay 4채널을 다시 실행한다.

---
ssh welabs@58.127.241.84

## 0. 프로젝트 이동 및 가상환경 활성화

```bash
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate
```

---

## 1. 최신 AI 코드 Pull

```bash
cd ~/yolo_training/strange_ai_lstm

git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements

git --no-pager log -1 --oneline
```

최근 커밋이 예를 들어 아래처럼 보이면 정상이다.

```text
fbfb87f fix: align run_rtsp_demo.py default dataset path with test_non_chromakey.csv
```

---

## 2. 기존 RTSP/Overlay 프로세스 전부 정리

### 2-1. Overlay 서버 종료

```bash
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true

pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
```

### 2-2. ffmpeg RTSP publisher 종료

```bash
pkill -9 ffmpeg 2>/dev/null || true
ps aux | grep ffmpeg | grep -v grep
```

아무것도 안 나오면 정상이다.

### 2-3. MediaMTX 서버도 새로 켤 경우 종료

```bash
docker stop mediamtx 2>/dev/null || true
```

---

## 3. MediaMTX RTSP 서버 실행

아래 명령어는 **별도 터미널에서 실행하고 계속 켜둔다.**

```bash
cd ~/yolo_training/strange_ai_lstm
bash scripts/run_rtsp_server.sh
```

다른 터미널에서 RTSP 포트 확인:

```bash
ss -ltnp | grep 8554
```

---

## 4. RTSP 테스트 영상 수동 송출 (demo_streamer.py)

백엔드의 자동 송출 기능이 제거되었으므로, **테스트용 시뮬레이션 영상이 필요하다면 GPU PC 터미널에서 수동으로 송출**해야 합니다.
(실제 CCTV와 연결되어 있는 운영 환경이라면 이 단계를 건너뜁니다.)

테스트용 mp4 파일들이 모여있는 폴더(예: `video_pool`)를 지정하여 아래 시작 스크립트를 한 번만 실행하면, 폴더 내 영상들이 cam1~cam4에 자동 할당되어 백그라운드로 송출됩니다.

```bash
cd ~/yolo_training/strange_ai_lstm

# 1. 스크립트로 4채널 한 번에 송출하기 (기본값인 video_pool 폴더 사용 시)
bash scripts/start_demo_stream.sh

# (만약 영상 폴더가 다르다면 경로를 입력하세요)
# bash scripts/start_demo_stream.sh /path/to/my_videos

# 종료할 때
bash scripts/stop_demo_stream.sh
```

> **(참고) 스크립트 대신 직접 개별 송출하고 싶을 때**
> 아래처럼 파이썬 명령어를 1채널씩 직접 실행할 수도 있습니다.
> ```bash
> # cam1에 단일 송출 (여러 터미널 띄워서 실행)
> python tools/demo_streamer.py --video sample_videos/fall.mp4 --rtsp-url rtsp://localhost:8554/cam1
> ```

---

## 5. RTSP cam1~cam4 확인

```bash
for i in 1 2 3 4; do
  echo "===== cam$i ====="
  ffprobe -v error \
    -rtsp_transport tcp \
    -read_intervals %+2 \
    -i rtsp://localhost:8554/cam${i} \
    -show_entries stream=codec_name,width,height,avg_frame_rate \
    -of default=noprint_wrappers=1
done
```

정상 예시:

```text
===== cam1 =====
codec_name=h264
width=640
height=360
avg_frame_rate=15/1
```

404가 나오면 해당 cam publisher가 제대로 안 올라온 것이다.

---

## 6. AI Overlay 4채널 실행

> **변경사항**: `--camera-login-id` 인수가 추가되었다.  
> 이 값은 **백엔드 DB의 `cameras.camera_login_id` 컬럼과 반드시 일치**해야  
> 이상행동 이벤트가 DB에 저장되고, 카메라 상태가 올바르게 갱신된다.

```bash
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

mkdir -p runs/overlay_logs

for idx in 1 2 3 4; do
  port=$((8009 + idx))
  nohup python -u scripts/serve_ai_overlay.py \
    --rtsp-url "rtsp://localhost:8554/cam${idx}" \
    --camera-id "cam_0${idx}" \
    --camera-login-id "cam_0${idx}" \
    --detector-mode real \
    --yolo-model yolo26n-pose.pt \
    --device 0 \
    --imgsz 640 \
    --detector-conf 0.10 \
    --track-thresh 0.10 \
    --match-thresh 0.20 \
    --track-buffer 45 \
    --min-box-area 100 \
    --bbox-smoothing-alpha 0.60 \
    --track-max-missing-seconds 3.0 \
    --action-model benchmark/results/lstm_yolo26n_train1000/YOLO26n-pose/best.pt \
    --action-device 0 \
    --action-threshold 0.3 \
    --classifier-input keypoints \
    --port "$port" \
    --publisher mqtt \
    --mqtt-host "54.116.39.252" \
    --mqtt-port 1883 \
    --mqtt-topic "safety/events" \
    --mqtt-client-id "ai-cam${idx}" \
    --print-events > "runs/overlay_logs/camera-${idx}.log" 2>&1 &
done
```

### MQTT 발행 토픽 정리

| 토픽 | 발행 시점 | 설명 |
|---|---|---|
| `safety/events` | 이상행동 감지 시 | `--mqtt-topic` 인수로 설정 |
| `safety/cameras/status` | RTSP 연결 성공/끊김/오류/재연결 시 | **자동 발행** (별도 설정 불필요) |

> `safety/cameras/status`는 `CameraStatusPublisher`가 내부적으로 자동 발행하므로  
> 별도의 인수 설정 없이 RTSP 연결 상태가 바뀔 때마다 백엔드로 전송된다.

### MQTT 페이로드 필드 정리 (백엔드 연동 기준)

**`safety/events` 페이로드**:
```json
{
  "event_type": "Faint",
  "type": "Faint",
  "camera_id": "cam_01",
  "camera_login_id": "cam_01",
  "timestamp": "2026-06-10T06:30:00Z",
  "detected_at": "2026-06-10T06:30:00Z",
  "severity": "HIGH",
  "confidence": 0.87,
  "score": 0.87,
  "bbox": [120, 80, 360, 420],
  "track_id": 2,
  "message_type": "AI_EVENT"
}
```

**`safety/cameras/status` 페이로드**:
```json
{
  "message_type": "CAMERA_STATUS",
  "camera_login_id": "cam_01",
  "status": "DISCONNECTED",
  "previous_status": "CONNECTED",
  "reason": "STREAM_ENDED",
  "edge_device_id": "edge-ai-01",
  "detected_at": "2026-06-10T06:30:00Z"
}
```

---

## 7. Overlay 포트 확인

```bash
ss -ltnp | grep -E "8010|8011|8012|8013"
```

정상 포트 매핑:

```text
camera-1 → http://localhost:8010/stream
camera-2 → http://localhost:8011/stream
camera-3 → http://localhost:8012/stream
camera-4 → http://localhost:8013/stream
```

---

## 8. Stream 응답 확인

```bash
for port in 8010 8011 8012 8013; do
  echo "===== port $port ====="
  curl --max-time 15 -s -o /tmp/stream_${port}.bin \
    -w "%{http_code} %{size_download} bytes\n" \
    http://127.0.0.1:${port}/stream
done

ls -lh /tmp/stream_*.bin
```

정상 예시:

```text
===== port 8010 =====
200 1234567 bytes
```

---

## 9. Summary 확인

```bash
for port in 8010 8011 8012 8013; do
  echo "===== summary $port ====="
  curl --max-time 5 -s http://127.0.0.1:${port}/summary | python -m json.tool
done
```

확인할 주요 항목:

```text
frames_processed
bbox_detections
keypoints_extracted
generated_sequences
lstm_predictions
new_tracks
lost_tracks
id_switch_like_events
track_diagnostics
track_age
missing_frames
detection_conf
```

---

## 10. Windows에서 브라우저로 보기 및 포트 터널링

네트워크 격리로 인해 GPU PC와 로컬 PC가 다이렉트 통신을 할 수 없습니다. 
따라서 로컬 브라우저로 스트림을 모니터링하고, AI 모델이 감지한 이벤트를 로컬 백엔드 DB(`alert_events` 테이블)에 정상 적재하기 위해서는 **로컬 포트포워딩(L)**과 **MQTT 포트(1883)의 역방향 SSH 터널링(R)**을 동시에 실행해 두어야 합니다.

Windows PowerShell에서 아래 SSH 터널링 통합 명령어를 실행하고 계속 유지합니다.

```cmd
ssh -N -L 8010:localhost:8010 -L 8011:localhost:8011 -L 8012:localhost:8012 -L 8013:localhost:8013 -R 1883:localhost:1883 welabs@58.127.241.84
```

> **주의**: `remote port forwarding failed for listen port 1883` 에러가 발생하면 GPU PC에 이미 MQTT(1883)가 켜져 있어 충돌한 것입니다. GPU PC 터미널에서 `sudo fuser -k 1883/tcp` 또는 `sudo systemctl stop mosquitto`로 포트를 비운 뒤 터널링을 다시 연결해 주세요.

- `-L 8010~8013`: GPU PC의 MJPEG 스트림서버 포트를 로컬 브라우저로 포워딩합니다.
- `-R 1883:localhost:1883`: GPU PC에서 발생하는 MQTT 이벤트(`safety/events`, `safety/cameras/status`)를 로컬 PC에 기동 중인 Mosquitto 브로커(1883)로 전달하여 로컬 백엔드 DB에 알람이 저장되도록 합니다.

터널링 완료 후, 로컬 Windows 브라우저에서 아래 주소로 접속해 실시간 오버레이 화면을 확인합니다.

```text
http://localhost:8010/stream
http://localhost:8011/stream
http://localhost:8012/stream
http://localhost:8013/stream
```

---

## 11. AWS DB 터널링 및 백엔드 서버 실행

백엔드 서버를 로컬에서 구동하기 전에, AWS RDS(데이터베이스)에 연결하기 위한 포트 터널링을 먼저 실행해야 합니다.

1. **AWS DB 터널링 실행** (Windows PowerShell)
새로운 터미널 창을 열고 아래 명령어를 실행하여 터널링을 유지합니다.
```cmd
aws ssm start-session --target i-0e43b10f72af9f159 --document-name AWS-StartPortForwardingSessionToRemoteHost --parameters "host=[smart-safety-db.cleq04iqogz6.ap-northeast-2.rds.amazonaws.com],portNumber=[5432],localPortNumber=[15432]"
```

2. **백엔드 서버 구동** (Windows PowerShell)
터널링이 성공적으로 연결된 후, 또 다른 새 터미널 창에서 백엔드 프로젝트 폴더(`strange_back`)로 이동하여 Spring Boot 서버를 실행합니다.
```cmd
cd strange_back
.\gradlew.bat bootRun
```

---
## 12. 로그 확인

```bash
tail -80 runs/overlay_logs/camera-1.log
tail -80 runs/overlay_logs/camera-2.log
tail -80 runs/overlay_logs/camera-3.log
tail -80 runs/overlay_logs/camera-4.log
```

MQTT 이벤트 발행 확인 (로그에서 검색):

```bash
# 이상행동 이벤트 발행 확인
grep "\[ai-overlay-event\]" runs/overlay_logs/camera-1.log | tail -10

# 카메라 상태 이벤트 발행 확인
grep "\[camera-status\]" runs/overlay_logs/camera-1.log | tail -10
```

RTSP publisher 로그:

```bash
tail -80 runs/rtsp_publisher_logs/cam1.log
tail -80 runs/rtsp_publisher_logs/cam2.log
tail -80 runs/rtsp_publisher_logs/cam3.log
tail -80 runs/rtsp_publisher_logs/cam4.log
```

---

## 13. CPU/GPU 사용량 확인

GPU:

```bash
watch -n 1 nvidia-smi
```

CPU:

```bash
htop
```

CPU 상위 프로세스:

```bash
ps aux --sort=-%cpu | head -20
```

---

## 14. 프로세스 종료 명령어 정리 (한 번에 끄기 & 개별 끄기)

어떤 스크립트를 수정하거나 다시 시작할 때는, **반드시 기존에 돌고 있는 프로세스를 먼저 종료해야 자원(GPU, 포트 등) 충돌이 발생하지 않습니다.**

### 🧨 한 번에 전부 초기화 (가장 권장)
아래 명령어를 통째로 복사해서 실행하면, 비디오 송출부터 AI 분석까지 관련된 모든 백그라운드 프로세스가 깨끗하게 지워집니다.

```bash
# 1. AI 오버레이 분석 스크립트 모두 강제 종료 (GPU 메모리 해제 및 MQTT 연결 해제)
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true

# 2. 오버레이 웹서버 포트 찌꺼기 해제 (8010~8013)
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true

# 3. 비디오 송출(RTSP Publisher) 강제 종료 (수동 송출의 경우에만)
pkill -9 ffmpeg 2>/dev/null || true

echo "모든 프로세스 종료 완료!"
```

### ✂️ 개별적으로 종료하기 (부분 재시작 시)

- **AI(오버레이)만 끄고 싶을 때**: (영상 송출은 건드리지 않고, AI 모델만 업데이트/재시작할 때)
  ```bash
  pkill -f "scripts/serve_ai_overlay.py"
  ```
- **영상 송출만 끄고 싶을 때**: (테스트용으로 켜둔 송출 프로세스만 죽이고 싶을 때)
  ```bash
  bash scripts/stop_demo_stream.sh
  ```
- **RTSP(MediaMTX) 서버 자체를 끄고 싶을 때**:
  ```bash
  docker stop mediamtx 2>/dev/null || true
  ```

---

## 14. 문제별 빠른 판단

### `404 Not Found`

```text
MediaMTX는 켜져 있지만 cam1~cam4 publisher가 안 떠 있는 상태.
ffmpeg publisher를 다시 켜야 한다.
```

### `ffmpeg` 상태가 `TL`

```text
프로세스가 stopped 상태.
kill -9 또는 pkill -9 ffmpeg 후 다시 실행한다.
```

### `/stream`이 timeout

```text
overlay warm-up 중이거나 HTTP/MJPEG frame 생성이 늦은 상태.
우선 15초 이상으로 확인하고, 로그를 본다.
```

### 프론트에서 4개 영상이 멈춤

```text
GPU보다 CPU/MJPEG/브라우저 렌더링 병목 가능성이 높다.
RTSP를 640x360, 15fps로 낮추고, 추후 --stream-fps, --output-width, --jpeg-quality 옵션을 추가한다.
```

### 카메라 상태 배지가 프론트에 안 보임

```text
1. --camera-login-id 값이 DB cameras.camera_login_id 와 일치하는지 확인한다.
2. 백엔드 로그에서 "[CameraStatus]" 로그가 찍히는지 확인한다.
3. SSH 터널링의 -R 1883 역방향 터널이 유지되고 있는지 확인한다.
```

### MQTT 이벤트가 백엔드에 안 들어옴

```text
1. GPU PC에서 mosquitto가 기동 중인지 확인: systemctl status mosquitto
2. SSH -R 1883 터널이 연결된 상태인지 확인한다.
3. 로그에서 "[mqtt] publish failed" 메시지가 있으면 MQTT 연결 자체 문제이다.
4. camera_login_id 가 DB에 등록된 값과 다르면 CAMERA_NOT_FOUND 오류가 발생한다.
```

---

## 15. 추천 실행 순서 요약

```bash
# 1. 이동 및 가상환경
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

# 2. 최신 코드
git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements
git --no-pager log -1 --oneline

# 3. 기존 프로세스 정리
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
pkill -9 ffmpeg 2>/dev/null || true

# 4. MediaMTX는 별도 터미널에서 실행
bash scripts/run_rtsp_server.sh

# 5. 테스트용 RTSP 영상 수동 송출
# 5-1. (추천) 스크립트로 4채널 자동 반복 송출
bash scripts/start_demo_stream.sh
# 5-2. (또는) 원하는 영상만 개별 송출 시
# python tools/demo_streamer.py --video sample_videos/fall.mp4 --rtsp-url rtsp://localhost:8554/cam1

# 6. RTSP 확인
for i in 1 2 3 4; do
  echo "===== cam$i ====="
  ffprobe -v error -rtsp_transport tcp -read_intervals %+2 \
    -i rtsp://localhost:8554/cam${i} \
    -show_entries stream=codec_name,width,height,avg_frame_rate \
    -of default=noprint_wrappers=1
done

# 7. Overlay 4개 실행 (--camera-login-id 추가됨)
mkdir -p runs/overlay_logs
for idx in 1 2 3 4; do
  port=$((8009 + idx))
  nohup python -u scripts/serve_ai_overlay.py \
    --rtsp-url "rtsp://localhost:8554/cam${idx}" \
    --camera-id "cam_0${idx}" \
    --camera-login-id "cam_0${idx}" \
    --detector-mode real \
    --yolo-model yolo26n-pose.pt \
    --device 0 \
    --imgsz 640 \
    --detector-conf 0.10 \
    --track-thresh 0.10 \
    --match-thresh 0.20 \
    --track-buffer 45 \
    --min-box-area 100 \
    --bbox-smoothing-alpha 0.60 \
    --track-max-missing-seconds 3.0 \
    --action-model benchmark/results/lstm_yolo26n_train1000/YOLO26n-pose/best.pt \
    --action-device 0 \
    --action-threshold 0.3 \
    --classifier-input keypoints \
    --port "$port" \
    --publisher mqtt \
    --mqtt-host "54.116.39.252" \
    --mqtt-port 1883 \
    --mqtt-topic "safety/events" \
    --mqtt-client-id "ai-cam${idx}" \
    --print-events > "runs/overlay_logs/camera-${idx}.log" 2>&1 &
done

# 8. stream 확인
for port in 8010 8011 8012 8013; do
  echo "===== port $port ====="
  curl --max-time 15 -s -o /tmp/stream_${port}.bin \
    -w "%{http_code} %{size_download} bytes\n" \
    http://127.0.0.1:${port}/stream
done

# 9. MQTT 이벤트 발행 확인
grep "\[ai-overlay-event\]" runs/overlay_logs/camera-1.log | tail -5
grep "\[camera-status\]" runs/overlay_logs/camera-1.log | tail -5
```
