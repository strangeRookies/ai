# AI Edge Worker & Backend 실행 스크립트

> **💡 요약: 전체 시스템 실행을 위해 총 5개의 터미널 창이 필요합니다.**
> - **[터미널 1] (GPU 서버)**: MediaMTX 실행 및 영상 폴더 기반 자동 RTSP 송출 (`start_simulated_rtsp_from_folder.py`) 실행
> - **[터미널 2] (GPU 서버)**: AI 분석 엔진 (`run_registered_cameras.py` + `--skip-simulated-ffmpeg`) 실행
> - **[터미널 3] (로컬 PC)**: GPU 서버 ↔ 로컬 PC 포트포워딩 (양방향 터널링)
> - **[터미널 4] (로컬 PC)**: AWS RDS DB 터널링
> - **[터미널 5] (로컬 PC)**: Spring Boot 백엔드 서버 실행
> - *(선택)* **[터미널 6] (GPU 서버)**: 실시간 로그 및 리소스 모니터링
> 
> *참고: 본 가이드는 신규 추가된 '영상 폴더 기반 자동 송출 및 AI 분석 흐름'을 기준으로 작성되었습니다.*

---


## 1. GPU 서버 접속 및 전체 프로세스 초기화
```bash
ssh welabs@58.127.241.84

cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

# 최신 코드 업데이트
git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements

# 기존 백그라운드 프로세스 모두 강제 종료
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
pkill -9 ffmpeg 2>/dev/null || true
docker stop mediamtx 2>/dev/null || true
```

## 2. 등록 카메라 기반 실행 (권장)

### [터미널 1] (GPU 서버) - MediaMTX 실행 및 영상 폴더 기반 자동 RTSP 송출
백엔드에 등록된 `SIMULATED_RTSP` 카메라 목록에 맞추어, 지정된 폴더의 비디오 파일들을 1:1 또는 순환 매핑하여 RTSP 스트림을 자동 송출합니다.
```bash
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

# MediaMTX 서버 실행
bash scripts/run_rtsp_server.sh

# 영상 폴더 기반 RTSP 자동 송출 스크립트 실행
# (영상 개수 < 카메라 개수일 경우, 영상 파일이 순환하며 배정됨)
python scripts/start_simulated_rtsp_from_folder.py \
  --video-dir video_pool \
  --backend-url "http://127.0.0.1:8080" \
  --rtsp-host 127.0.0.1 \
  --rtsp-port 8554 \
  --poll-interval 30
```

### [터미널 2] (GPU 서버) - AI 분석 엔진 (Overlay) 실행
백엔드에 등록된 ACTIVE 카메라 목록을 동기화하여 실시간 분석을 수행합니다. **중요: `--skip-simulated-ffmpeg` 옵션을 붙여 자체 중복 송출을 방지합니다.**
```bash
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

python scripts/run_registered_cameras.py \
  --backend-base-url "http://127.0.0.1:8080" \
  --rtsp-base-url "rtsp://@58.127.241.84:8554" \
  --video-pool video_pool \
  --overlay-base-port 8010 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --device 0 \
  --tracking-mode supervision \
  --action-model benchmark/results/lstm_yolo26n_train1000/YOLO26n-pose/best.pt \
  --action-device 0 \
  --action-threshold 0.3 \
  --classifier-input keypoints \
  --publisher mqtt \
  --mqtt-host "3.38.142.174" \
  --mqtt-port 1883 \
  --mqtt-topic "safety/events" \
  --mqtt-client-id-prefix "ai-registered" \
  --refresh-interval-seconds 30.0 \
  --print-events \
  --skip-simulated-ffmpeg
```

## 3. [수동/테스트] 영상 송출 (RTSP & 테스트 비디오)
```bash
# 1. MediaMTX 서버 백그라운드 실행
bash scripts/run_rtsp_server.sh

# 2. 테스트용 시뮬레이션 영상 송출 시작 (cam1 ~ cam4)
bash scripts/start_demo_stream.sh
```

## 4. [수동/테스트] AI 분석 엔진 (Overlay) 실행
```bash
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
    --tracking-mode supervision \
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
    --mqtt-host "3.38.142.174" \
    --mqtt-port 1883 \
    --mqtt-topic "safety/events" \
    --mqtt-client-id "ai-cam${idx}" \
    --print-events > "runs/overlay_logs/camera-${idx}.log" 2>&1 &
done
```

## 5. 백엔드 및 DB 접속 환경 구성 (로컬 PC)

로컬 PC(Windows PowerShell)에서 **새 창**을 열고 각각 실행합니다.

### 5-1. GPU 서버 ↔ 로컬 PC 양방향 포트포워딩
GPU 서버의 AI Worker가 로컬 백엔드(8080)에 접속할 수 있도록 원격 포워딩(`-R`)을 추가하고, 로컬 PC에서 GPU 서버의 영상 스트림(8010~8013)을 볼 수 있도록 로컬 포워딩(`-L`)을 한 번에 결합하여 실행합니다.
```cmd
ssh -N -L 8010:localhost:8010 -L 8011:localhost:8011 -L 8012:localhost:8012 -L 8013:localhost:8013 -R 8080:localhost:8080 welabs@58.127.241.84
```

### 5-2. AWS RDS DB 터널링
새 PowerShell 창에서 실행합니다.
```cmd
aws ssm start-session --target i-0e43b10f72af9f159 --document-name AWS-StartPortForwardingSessionToRemoteHost --parameters "host=[smart-safety-db.cleq04iqogz6.ap-northeast-2.rds.amazonaws.com],portNumber=[5432],localPortNumber=[15432]"
```

### 5-3. Spring Boot 백엔드 서버 실행
DB 터널링 연결이 완료된 후, 새 PowerShell 창에서 실행합니다.
```cmd
cd strange_back
.\gradlew.bat bootRun
```

## 6. 모니터링 및 문제 점검 (GPU 서버 터미널)

```bash
# 영상 포트 응답 확인 (정상 시 200 출력)
for port in 8010 8011 8012 8013; do curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:${port}/stream; done

# 실시간 이벤트 및 에러 로그 확인
tail -f runs/overlay_logs/camera-1.log

# GPU/CPU 사용량 모니터링
watch -n 1 nvidia-smi
htop
```
