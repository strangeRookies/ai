# Front Stream / AI Overlay 실행 가이드

이 문서는 Docker 변경 이후의 실행 순서를 기준으로 정리한 최신 가이드입니다.

기준 경로:

- Windows/로컬 workspace: `C:\Users\user\Documents\최종 쉴더스`
- GPU PC AI repo: `/home/welabs/yolo_training/strange_ai_lstm`
- 표준 cameraLoginId: `cam_01`, `cam_02`, `cam_03`, `cam_04`

핵심 분리:

- 일반 영상: MediaMTX WebRTC/WHEP `8889`
- HLS fallback: MediaMTX HLS `8888`
- RTSP publish/input: MediaMTX RTSP `8554`
- AI overlay: `serve_ai_overlay.py`가 여는 MJPEG `8010+`
- MQTT: Mosquitto `1883`, topic `safety/events`

> 중요: AI overlay 포트는 카메라 번호로 고정하지 않습니다. 예를 들어 `cam_04`가 `8010`을 사용할 수 있습니다. 프론트는 `8010`, `8011`을 직접 계산하지 않고 backend의 `/api/cameras/{cameraLoginId}/ai-overlay` API에서 `overlayUrl/status`를 조회합니다.

---

## 0. 전체 실행 순서 요약

권장 순서:

1. Windows에서 backend/frontend/DB/MQTT Docker 실행
2. Windows에서 GPU PC SSH 터널 실행
3. GPU PC에서 기존 MediaMTX/AI/RTSP publisher 정리
4. GPU PC에서 MediaMTX 실행
5. GPU PC에서 folder-based RTSP publisher 실행
6. GPU PC에서 AI runner 실행
7. Windows 브라우저에서 frontend 확인
8. GPU PC에서 `pgrep`, `ss`로 worker/port 확인

왜 이 순서인가:

- AI runner는 backend의 `/api/cameras/active`를 조회해야 합니다.
- GPU PC에서 backend를 `127.0.0.1:8080`으로 보려면 Windows SSH 터널의 `-R 8080:127.0.0.1:8080`이 먼저 살아 있어야 합니다.
- MediaMTX `8554/8888/8889`가 떠 있어야 RTSP publisher와 WebRTC/HLS 재생이 정상 동작합니다.

---

## 1. Windows에서 Docker 서비스 실행

Windows PowerShell에서 실행합니다.

```powershell
cd "C:\Users\user\Documents\최종 쉴더스"

# 1. 안전하게 백그라운드로 도커 서비스 빌드 및 실행 (Redis 포함)
docker compose -f strange_infra/docker-compose.yml --profile redis up -d --build
docker compose -f strange_infra/docker-compose.yml ps
```

> [!CAUTION]
> **DB 데이터 영구 삭제 방지 (중요!)**
> * 단순히 서비스를 껐다 켜거나 재부팅하고 싶을 때는 반드시 **`docker compose -f strange_infra/docker-compose.yml --profile redis down`** (옵션 없음)을 사용하세요.
> * **`docker compose down -v`** 명령어의 **`-v` (Volume 삭제) 옵션**은 컨테이너 데이터 저장소(named volume인 `postgres-data`)를 영구적으로 완전히 삭제합니다. 이로 인해 가입한 계정 정보와 Seeding된 데이터가 전부 지워지므로, 초기화 목적이 아닌 경우 **절대 `-v`를 사용하지 마십시오.**
> * **안전한 재시작 방법**:
>   ```powershell
>   # 데이터는 그대로 유지한 채 안전하게 껐다 켜기 (Redis 포함)
>   docker compose -f strange_infra/docker-compose.yml --profile redis down
>   docker compose -f strange_infra/docker-compose.yml --profile redis up -d
>   ```

> [!NOTE]
> **DB 환경 프로필 주의 사항**
> * **Docker Compose 환경**: `strange_infra/docker-compose.yml`을 통해 백엔드를 실행하면, Docker 내부의 독립적인 **로컬 PostgreSQL 컨테이너**(`postgres:5432/strange_safety`)에 데이터를 저장합니다.
> * **로컬 IDE/Gradle 환경**: `strange_back/.env`를 주입받아 호스트 OS에서 백엔드를 직접 실행하면, 터널링(`localhost:15432`)을 통해 **외부 AWS RDS DB**를 바라봅니다.
> * 서로 다른 DB를 보고 있기 때문에, 로컬 Gradle을 쓸 때 가입한 계정은 Docker Compose 환경에서 로그인할 수 없습니다. 본인이 현재 어느 DB 환경을 바라보며 연동 테스트를 하고 있는지 명확히 구분하세요.

이 compose가 올리는 주요 서비스:

- `strange-backend`: `localhost:8080`
- `strange-frontend`: `localhost:3000`
- `strange-postgres`: `localhost:5432`
- `strange-mosquitto`: `localhost:1883`
- `strange-redis`: `localhost:6379` (Redis 추가됨)

확인:

```powershell
curl http://localhost:8080/api/cameras/active
docker ps
```

주의:

- frontend Docker 빌드 기본값은 `VITE_STREAM_MODE=webrtc`입니다.
- `VITE_OVERLAY_BASE_URL` 값이 compose에 남아 있어도 최신 frontend overlay 흐름에서는 직접 포트 계산에 쓰지 않습니다. overlay URL은 backend API에서 조회합니다.

---

## 2. Windows에서 GPU PC SSH 터널 실행

새 PowerShell 또는 CMD 창을 열고 아래의 **한 줄 명령어**를 복사하여 실행하고 창을 계속 켜 둡니다. (백틱 없이 한 줄로 복사하여 붙여넣기에 가장 편리합니다)

```powershell
ssh -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 8080:127.0.0.1:8080 welabs@58.127.241.84
```

> [!TIP]
> **원클릭 자동 실행 방법**
> 프로젝트 루트 디렉터리에 생성되어 있는 [SSH_터널_딸깍.bat](file:///c:/Users/user/Documents/최종%20쉴더스/SSH_터널_딸깍.bat) 파일을 더블 클릭하여 실행하면 터널링 세션을 손쉽게 바로 시작할 수 있습니다.

터널 역할:

- Windows 브라우저 -> GPU PC MediaMTX/HLS/WebRTC/AI overlay 접근
- GPU PC AI runner -> Windows backend `localhost:8080` 접근

포트 의미:

- `-L 8888`: Windows `localhost:8888` -> GPU PC HLS
- `-L 8889`: Windows `localhost:8889` -> GPU PC WebRTC/WHEP
- `-L 8189`: Windows `localhost:8189` -> GPU PC WebRTC ICE
- `-L 8010~8013`: Windows `localhost:8010~8013` -> GPU PC AI overlay
- `-R 8080`: GPU PC `127.0.0.1:8080` -> Windows backend `127.0.0.1:8080`

---

## 3. GPU PC 접속 및 기존 프로세스 정리

GPU PC에 접속합니다.

```bash
ssh welabs@58.127.241.84
cd /home/welabs/yolo_training/strange_ai_lstm
```

최신 코드 반영:

```bash
git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements
```

기존 프로세스 정리:

```bash
pkill -f "scripts/run_registered_cameras.py" 2>/dev/null || true
pkill -f "scripts/start_simulated_rtsp_from_folder.py" 2>/dev/null || true
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
pkill -f "rtsp://127.0.0.1:8554" 2>/dev/null || true

fuser -k 8010/tcp 2>/dev/null || true
fuser -k 8011/tcp 2>/dev/null || true
fuser -k 8012/tcp 2>/dev/null || true
fuser -k 8013/tcp 2>/dev/null || true

docker rm -f mediamtx 2>/dev/null || true
```

주의:

- MediaMTX는 GPU PC에서 Docker container `mediamtx`로 실행됩니다.
- `--network=host`를 쓰므로 `docker ps`의 PORTS 칸에 `8554`, `8888`, `8889`가 안 보여도 host에서는 LISTEN 중일 수 있습니다.

---

## 4. GPU PC에서 MediaMTX 실행

GPU PC 터미널 1:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
bash scripts/run_rtsp_server.sh
```

정상 확인:

```bash
ss -lntup | grep -E "8554|8888|8889|8189"
docker ps | grep mediamtx
```

기대 상태:

- `8554`: LISTEN
- `8888`: LISTEN
- `8889`: LISTEN
- `8189`: LISTEN 또는 UDP open

---

## 5. GPU PC에서 folder-based RTSP publisher 실행

GPU PC 터미널 2:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate

python scripts/start_simulated_rtsp_from_folder.py \
  --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --backend-url http://localhost:8080 \
  --rtsp-host 127.0.0.1 \
  --rtsp-port 8554 \
  --poll-interval 30 \
  --ffmpeg-mode copy
```

송출 확인:

```bash
curl -L http://127.0.0.1:8888/cam_01/index.m3u8
curl -L http://127.0.0.1:8888/cam_04/index.m3u8
```

정상이라면 `#EXTM3U`가 출력됩니다.

WHEP 확인:

```bash
curl -I http://127.0.0.1:8889/cam_01/whep
```

주의:

- `8889`가 LISTEN이어도 해당 cameraLoginId로 RTSP publisher가 공급 중이 아니면 WHEP 404가 날 수 있습니다.
- 포트 LISTEN과 스트림 존재는 별개입니다.

---

## 6. GPU PC에서 AI runner 실행

GPU PC 터미널 3:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate

python scripts/run_registered_cameras.py \
  --backend-base-url http://127.0.0.1:8080 \
  --rtsp-base-url rtsp://127.0.0.1:8554 \
  --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --overlay-base-port 8010 \
  --overlay-report-enabled \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --publisher mqtt \
  --mqtt-host 15.165.248.37 \
  --mqtt-port 1883 \
  --mqtt-topic safety/events \
  --skip-simulated-ffmpeg
```

역할:

- backend `/api/cameras/active` 조회
- 등록된 active AI camera마다 worker 1개 실행
- `serve_ai_overlay.py`를 cameraLoginId 단위로 실행
- overlay port는 `8010+`에서 동적으로 할당
- backend `/api/internal/ai-overlays/report`로 overlay 상태 보고

중요:

- worker는 사용자 단위가 아니라 cameraLoginId 단위입니다.
- 같은 카메라 overlay를 여러 사용자가 열어도 새 worker를 만들지 않고 기존 stream을 공유합니다.
- 한 카메라 화면에 사람이 여러 명 있어도 overlay port는 1개입니다.

---

## 7. Windows 브라우저에서 확인

Frontend:

```text
http://localhost:3000
```

직접 stream smoke:

```text
HLS:
http://localhost:8888/cam_01/index.m3u8

WebRTC/WHEP smoke:
benchmark/webrtc_whep_smoke.html?url=http://localhost:8889/cam_01/whep
```

AI overlay 직접 확인:

```text
http://localhost:8010/stream
```

단, `8010`은 현재 실행 중인 카메라에 따라 달라질 수 있습니다. 실제 frontend는 이 주소를 직접 계산하지 않고 backend overlay API 결과를 사용합니다.

---

## 8. Frontend 환경 변수 기준

기본 권장값:

```env
VITE_BACKEND_BASE_URL=http://localhost:8080
VITE_STREAM_MODE=webrtc
VITE_WEBRTC_BASE_URL=http://localhost:8889
VITE_HLS_BASE_URL=http://localhost:8888
VITE_STREAM_FALLBACK_ENABLED=true
```

Overlay 확인 모드:

```env
VITE_BACKEND_BASE_URL=http://localhost:8080
VITE_STREAM_MODE=overlay
VITE_HLS_BASE_URL=http://localhost:8888
```

주의:

- 최신 overlay 모드에서는 `VITE_OVERLAY_BASE_URL=http://localhost:8010`을 기준으로 포트를 직접 계산하지 않습니다.
- frontend는 backend API에서 `overlayUrl/status`를 조회합니다.
- `VITE_OVERLAY_BASE_URL`이 Docker compose build arg에 남아 있어도 legacy 값으로 보고, 신규 흐름에서는 backend registry가 기준입니다.

---

## 9. 상태 확인 명령

GPU PC:

```bash
pgrep -af "run_registered_cameras.py"
pgrep -af "start_simulated_rtsp_from_folder.py"
pgrep -af "serve_ai_overlay.py"

ss -lntup | grep -E "8010|8011|8012|8013|8889|8888|8554|8189"
docker ps
```

정상 예시:

```text
serve_ai_overlay.py --camera-id cam_04 --camera-login-id cam_04 --port 8010 ...
tcp LISTEN 0 5 0.0.0.0:8010 users:(("python",pid=...,fd=3))
tcp LISTEN ... *:8889
tcp LISTEN ... *:8888
tcp LISTEN ... *:8554
```

중복 worker 확인:

```bash
pgrep -af "serve_ai_overlay.py"
```

같은 `cameraLoginId`가 여러 줄로 나오면 비정상입니다. 정상은 cameraLoginId 하나당 worker 한 줄입니다.

Windows:

```powershell
docker compose -f strange_infra/docker-compose.yml ps
curl http://localhost:8080/api/cameras/active
curl http://localhost:8888/cam_01/index.m3u8
```

---

## 10. MQTT 확인

Windows Docker Mosquitto를 쓸 때:

```powershell
docker exec -it strange-mosquitto mosquitto_sub -h localhost -p 1883 -t safety/events
```

GPU PC AI runner가 외부 MQTT를 쓰는 경우:

```bash
--mqtt-host 15.165.248.37 --mqtt-port 1883 --mqtt-topic safety/events
```

주의:

- 로컬 Mosquitto `localhost:1883`과 외부 MQTT `15.165.248.37:1883`을 혼동하지 마세요.
- backend Docker가 구독하는 MQTT와 AI runner가 publish하는 MQTT가 다르면 이벤트가 backend에 안 들어옵니다.
- 통합 테스트에서는 둘 중 하나로 통일해야 합니다.

---

## 11. 자주 꼬이는 지점

### 1. backend를 켜기 전에 AI runner를 실행함

증상:

- AI runner가 `/api/cameras/active` 조회 실패
- active camera 없음
- overlay worker가 안 뜸

해결:

```bash
curl http://127.0.0.1:8080/api/cameras/active
```

GPU PC에서 이 명령이 통해야 합니다. 안 되면 Windows SSH 터널 `-R 8080:127.0.0.1:8080`을 확인하세요.

### 2. MediaMTX는 떠 있는데 WHEP 404

원인:

- `8889` 포트는 열려 있지만 `cam_01` RTSP publisher가 없음

확인:

```bash
curl -L http://127.0.0.1:8888/cam_01/index.m3u8
pgrep -af "start_simulated_rtsp_from_folder.py"
```

### 3. overlay 8010이 cam_01이라고 착각함

원인:

- overlay port는 cameraLoginId 고정 매핑이 아니라 runtime 할당
- 실제로 `cam_04`가 `8010`을 사용할 수 있음

확인:

```bash
pgrep -af "serve_ai_overlay.py"
```

### 4. Docker `PORTS`에 MediaMTX 포트가 안 보임

원인:

- MediaMTX가 `--network=host`로 실행됨

확인:

```bash
ss -lntup | grep -E "8554|8888|8889|8189"
```

### 5. MQTT 이벤트가 backend에 안 들어옴

원인:

- backend는 Docker Mosquitto를 보고 있고 AI는 외부 MQTT로 publish 중일 수 있음

확인:

```bash
docker logs strange-backend
docker exec -it strange-mosquitto mosquitto_sub -h localhost -p 1883 -t safety/events
```

### 6. Docker로 켰을 때 로그인 실패 / 계정이 없는 경우

원인:

- **환경 간 DB 불일치**: 로컬 IDE/Gradle 환경에서 외부 AWS RDS DB(`localhost:15432`)에 회원가입을 한 계정인데, Docker Compose 환경의 로컬 PostgreSQL 컨테이너(`postgres:5432`)로 로그인하려는 경우입니다.
- **DB 볼륨 초기화**: 이전에 `docker compose down -v`를 실행하여 Docker 로컬 DB 볼륨(`postgres-data`)이 완전 삭제된 경우입니다.

해결:

* **해결 A (로컬 가입)**: Docker Compose가 기동된 상태에서 프론트엔드(`http://localhost:5173`)의 **회원가입(Sign Up)** 페이지로 가 신규 가입을 마칩니다. 6자리 인증번호는 Windows 터미널에서 `docker logs strange-backend --tail 30`을 쳐서 로그에 출력된 `[MOCK-SMS]` 번호를 기입해 통과합니다.
* **해결 B (AWS DB 직접 연동)**: Docker 환경에서도 외부 AWS DB를 바라보아야 할 경우, `strange_infra/docker-compose.yml` 내부의 `backend.environment`에 주입되는 `DB_URL` 값을 로컬 호스트 터널링 주소인 `jdbc:postgresql://host.docker.internal:15432/postgres?sslmode=require`로 주입하고, DB ID/PW도 실제 AWS RDS 자격 증명에 맞추어 구성해야 합니다. (단, 로컬에 15432 터널이 유지되고 있어야 함)

---

## 12. 한 번에 실행하는 빠른 순서

Windows PowerShell 1:

```powershell
cd "C:\Users\user\Documents\최종 쉴더스"
docker compose -f strange_infra/docker-compose.yml up -d --build
```

Windows PowerShell 2:

```powershell
ssh -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 8080:127.0.0.1:8080 welabs@58.127.241.84
```

GPU PC terminal:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
git fetch origin
git checkout codex/ai-worker-flow-improvements
git pull origin codex/ai-worker-flow-improvements

pkill -f "scripts/run_registered_cameras.py" 2>/dev/null || true
pkill -f "scripts/start_simulated_rtsp_from_folder.py" 2>/dev/null || true
pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true
docker rm -f mediamtx 2>/dev/null || true

nohup bash scripts/run_rtsp_server.sh > rtsp_server.log 2>&1 </dev/null &

source .venv/bin/activate
nohup python scripts/start_simulated_rtsp_from_folder.py \
  --video-dir /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --backend-url http://localhost:8080 \
  --rtsp-host 127.0.0.1 \
  --rtsp-port 8554 \
  --poll-interval 30 \
  --ffmpeg-mode copy \
  > publisher.log 2>&1 </dev/null &

nohup python scripts/run_registered_cameras.py \
  --backend-base-url http://127.0.0.1:8080 \
  --rtsp-base-url rtsp://127.0.0.1:8554 \
  --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --overlay-base-port 8010 \
  --overlay-report-enabled \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --publisher mqtt \
  --mqtt-host 15.165.248.37 \
  --mqtt-port 1883 \
  --mqtt-topic safety/events \
  --skip-simulated-ffmpeg \
  > ai_runner.log 2>&1 </dev/null &
```

마지막 확인:

```bash
pgrep -af "serve_ai_overlay.py"
ss -lntup | grep -E "8010|8011|8012|8013|8889|8888|8554|8189"
tail -f ai_runner.log
```
## WebRTC / MQTT metadata 분리 기준

최신 AI worker 기준으로 WebRTC/WHEP는 영상 송출만 담당합니다. AI worker는 bbox를 영상 위에 직접 그린 MJPEG stream에 의존해서 metadata를 전달하지 않고, 별도 MQTT JSON payload를 발행합니다.

- 실시간 overlay 좌표: MQTT `camera` topic
- 확정 이상행동 이벤트: MQTT `event` topic
- legacy 호환: `--mqtt-topic` 또는 `MQTT_TOPIC`을 지정하면 event topic alias로만 사용합니다.
- AI 담당 범위: MQTT publish까지만 수행합니다. DB 저장, Redis write, WebSocket broadcast, Frontend Zustand 업데이트는 AI가 직접 처리하지 않습니다.
- Backend 담당 범위: `camera` topic overlay payload를 subscribe한 뒤 WebSocket으로 Frontend에 전달합니다. `event` topic 확정 이벤트만 DB 저장 대상입니다.
- Frontend 담당 범위: WebRTC video 위에 WebSocket/Zustand로 받은 bbox를 그립니다.

Overlay payload는 감지 결과가 없거나 아직 LSTM sequence 판단이 나오지 않은 frame에서도 빈 `events: []` 배열을 publish합니다. 이렇게 해야 Frontend가 TTL에만 의존하지 않고 즉시 overlay를 지울 수 있습니다.

Runner 실행 시 topic은 아래처럼 명시하는 것을 권장합니다.

```bash
python scripts/run_registered_cameras.py \
  --backend-base-url http://127.0.0.1:8080 \
  --rtsp-base-url rtsp://127.0.0.1:8554 \
  --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos \
  --overlay-base-port 8010 \
  --overlay-report-enabled \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --publisher mqtt \
  --mqtt-host 15.165.248.37 \
  --mqtt-port 1883 \
  --mqtt-camera-topic camera \
  --mqtt-event-topic event \
  --skip-simulated-ffmpeg
```

Overlay payload 예시:

```json
{
  "schemaVersion": "1.0",
  "messageType": "overlay",
  "timestampMs": 1782180000123,
  "streamId": "cam_01",
  "frameWidth": 640,
  "frameHeight": 360,
  "events": [
    {
      "type": "faint",
      "confidence": 0.72,
      "trackingId": 3,
      "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150}
    }
  ]
}
```

Confirmed event payload 예시:

```json
{
  "schemaVersion": "1.0",
  "messageType": "event",
  "eventId": "evt-20260623-cam_01-000001",
  "timestampMs": 1782180000123,
  "streamId": "cam_01",
  "type": "faint",
  "memoText": "쓰러짐 의심!",
  "confidence": 0.92,
  "trackingId": 3,
  "frameWidth": 640,
  "frameHeight": 360,
  "boundingBox": {"x": 120, "y": 80, "width": 200, "height": 150}
}
```

`streamId`는 반드시 backend 등록 카메라의 `cameraLoginId`와 동일해야 합니다. bbox 좌표는 AI 추론 frame 기준 픽셀 좌표이며, Frontend는 `frameWidth/frameHeight`와 실제 video 표시 크기를 기준으로 scale 변환해서 그려야 합니다.
