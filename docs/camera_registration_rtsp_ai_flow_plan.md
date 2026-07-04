# 카메라 등록 기반 RTSP/AI/프론트 연동 개선 계획

## 1. 현재 문제 요약

현재 AI 이벤트는 GPU PC의 AI 스크립트가 MQTT `safety/events`로 발행하고, 백엔드가 이를 DB 저장 및 WebSocket/STOMP 알림으로 전달하는 흐름은 동작한다.

다만 프론트 카메라 영상 카드는 등록된 카메라 정보와 완전히 같은 기준으로 연결되어 있지 않다. 일부 화면/훅은 `cam1`, `cam_01`, `camera-1` 같은 고정 ID 또는 고정 스트림 경로를 사용하고, `LiveCameraGrid`는 브라우저에서 직접 재생할 수 없는 `rtspUrl`을 `<img>`의 `src`로 쓰는 구조가 남아 있다.

따라서 "알림은 들어오는데 영상 카드와 같은 카메라로 매칭되지 않는" 상태가 발생할 수 있다.

## 2. 현재 코드 기준 실제 흐름

### 2.1 백엔드 Camera 구조

| 구분 | 확인한 파일 | 현재 구현 |
|---|---|---|
| Entity | `strange_back/src/main/java/com/strange/safety/camera/entity/Camera.java` | `id(camera_id)`, `cameraLoginId(camera_login_id)`, `rtspUrl`, `status`, `aiEnabled`, `connectionStatus`, `sourceType`, `assignedVideoPath` 보유 |
| 생성 DTO | `strange_back/src/main/java/com/strange/safety/camera/dto/CreateCameraRequest.java` | `cameraLoginId`, `cameraName`, `cameraSerialNumber`, `cameraPassword`, `rtspUrl`, `locationDescription`, `aiEnabled`, `sourceType` 수신 |
| 수정 DTO | `strange_back/src/main/java/com/strange/safety/camera/dto/UpdateCameraRequest.java` | `rtspUrl`, `status`, `aiEnabled`, `sourceType`, `assignedVideoPath` 등 수정 가능 |
| 응답 DTO | `strange_back/src/main/java/com/strange/safety/camera/dto/CameraResponse.java` | `cameraId`, `cameraLoginId`, `rtspUrl`, `status`, `aiEnabled`, `sourceType`, `assignedVideoPath` 반환. `connectionStatus`, `displayStreamUrl`은 현재 응답에 없음 |
| API | `strange_back/src/main/java/com/strange/safety/camera/controller/CameraController.java` | 등록/조회/수정/삭제 및 `GET /api/cameras/active` 존재 |
| Service | `strange_back/src/main/java/com/strange/safety/camera/service/CameraService.java` | `SIMULATED_RTSP` 등록 시 영상 풀에서 mp4 할당 후 RTSP URL 생성 및 시뮬레이션 시작 |

현재 백엔드는 DB numeric `cameraId`와 외부 식별자인 `cameraLoginId`를 모두 가진다. AI/MQTT/RTSP 경로에서는 `cameraLoginId`를 기준 키로 쓰는 편이 현재 코드와 가장 잘 맞는다.

### 2.2 백엔드 MQTT/알림 흐름

| 구분 | 확인한 파일 | 현재 구현 |
|---|---|---|
| MQTT 수신 | `strange_back/src/main/java/com/strange/safety/mqtt/MqttSafetyEventSubscriber.java` | `safety/events`와 `safety/cameras/status` 구독 |
| AI 이벤트 DTO | `strange_back/src/main/java/com/strange/safety/event/dto/SafetyEventDto.java` | `camera_id`, `event_type`, `confidence`, `timestamp`, `bbox`, `track_id` 등 수신. `camera_login_id` 필드는 현재 DTO에 없음 |
| 이벤트 저장 | `strange_back/src/main/java/com/strange/safety/event/service/AlertEventService.java` | `dto.cameraId()`를 `Camera.cameraLoginId`로 조회. 값이 없으면 `"cam_01"` fallback |
| 알림 브로드캐스트 | `strange_back/src/main/java/com/strange/safety/event/service/AlertBroadcastService.java` | `/topic/alerts`로 전송 |
| 카메라 상태 DTO | `strange_back/src/main/java/com/strange/safety/camera/dto/CameraStatusEventDto.java` | `camera_id`, `camera_login_id`, `status`, `reason`, `rtsp_url_masked`, `timestamp` 등 수신 |
| 상태 저장 | `strange_back/src/main/java/com/strange/safety/camera/service/CameraStatusService.java` | `cameraLoginId` 필수. `Camera.cameraLoginId`로 조회 후 `connectionStatus` 업데이트 |
| 상태 브로드캐스트 | `strange_back/src/main/java/com/strange/safety/camera/service/CameraStatusBroadcastService.java` | `/topic/camera-status`로 전송 |

확인된 문제 지점:

- `SafetyEventDto`는 `camera_login_id`를 받지 않는다.
- `AlertEventService`는 `camera_id` 값을 실제 DB numeric ID가 아니라 `cameraLoginId`로 해석한다.
- `AlertEventService`에 `camera_id`가 없으면 `"cam_01"`로 fallback하는 로직이 있어, 등록 카메라와 다른 기본 카메라로 저장될 수 있다.
- `RtspSimulationService.publishStatus()`는 상태 payload에 `camera_id`만 넣는데, `CameraStatusService`는 `camera_login_id`가 없으면 처리하지 않는다. 이 상태 이벤트는 현재 구조에서 저장되지 않을 가능성이 높다.

### 2.3 프론트 카메라/알림 흐름

| 구분 | 확인한 파일 | 현재 구현 |
|---|---|---|
| Camera API 타입 | `strange_front/src/app/api/cameraApi.ts` | `CameraResponse`에 `cameraId`, `cameraLoginId`, `rtspUrl`, `status`, `sourceType`, `assignedVideoPath` 존재. `connectionStatus`, `displayStreamUrl` 없음 |
| 등록 카메라 로딩 | `strange_front/src/features/dashboard/pages/UserDashboard.tsx` | 시설별 카메라를 `fetchCamerasByFacility()`로 조회 |
| LiveCamera 변환 | `strange_front/src/features/dashboard/pages/UserDashboard.tsx` | `id = cameraId.toString()`, `streamUrl = rtspUrl || ''`, `connectionStatus = status === ACTIVE ? online : offline` |
| 영상 표시 | `strange_front/src/features/dashboard/components/LiveCameraGrid.tsx` | `<img src={camera.streamUrl}>`로 표시 |
| 정적 카메라 데이터 | `strange_front/src/features/dashboard/data/cameras.ts` | `LIVE_CAMERAS`, `STREAM_BASE_URL`, `/stream/{cameraId}` 방식의 정적/기본 경로 존재 |
| 정적 카메라 훅 | `strange_front/src/features/dashboard/hooks/useLiveCameras.ts` | 정적 `LIVE_CAMERAS`를 기준으로 `/cameras` 상태를 병합 |
| 카메라 상태 WS | `strange_front/src/features/dashboard/hooks/useCameraStatusWebSocket.ts` | `/topic/camera-status` 구독. `cameraLoginId ?? cameraId`를 key로 map 저장 |
| AI 이벤트 WS | `strange_front/src/features/dashboard/hooks/useAiEvents.ts` | `/topic/alerts` 구독. 현재 이벤트의 `camera_id` 중심 |
| 알림-카메라 매칭 | `strange_front/src/shared/utils/aiAlerts.ts` | 이벤트 `camera_id`를 `camera.id`, `camera.name`, `camera.location`과 비교. `cameraLoginId` 직접 매칭은 없음 |

확인된 문제 지점:

- `LiveCamera.id`가 DB numeric `cameraId` 문자열인데, 상태 WebSocket map은 주로 `cameraLoginId`를 key로 쓴다.
- `LiveCameraGrid`는 `<img>`를 사용하므로 RTSP URL을 직접 재생할 수 없다.
- 프론트가 등록 카메라 목록을 가져오는 흐름은 있으나, 표시 URL은 `rtspUrl` 그대로라 브라우저 재생용 URL이 아니다.
- `LIVE_CAMERAS`/`useLiveCameras`/`IntegratedDashboard.tsx`에는 `camera-1`, `CCTV-01`, `/stream/{id}` 계열의 정적 흐름이 남아 있다.

### 2.4 AI RTSP/MQTT 흐름

| 구분 | 확인한 파일 | 현재 구현 |
|---|---|---|
| 실행 문서 | `AI 실행 스크립트.md` | `rtsp://localhost:8554/cam1`~`cam4`, `cam_01`~`cam_04` 중심의 수동 실행 예시 |
| 데모 송출 | `scripts/start_demo_stream.sh` | `video_pool` mp4를 `rtsp://localhost:8554/cam{i}`로 반복 송출 |
| 단일 영상 송출 | `scripts/publish_sample_video.sh` | mp4를 지정 RTSP path로 송출 |
| MediaMTX 설정 | `stream/mediamtx.yml` | `cam1`~`cam4` path 설정 확인 |
| RTSP 분석 | `scripts/run_rtsp_inference.py` | 기본값 `rtsp://localhost:8554/cam1`, `camera-id cam_01` |
| 오버레이/AI 서버 | `scripts/serve_ai_overlay.py` | `--rtsp-url`, `--camera-id`, `--camera-login-id`를 CLI로 받음 |
| 이벤트 payload | `ai/inference/rtsp_runtime.py` | `camera_id`, `camera_login_id`, `event_type`, `confidence`, `timestamp`, `bbox`, `track_id` 포함 |
| 카메라 상태 발행 | `ai/publishers/camera_status_publisher.py` | `safety/cameras/status` 발행. payload builder는 `camera_id` optional 지원, publisher 클래스는 현재 `camera_login_id` 중심 |

확인된 문제 지점:

- AI 실행은 등록 카메라 API를 조회하지 않고, 사람이 `cam1`/`cam_01` 값을 맞춰 실행해야 한다.
- `start_demo_stream.sh`도 등록 카메라의 `cameraLoginId`가 아니라 `cam1`~`cam4` 고정 path를 사용한다.
- 이벤트 payload에는 `camera_login_id`가 포함되지만, 백엔드 `SafetyEventDto`가 이 필드를 직접 받지 않는다.
- 현재 호환을 위해서는 AI 이벤트의 `camera_id`를 DB numeric id가 아니라 `cameraLoginId` 값으로 보내야 `AlertEventService`가 카메라를 찾을 수 있다.

## 3. 목표 흐름

```text
프론트 카메라 등록
→ Backend Camera DB 저장
→ GPU PC가 Backend의 active AI 카메라 목록 조회
→ camera_login_id 기준 RTSP path 또는 실제 rtsp_url 결정
→ GPU PC가 mp4 또는 CCTV 영상을 FFmpeg/MediaMTX로 RTSP 송출
→ AI 스크립트가 해당 rtsp_url 분석
→ 이상행동 감지 시 MQTT safety/events 발행
→ RTSP 연결 상태 변경 시 MQTT safety/cameras/status 발행
→ Backend DB 저장 및 WebSocket/STOMP 전달
→ Frontend LiveCameraGrid와 AiAlertCard가 같은 camera_login_id 기준으로 매칭
```

## 4. 매칭 기준

권장 기준은 다음과 같다.

| 용도 | 권장 필드 | 이유 |
|---|---|---|
| DB 내부 PK | `cameraId` / `camera_id` numeric | 백엔드 Entity의 primary key |
| RTSP path | `cameraLoginId` / `camera_login_id` | 사람이 지정하는 외부 식별자이며 path로 쓰기 쉬움 |
| MQTT AI 이벤트 매칭 | `camera_login_id` 우선, 기존 호환용 `camera_id`도 같은 값 전송 | 현재 백엔드는 `camera_id`를 `Camera.cameraLoginId`로 조회 |
| MQTT 상태 이벤트 매칭 | `camera_login_id` 필수 | `CameraStatusService`가 `cameraLoginId`로 조회 |
| 프론트 카드/알림 매칭 | `cameraLoginId` | 상태 WS와 AI 이벤트를 같은 key로 맞추기 쉬움 |

최소 수정 단계에서는 AI가 `camera_id=<camera_login_id>`와 `camera_login_id=<camera_login_id>`를 모두 보내도록 유지하는 것이 안전하다. 이후 백엔드가 `camera_login_id`를 정식 필드로 처리하도록 바꾼 뒤, numeric `camera_id`와 외부 `camera_login_id`를 분리할 수 있다.

## 5. RTSP 송출 구조

### 5.1 실제 CCTV

`Camera.rtspUrl`에 실제 CCTV RTSP 주소를 저장하고, GPU PC AI runner가 `GET /api/cameras/active`로 가져온 뒤 해당 URL을 분석한다.

```text
camera_login_id = lobby_01
rtsp_url        = rtsp://user:password@cctv-host:554/stream1
AI input        = rtsp_url
MQTT key        = lobby_01
```

### 5.2 SIMULATED_RTSP

실제 CCTV가 없을 때는 등록 카메라의 `assignedVideoPath` 또는 video_pool mp4를 GPU PC에서 MediaMTX로 반복 송출한다.

권장 path 예시:

```text
rtsp://GPU_PC_IP:8554/{camera_login_id}
```

현재 백엔드 `RtspSimulationService.generateRtspUrl()`은 다음 규칙을 쓴다.

```text
rtsp://localhost:8554/cam-{cameraLoginId}
```

최종 결정은 B안, 즉 `cameraLoginId`를 path에 그대로 쓰는 방식이다.

| 선택지 | 설명 |
|---|---|
| A안 | 현재 백엔드 규칙 유지: `rtsp://GPU_PC_IP:8554/cam-{cameraLoginId}` |
| B안 - 선택 | 더 단순한 규칙으로 변경: `rtsp://GPU_PC_IP:8554/{cameraLoginId}` |

따라서 백엔드 `RtspSimulationService.generateRtspUrl()`도 추후 `cam-` prefix를 제거해 `rtsp://GPU_PC_IP:8554/{cameraLoginId}` 형태로 맞추는 것이 목표다. 기존 `cam-` prefix가 붙은 데이터나 스크립트가 있으면 마이그레이션 또는 하위 호환 fallback을 별도로 처리한다.

주의: 현재 `RtspSimulationService`는 백엔드 서버 프로세스에서 `ffmpeg`를 실행한다. 백엔드가 GPU PC가 아닌 다른 서버에서 실행된다면, GPU PC의 MediaMTX로 송출한다는 목표와 맞지 않을 수 있다. 이 경우 백엔드는 카메라 등록/DB 저장만 담당하고, GPU PC runner가 등록 카메라 목록을 조회해 FFmpeg 송출을 담당하도록 분리하는 편이 맞다.

## 6. display_stream_url / 영상 표시 URL 방향

브라우저는 RTSP를 직접 재생하지 못하므로 프론트 `LiveCameraGrid`에 들어갈 URL은 RTSP가 아니라 MJPEG/HLS/WebRTC 같은 브라우저 재생용 URL이어야 한다.

현재 `LiveCameraGrid.tsx`는 `<img src={camera.streamUrl}>` 구조이므로 가장 작은 변경은 MJPEG URL을 넣는 방식이다.

권장 방식:

| 필드 | 예시 | 역할 |
|---|---|---|
| `rtspUrl` | `rtsp://GPU_PC_IP:8554/lobby_01` | AI 분석/송출용 |
| `displayStreamUrl` | `http://GPU_PC_IP:8000/stream/lobby_01` 또는 `http://GPU_PC_IP:8010/stream` | 브라우저 표시용 |

`displayStreamUrl`은 다음 두 방식 중 하나로 붙일 수 있다.

| 선택지 | 장점 | 수정 위치 |
|---|---|---|
| 백엔드 `CameraResponse`에 추가 | 프론트가 받은 값을 그대로 사용 가능 | `CameraResponse`, `CameraService`, 설정값 |
| 프론트에서 `VITE_STREAM_BASE_URL + cameraLoginId`로 계산 | 백엔드 변경 작음 | `cameraApi.ts`, `UserDashboard.tsx` |

운영 구조를 명확히 하려면 백엔드 응답에 `displayStreamUrl`을 포함하는 방향이 좋다. 단, 1단계 최소 구현에서는 프론트에서 `cameraLoginId` 기반으로 계산해도 된다.

## 7. MQTT topic/payload 정리

### 7.1 AI 이벤트: `safety/events`

현재 AI 이벤트 payload는 `ai/inference/rtsp_runtime.py`에서 생성한다. 앞으로 최소 필드는 다음처럼 맞춘다.

```json
{
  "message_type": "AI_EVENT",
  "event_type": "faint",
  "type": "faint",
  "camera_id": "lobby_01",
  "camera_login_id": "lobby_01",
  "confidence": 0.87,
  "timestamp": "2026-06-11T12:00:00Z",
  "detected_at": "2026-06-11T12:00:00Z",
  "bbox": [10, 20, 100, 200],
  "track_id": 3
}
```

현재 백엔드 호환 때문에 `camera_id`에는 numeric DB id가 아니라 `cameraLoginId` 값을 넣어야 한다. 이후 `SafetyEventDto`와 `AlertEventService`를 수정하면 `camera_login_id`를 우선 사용하고 `camera_id`는 numeric id로 분리할 수 있다.

### 7.2 카메라 상태: `safety/cameras/status`

`CameraStatusService`는 `camera_login_id`를 필수로 사용한다.

```json
{
  "message_type": "CAMERA_STATUS",
  "camera_login_id": "lobby_01",
  "status": "DISCONNECTED",
  "previous_status": "CONNECTED",
  "reason": "rtsp_read_failed",
  "edge_device_id": "gpu-pc-01",
  "rtsp_url_masked": "rtsp://GPU_PC_IP:8554/lobby_01",
  "timestamp": "2026-06-11T12:05:00Z"
}
```

현재 `RtspSimulationService.publishStatus()`는 `camera_id`에 `cameraLoginId`를 넣고 있어, `camera_login_id`도 함께 보내도록 수정이 필요하다.

## 8. 백엔드 수정 대상

| 우선순위 | 파일 | 수정 방향 |
|---|---|---|
| 1 | `strange_back/src/main/java/com/strange/safety/event/dto/SafetyEventDto.java` | `camera_login_id` / `cameraLoginId` 필드 추가 |
| 1 | `strange_back/src/main/java/com/strange/safety/event/service/AlertEventService.java` | `camera_login_id` 우선으로 `Camera.cameraLoginId` 조회, 기존 `camera_id` fallback 유지, `"cam_01"` fallback 최소화 또는 경고 강화 |
| 1 | `strange_back/src/main/java/com/strange/safety/camera/dto/CameraResponse.java` | `connectionStatus`, `lastConnectionReportAt`, 필요 시 `displayStreamUrl` 추가 |
| 1 | `strange_back/src/main/java/com/strange/safety/camera/service/CameraService.java` | 응답 변환 시 신규 필드 포함, `displayStreamUrl` 계산 정책 반영 |
| 2 | `strange_back/src/main/java/com/strange/safety/camera/service/RtspSimulationService.java` | status payload에 `camera_login_id` 추가. GPU PC 송출 구조와 역할 분리 검토 |
| 2 | `strange_back/src/main/java/com/strange/safety/camera/controller/CameraController.java` | `GET /api/cameras/active`가 GPU PC runner에서 사용 가능한 인증/응답 구조인지 확인 |
| 3 | 설정 파일 | `stream.display-base-url`, `simulation.rtsp.base-url` 등 환경별 URL 설정 정리 |

## 9. AI 스크립트 수정 대상

| 우선순위 | 파일 | 수정 방향 |
|---|---|---|
| 1 | `scripts/run_registered_cameras.py` | 백엔드 `GET /api/cameras/active` 조회 후 카메라별 RTSP 송출/AI 분석 프로세스 실행 |
| 1 | `scripts/serve_ai_overlay.py` | 기존 `--camera-id`, `--camera-login-id`, `--rtsp-url`를 등록 카메라 runner가 넘겨주도록 사용. `camera_id`는 당분간 `cameraLoginId`와 같은 값 |
| 1 | `ai/inference/rtsp_runtime.py` | 이미 `camera_id`, `camera_login_id`, `event_type`, `confidence`, `timestamp` 포함. 백엔드 수정 후에도 유지 |
| 2 | `ai/publishers/camera_status_publisher.py` | publisher 클래스에서 optional `camera_id`도 받을 수 있게 정리. status payload에는 `camera_login_id` 필수 유지 |
| 2 | `scripts/start_demo_stream.sh` | `cam1`~`cam4` 고정 대신 `cameraLoginId` path와 video_pool 할당을 받을 수 있게 개선 |
| 2 | `scripts/publish_sample_video.sh` | 등록 카메라의 `cameraLoginId`를 RTSP path로 사용하도록 실행 예시 정리 |
| 3 | `stream/mediamtx.yml` | `cam1`~`cam4` 고정 path 대신 등록 카메라 path를 수용할 수 있는지 확인/정리 |

GPU PC 기준 실행 예시:

```bash
cd ~/yolo_training/strange_ai_lstm

python scripts/run_registered_cameras.py \
  --backend-base-url http://BACKEND_HOST:18080 \
  --mqtt-host EMQX_HOST \
  --mqtt-port 1883 \
  --rtsp-base-url rtsp://GPU_PC_IP:8554 \
  --video-pool video_pool \
  --overlay-base-port 8010 \
  --tracking-mode supervision
```

위 runner는 등록 카메라별로 `serve_ai_overlay.py`를 실행한다. `REAL_RTSP`는 백엔드 `rtspUrl`을 그대로 분석하고, `SIMULATED_RTSP`는 `assignedVideoPath` 또는 `video_pool` mp4를 `rtsp://GPU_PC_IP:8554/{cameraLoginId}`로 송출한 뒤 분석한다. 기존 단일 카메라 실행은 다음 구조를 유지할 수 있다.

```bash
cd ~/yolo_training/strange_ai_lstm

python scripts/serve_ai_overlay.py \
  --rtsp-url rtsp://GPU_PC_IP:8554/lobby_01 \
  --camera-id lobby_01 \
  --camera-login-id lobby_01 \
  --mqtt-host EMQX_HOST \
  --mqtt-topic safety/events \
  --tracking-mode supervision
```

## 10. 프론트 수정 대상

| 우선순위 | 파일 | 수정 방향 |
|---|---|---|
| 1 | `strange_front/src/app/api/cameraApi.ts` | `CameraResponse`에 `connectionStatus`, `lastConnectionReportAt`, `displayStreamUrl` 추가 |
| 1 | `strange_front/src/features/dashboard/pages/UserDashboard.tsx` | `LiveCamera` 생성 시 `cameraLoginId`를 포함하고, `streamUrl`은 `displayStreamUrl` 또는 `VITE_STREAM_BASE_URL/{cameraLoginId}` 사용 |
| 1 | `strange_front/src/features/dashboard/components/LiveCameraGrid.tsx` | 상태 lookup을 `cameraLoginId` 기준으로 우선 처리 |
| 1 | `strange_front/src/features/dashboard/hooks/useCameraStatusWebSocket.ts` | 이미 `cameraLoginId ?? cameraId` key 사용. `LiveCamera` key와 맞추기 |
| 1 | `strange_front/src/features/dashboard/hooks/useAiEvents.ts` | `camera_login_id`도 파싱하도록 타입/정규화 추가 |
| 1 | `strange_front/src/shared/utils/aiAlerts.ts` | 알림 매칭 시 `event.camera_login_id`와 `camera.cameraLoginId` 우선 비교 |
| 2 | `strange_front/src/features/dashboard/data/cameras.ts` | 운영 화면에서 정적 `LIVE_CAMERAS` 의존 제거 또는 demo 전용으로 분리 |
| 2 | `strange_front/src/features/dashboard/hooks/useLiveCameras.ts` | 정적 카메라 polling 방식 대신 등록 카메라 API 기반 흐름과 통합 |
| 3 | `strange_front/src/features/dashboard/pages/IntegratedDashboard.tsx` | 데모용 하드코딩 카메라가 운영 화면에 섞이지 않도록 정리 |

## 11. 단계별 구현 순서

1. **식별자 정리**
   - 전체 흐름에서 `cameraLoginId`를 외부 연동 키로 확정한다.
   - AI payload는 `camera_id=<cameraLoginId>`, `camera_login_id=<cameraLoginId>`를 모두 보낸다.

2. **백엔드 수신 안정화**
   - `SafetyEventDto`에 `camera_login_id` 추가.
   - `AlertEventService`가 `camera_login_id`를 우선 사용하도록 수정.
   - `CameraResponse`에 `connectionStatus`와 `displayStreamUrl` 추가 여부 결정.

3. **프론트 매칭 수정**
   - `LiveCamera`에 `cameraLoginId`를 추가한다.
   - 영상 카드, 상태 배지, AI 알림 매칭을 `cameraLoginId` 기준으로 맞춘다.
   - `streamUrl`은 RTSP가 아니라 브라우저 표시용 URL을 사용한다.

4. **GPU PC 등록 카메라 runner 추가**
   - `GET /api/cameras/active`로 카메라 목록을 가져온다.
   - `REAL_RTSP`는 DB의 `rtspUrl` 분석.
   - `SIMULATED_RTSP`는 `assignedVideoPath` 또는 video_pool mp4를 `cameraLoginId` path로 송출 후 분석.

5. **상태 이벤트 연결**
   - RTSP 연결 성공/실패/재연결 상태를 `safety/cameras/status`로 발행한다.
   - 백엔드가 `connectionStatus`를 업데이트하고 프론트 배지가 바뀌는지 확인한다.

6. **정적 cam 경로 제거**
   - 운영 화면에서 `cam1`, `cam_01`, `camera-1` 고정 의존을 제거한다.
   - 데모 스크립트는 등록 카메라 기반 실행 예시로 갱신한다.

## 12. 테스트 시나리오

### 12.1 정상 등록/송출/감지

```bash
cd ~/yolo_training/strange_ai_lstm
```

1. 프론트에서 `cameraLoginId=lobby_01`인 카메라를 등록한다.
2. 백엔드 DB 또는 API 응답에서 `camera_login_id`, `rtsp_url`, `source_type`, `ai_enabled` 저장을 확인한다.
3. GPU PC에서 해당 카메라의 RTSP 송출을 시작한다.
4. AI가 해당 `rtsp_url`을 분석하는지 로그에서 확인한다.
5. AI가 `safety/events`로 `camera_id=lobby_01`, `camera_login_id=lobby_01` 이벤트를 발행하는지 확인한다.
6. 백엔드가 이벤트를 DB에 저장하는지 확인한다.
7. 프론트 `AiAlertCard`에 알림이 뜨는지 확인한다.
8. `LiveCameraGrid`에서 같은 `cameraLoginId=lobby_01` 카드가 표시되고 알림과 매칭되는지 확인한다.

### 12.2 RTSP 중지/상태 변경

1. GPU PC에서 `lobby_01` RTSP 송출을 중지한다.
2. AI runner가 RTSP read 실패를 감지한다.
3. AI가 `safety/cameras/status`로 `camera_login_id=lobby_01`, `status=DISCONNECTED` 또는 `ERROR`를 발행한다.
4. 백엔드가 `Camera.connectionStatus`를 업데이트한다.
5. `/topic/camera-status` WebSocket/STOMP 메시지가 프론트에 도착한다.
6. `LiveCameraGrid`의 같은 카메라 카드 상태 배지가 오프라인/에러로 바뀌는지 확인한다.

### 12.3 SIMULATED_RTSP

1. 프론트에서 `sourceType=SIMULATED_RTSP`로 카메라를 등록한다.
2. 백엔드가 `assignedVideoPath`를 할당하는지 확인한다.
3. GPU PC runner가 해당 mp4를 `rtsp://GPU_PC_IP:8554/{cameraLoginId}` 또는 합의된 path로 반복 송출한다.
4. AI가 이 RTSP를 분석하고, 이벤트/상태 payload에 같은 `cameraLoginId`를 넣는지 확인한다.

## 13. 현재 코드에서 확인하지 못한 부분

- 실제 운영 DB migration/schema 파일은 이번 문서 작성 범위에서 확인하지 못했다. Entity 기준 필드는 확인했지만 실제 DB 컬럼 반영 상태는 확인 필요.
- `GET /api/cameras/active`를 GPU PC에서 호출할 때 인증이 필요한지, 운영 배포에서 접근 가능한지는 확인 필요.
- MediaMTX가 현재 설정으로 동적 path를 허용하는지 확인 필요. 현재 `stream/mediamtx.yml`에서는 `cam1`~`cam4` path 중심으로 확인된다.
- `displayStreamUrl`을 백엔드에서 생성할지, 프론트 환경변수로 생성할지는 구현 전 결정 필요.
- GPU PC에서 백엔드가 아닌 별도 runner가 FFmpeg/MediaMTX 송출을 담당할 때, 백엔드의 `RtspSimulationService`를 계속 사용할지 비활성화할지 확인 필요.

## 14. 다음 1단계 구현 명령어 후보

다음 Codex goal은 백엔드/프론트/AI를 한 번에 크게 바꾸기보다, 식별자와 표시 URL부터 작게 맞추는 것이 좋다.

```text
/goal 현재 repo 코드만 확인해서 1단계 최소 구현을 해줘.
목표:
- AI MQTT 이벤트는 camera_login_id를 백엔드가 우선 사용하도록 SafetyEventDto/AlertEventService를 수정
- CameraResponse에 connectionStatus를 포함
- 프론트 LiveCamera 변환과 AiAlert 매칭을 cameraLoginId 기준으로 수정
- RTSP를 직접 img src로 쓰지 않도록 displayStreamUrl 또는 VITE_STREAM_BASE_URL 기반 URL 계획 중 최소 구현 반영
- 기존 EMQX MQTT + WebSocket/STOMP 흐름 유지
- Redis Pub/Sub 사용 금지
```
