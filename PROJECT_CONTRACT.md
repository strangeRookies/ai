# PROJECT_CONTRACT.md

이 문서는 AI, Backend, Frontend, Infra/Docs 파트 간의 데이터 형식, 네트워크 통신 스펙 및 카메라 식별자에 관한 **공통 개발 계약**을 정의합니다. 모든 파트의 에이전트는 이 계약을 준수하여 작업해야 하며, 임의로 필드명을 변경하거나 스펙을 위반해서는 안 됩니다.

---

## 1. 카메라 식별자 및 스트리밍 규약

### 1.1. 카메라 식별자 (cameraLoginId)
- 카메라의 고유 논리 식별자로 **`cameraLoginId`** (DB 컬럼: `camera_login_id`)를 사용합니다.
- `cam1` ~ `cam4`와 같이 정적으로 하드코딩된 경로를 사용하거나 새로 만들지 않습니다.
- 모든 로직은 카메라 개수와 식별자를 **동적(Dynamic)**으로 처리해야 합니다.

### 1.2. RTSP Publish URL 스펙
- AI 엔진이 카메라의 RTSP 스트림을 MediaMTX에 발행할 때 사용하는 경로 규약:
  ```text
  rtsp://<host>:8554/{cameraLoginId}
  ```
  *(예: `rtsp://localhost:8554/lobby_01`)*
- `cam-` 등의 접두사(prefix)를 붙이거나 임의로 변형하지 않습니다.

### 1.3. HLS View URL 스펙
- 프론트엔드 대시보드에서 비디오 플레이어로 스트림을 렌더링하기 위해 사용하는 HLS 규약:
  ```text
  http://<host>:8888/{cameraLoginId}/index.m3u8
  ```
  *(예: `http://localhost:8888/lobby_01/index.m3u8`)*

---

## 2. MQTT 메시징 계약

- AI 분석 서버가 이벤트를 발행하고 백엔드가 구독하는 MQTT Topic: **`safety/events`**

### 2.1. AI Alert Event Payload 스펙
AI 엔진이 검출한 쓰러짐 등 안전 사고 이벤트를 전송할 때 사용하는 JSON 스키마입니다. 필드명과 타입을 철저히 준수해야 합니다.

```json
{
  "type": "fall_detected",
  "camera_id": "lobby_01",
  "camera_login_id": "lobby_01",
  "timestamp": "2026-06-18T08:54:41Z",
  "severity": "HIGH",
  "message": "쓰러짐 의심 상황이 감지되었습니다.",
  "source": "edge-ai",
  "track_id": 1,
  "metadata": {
    "bbox": [100, 150, 280, 390],
    "confidence": 0.87,
    "rule_score": 0.91,
    "pose_state": "LYING",
    "model_name": "yolo26n-pose"
  }
}
```

#### 필드 명세
| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `type` | String | 이벤트 타입 (예: `"fall_detected"`) |
| `camera_id` | String | 연동 호환성을 위해 `cameraLoginId` 값을 설정 |
| `camera_login_id`| String | 명시적인 카메라 로그인 ID 식별자 (`cameraLoginId`) |
| `timestamp` | String | ISO 8601 형식의 이벤트 발생 일시 |
| `severity` | String | 이벤트 위험 심각도 (`"INFO"`, `"WARNING"`, `"HIGH"`) |
| `message` | String | 운영자 알림용 텍스트 메시지 |
| `source` | String | 이벤트 발생 소스 식별자 (`"edge-ai"`) |
| `track_id` | Integer | (선택) 감지된 객체의 고유 트래킹 ID |
| `metadata` | Object | 상세 진단 메타데이터 객체 |
| `metadata.bbox` | Array | `[x_min, y_min, x_max, y_max]` 형식의 바운딩 박스 좌표 |
| `metadata.confidence` | Float | 객체 감지 및 포즈 모델 신뢰도 |
| `metadata.rule_score` | Float | 쓰러짐 판단 규칙 점수 또는 LSTM 확률값 |
| `metadata.pose_state` | String | 포즈 상태 판정 결과 (예: `"LYING"`, `"STANDING"`) |
| `metadata.model_name` | String | 분석에 사용된 모델명 (예: `"yolo26n-pose"`) |

---

## 3. 백엔드 및 프론트엔드 API/WebSocket 계약

### 3.1. Spring Boot Camera Response DTO
백엔드가 카메라 목록 및 단건 조회 시 전달하는 DTO 규약입니다.
- **`cameraId`**: 데이터베이스 PK (Numeric ID)
- **`cameraLoginId`**: 고유 식별 명칭 (String, 예: `"lobby_01"`)
- **`rtspUrl`**: 해당 카메라의 원본 또는 시뮬레이션 RTSP 소스 URL
- **`status`**: 활성화 여부
- **`aiEnabled`**: AI 분석 대상 지정 여부
- **`sourceType`**: 카메라 영상 소스 타입 (`"REAL_RTSP"`, `"SIMULATED_RTSP"`)
- **`assignedVideoPath`**: 시뮬레이션용 비디오 파일 경로

### 3.2. 카메라 상태 WebSocket 토픽
- Topic: `/topic/camera-status`
- Payload: 카메라의 연결 상태 정보를 전달합니다.
- Key: `cameraLoginId`를 기준으로 프론트엔드에서 매핑합니다.

---

## 4. 계약 변경 절차

기존 API, DTO, WebSocket payload 및 MQTT 필드를 변경해야 하는 경우, 코드를 직접 수정하기 전에 반드시 다음 절차를 따릅니다:
1. `PROJECT_CONTRACT.md`에 변경하려는 스펙을 수정 작성합니다.
2. 아래 **[계약 변경 이력]**에 변경 대상 파트, 필드명, 변경 사유, 변경 일자를 기록합니다.
3. Integration 에이전트와 전체 에이전트의 싱크를 확인한 후 작업을 진행합니다.

### 계약 변경 이력
| 버전 | 일자 | 변경자 | 변경 대상 | 변경 사유 및 내용 |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | 2026-06-18 | Antigravity | - | 초기 공통 계약 정의 |
