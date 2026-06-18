# 📡 MQTT 토픽별 메시지 규격 정의서 (MQTT Topic Specification)

본 문서는 AI Edge 분석 엔진, 백엔드 중계 서버, 그리고 프론트엔드 대시보드 간의 실시간 메시지 연동에 사용되는 MQTT 토픽 및 데이터 규격을 정의합니다.

---

## 📌 1. 개요 (Overview)

시스템은 실시간 이상행동 감지 및 카메라 연동 관제를 위해 MQTT 프로토콜을 사용합니다. AI 분석 엔진(Python)이 이벤트를 발행(Publish)하면, 백엔드 서버(Spring Boot)가 이를 구독(Subscribe)하여 DB에 저장하고 WebSocket(STOMP)을 통해 프론트엔드로 실시간 전달합니다.

사용되는 MQTT 토픽은 크게 **두 가지**입니다.
1. `safety/events`: AI 모델이 쓰러짐 등의 위험 이벤트를 검출했을 때 발행
2. `safety/cameras/status`: 카메라의 RTSP 연결 상태 변화가 있을 때 발행

---

## 🚨 2. 이상행동 감지 이벤트 토픽 (`safety/events`)

AI 분석 엔진이 프레임 분석을 거쳐 쓰러짐(Faint) 등의 이상행동 임계치 조건을 만족하면 경보 이벤트를 발행합니다. (불필요한 반복 경보를 막기 위해 Cooldown 10초 적용)

### A. 연동 흐름
* **발행(Publish):** AI Edge 서버 (Python)
* **구독(Subscribe):** 백엔드 Spring Boot

### B. Payload 예시 (JSON)
```json
{
  "message_type": "AI_EVENT",
  "event_type": "Faint",
  "type": "Faint",
  "camera_id": "cam_01",
  "camera_login_id": "cam_01",
  "timestamp": "2026-06-17T00:30:57Z",
  "detected_at": "2026-06-17T00:30:57Z",
  "severity": "HIGH",
  "confidence": 0.87,
  "score": 0.87,
  "faint_prob": 0.87,
  "bbox": [100, 150, 280, 390],
  "track_id": 7,
  "clip_path": "/home/welabs/yolo_training/clips/cam_01_faint.mp4",
  "clip_url": "http://127.0.0.1:8080/clips/cam_01_faint.mp4"
}
```

### C. 상세 필드 정의
| 필드명 | 타입 | 필수 여부 | 설명 |
| :--- | :--- | :--- | :--- |
| `message_type` | String | 필수 | 메시지 유형 식별용 상수 (`"AI_EVENT"`) |
| `event_type` / `type` | String | 필수 | 이상행동 종류 (`"Faint"` / `"Normal"` 등) |
| `camera_id` | String | 필수 | AI 내부 카메라 고유 식별자 |
| `camera_login_id` | String | 필수 | 백엔드/프론트엔드 매핑용 로그인 식별 ID (미지정시 `camera_id` 사용) |
| `timestamp` / `detected_at` | String | 필수 | ISO-8601 UTC 포맷 문자열 (`YYYY-MM-DDTHH:mm:ssZ`) 또는 Unix Epoch (초/Float) |
| `severity` | String | 필수 | 위험도 단계 (`"HIGH"` / `"CRITICAL"` / `"INFO"` 등) |
| `confidence` / `score` | Double | 필수 | 검출 신뢰도 및 판별 확률값 (0.0 ~ 1.0) |
| `faint_prob` | Double | 선택 | Faint 클래스의 원시 소프트맥스 확률값 |
| `bbox` | Array | 선택 | 이상행동 대상자의 바운딩 박스 좌표 `[x_min, y_min, x_max, y_max]` |
| `track_id` | Int / String | 선택 | 객체 추적 추적 고유 ID |
| `clip_path` | String | 선택 | 로컬 저장된 이벤트 비디오 클립 경로 |
| `clip_url` | String | 선택 | 클립 영상을 접근할 수 있는 Web URL |

### D. 관련 코드 레퍼런스
* **발행기 구현 (Python):** [build_inference_event_payload](file:///c:/Users/user/Documents/최종 쉴더스/ai/inference/rtsp_runtime.py#L138-L190)
* **구독기 구현 (Java):** [MqttSafetyEventSubscriber.java](file:///c:/Users/user/Documents/최종 쉴더스/strange_back/src/main/java/com/strange/safety/event/MqttSafetyEventSubscriber.java)
* **DTO 매핑 클래스 (Java):** [SafetyEventDto.java](file:///c:/Users/user/Documents/최종 쉴더스/strange_back/src/main/java/com/strange/safety/event/SafetyEventDto.java)

---

## 📹 3. 카메라 RTSP 연결 상태 토픽 (`safety/cameras/status`)

AI 분석 서버가 대상 카메라의 RTSP 스트림 연결을 상태 변화(연결 성공, 끊김, 재연결 시도 등)가 발생할 때마다 발행합니다. 중복 발행을 방지하여 상태 전환 시점에만 발행됩니다.

### A. 연동 흐름
* **발행(Publish):** AI Edge 서버 (Python)
* **구독(Subscribe):** 백엔드 Spring Boot

### B. Payload 예시 (JSON)
```json
{
  "message_type": "CAMERA_STATUS",
  "camera_login_id": "cam_01",
  "status": "CONNECTED",
  "detected_at": "2026-06-17T00:30:57Z",
  "previous_status": "DISCONNECTED",
  "reason": "RTSP_CONNECTED",
  "edge_device_id": "edge-ai-01",
  "rtsp_url_masked": "rtsp://user:***@192.168.1.100:554/stream",
  "camera_id": "cam_01"
}
```

### C. 상세 필드 정의
| 필드명 | 타입 | 필수 여부 | 설명 |
| :--- | :--- | :--- | :--- |
| `message_type` | String | 필수 | 메시지 유형 식별용 상수 (`"CAMERA_STATUS"`) |
| `camera_login_id` | String | 필수 | 백엔드/프론트엔드 매핑용 로그인 식별 ID |
| `status` | String | 필수 | 현재 연결 상태 (`"CONNECTED"` \| `"DISCONNECTED"` \| `"RECONNECTING"` \| `"ERROR"` \| `"DISABLED"`) |
| `detected_at` | String | 필수 | 감지 시점의 ISO-8601 UTC 포맷 문자열 |
| `previous_status` | String | 선택 | 직전 상태값 |
| `reason` | String | 선택 | 상태 변경 상세 사유 (예: `RTSP_TIMEOUT`, `RECONNECT_ATTEMPT_1`, `AUTH_FAILED`) |
| `edge_device_id` | String | 선택 | AI 엣지 디바이스 고유 식별자 |
| `rtsp_url_masked` | String | 선택 | 보안을 위해 계정 정보(패스워드)가 마스킹된 RTSP 스트림 주소 |
| `camera_id` | String | 선택 | AI 내부 카메라 고유 ID |

### D. 관련 코드 레퍼런스
* **발행기 구현 (Python):** [build_camera_status_payload](file:///c:/Users/user/Documents/최종 쉴더스/ai/publishers/camera_status_publisher.py#L24-L61) 및 [CameraStatusPublisher](file:///c:/Users/user/Documents/최종 쉴더스/ai/publishers/camera_status_publisher.py#L81)
* **구독기 구현 (Java):** [MqttSafetyEventSubscriber.java](file:///c:/Users/user/Documents/최종 쉴더스/strange_back/src/main/java/com/strange/safety/event/MqttSafetyEventSubscriber.java)
* **DTO 매핑 클래스 (Java):** [CameraStatusEventDto.java](file:///c:/Users/user/Documents/최종 쉴더스/strange_back/src/main/java/com/strange/safety/event/CameraStatusEventDto.java)

---

## 💡 4. 개발자를 위한 연동 및 역직렬화 팁

1. **Snake Case 와 Camel Case 호환성:**
   AI 파트(Python)에서는 주로 `snake_case`로 필드명을 발행하나, 백엔드(Java/Spring Boot)는 DTO 매핑 시 `@JsonAlias`를 활용하여 `camelCase`와 `snake_case`를 동시에 처리합니다. 따라서 추가 필드를 정의할 때 `@JsonAlias` 또는 `@JsonProperty`를 활용하여 이름 불일치 에러를 방지해야 합니다.

2. **유연한 Timestamp 처리:**
   * AI 서버가 발행하는 timestamp가 Float 형식(Unix epoch 초)일 경우와 ISO-8601 문자열 포맷일 경우를 모두 허용하기 위해 백엔드 [SafetyEventDto.java](file:///c:/Users/user/Documents/최종 쉴더스/strange_back/src/main/java/com/strange/safety/event/SafetyEventDto.java)는 `rawTimestamp(Object)` 필드로 받은 뒤 `resolvedTimestamp()` 도우미 메소드를 통해 동적으로 `Instant`로 변환하여 처리합니다.
