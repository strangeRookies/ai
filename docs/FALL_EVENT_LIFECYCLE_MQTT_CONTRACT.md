# Fall / Faint 이벤트 라이프사이클 · MQTT 계약 (Phase C)

AI Edge (`strange_ai`) → MQTT `safety/events` → Backend → Frontend 연동용 계약 정리.

---

## 1. 이벤트 종류 (eventType / type)

| eventType | alertKind | 의미 | 프론트 권장 문구 |
|-----------|-----------|------|------------------|
| `faint` / `fall` | `new_fall` | 신규 낙상/실신 **1회 확정** | 쓰러짐 의심! |
| `FAINT_SUSPECTED` | `unrecovered` | 낙상 확정 후 cooldown 지나도 미회복·실신 지속 의심 | **낙상 후 미회복/실신 의심** |
| `FALL_UNRECOVERED` | `unrecovered` | 낙상 확정 후 계속 누워 있음 (Fall 라벨 경로) | **낙상 후 계속 누워 있음** |

- **NEW_FALL 재발행 금지**: `POST_FALL_LYING` 동안 `faint`/`fall` 을 다시 보내지 않음.
- **미회복 알림**: `unrecovered_after_seconds` (기본 = camera cooldown 10s) 이후 `FAINT_SUSPECTED` / `FALL_UNRECOVERED` 발행.
- **회복 후**: `RECOVERED` → 다시 `new_fall` 가능.

---

## 2. MQTT 토픽

- 토픽: **`safety/events`** (기존과 동일)
- `messageType`: `"event"` (현재 AI 발행 스키마 `schemaVersion` 1.1)

---

## 3. Payload 필드 (AI 발행 기준)

### 공통

| 필드 | 타입 | 설명 |
|------|------|------|
| `cameraLoginId` / `camera_login_id` | string | 카메라 식별자 |
| `eventId` | string | 이번 알림 ID |
| `type` / `event_type` | string | 위 eventType |
| `memoText` / `message` | string | 표시 문구 |
| `alertKind` / `alert_kind` | string | `new_fall` \| `unrecovered` |
| `trackingId` / `track_id` / `trackId` | int | track id |
| `confidence` | float | LSTM faint 등 |
| `bbox` / `boundingBox` | object/array | 박스 |
| `state` / `lifecycleState` | string | `POST_FALL_LYING` 등 |
| `postureLabel` / `posture_label` | string | `upright_like` \| `lying_like` \| `unknown` |
| `movementLevel` / `movement_level` | string | `still` \| `low` \| `high` \| `unknown` |

### unrecovered 전용

| 필드 | 타입 | 설명 |
|------|------|------|
| `originalEventId` / `original_event_id` | string | 최초 NEW_FALL 의 eventId |
| `durationSec` / `duration_sec` | float | 최초 확정 이후 경과 초 |

### 예시: NEW_FALL

```json
{
  "schemaVersion": "1.1",
  "messageType": "event",
  "eventId": "…",
  "cameraLoginId": "cam_01",
  "type": "faint",
  "event_type": "faint",
  "alertKind": "new_fall",
  "memoText": "쓰러짐 의심!",
  "trackingId": 7,
  "state": "POST_FALL_LYING",
  "postureLabel": "lying_like",
  "movementLevel": "low",
  "confidence": 0.81
}
```

### 예시: FAINT_SUSPECTED (미회복)

```json
{
  "schemaVersion": "1.1",
  "messageType": "event",
  "eventId": "…-unrecovered",
  "cameraLoginId": "cam_01",
  "type": "FAINT_SUSPECTED",
  "event_type": "FAINT_SUSPECTED",
  "alertKind": "unrecovered",
  "memoText": "낙상 후 계속 누워 있음",
  "originalEventId": "…-original-new-fall",
  "durationSec": 12.4,
  "trackingId": 7,
  "state": "POST_FALL_LYING",
  "postureLabel": "lying_like",
  "movementLevel": "still",
  "confidence": 0.77
}
```

---

## 4. 백엔드 연동 권장 (Backend Agent 핸드오프)

> AI 에이전트는 `strange_back`을 수정하지 않음. 아래는 구현 가이드.

1. `event_type` / `type` 에 `FAINT_SUSPECTED`, `FALL_UNRECOVERED` 허용.
2. DB/알림 테이블에 `original_event_id`, `duration_sec`, `alert_kind`, `posture_label`, `movement_level`, `lifecycle_state` 컬럼 또는 JSON 확장 필드.
3. STOMP/WebSocket 푸시 시 **NEW_FALL 과 다른 severity 아이콘·문구** 매핑.
4. 동일 `originalEventId` 로 미회복 알림을 그룹핑하면 대시보드 UX에 유리.

---

## 5. 프론트 연동 권장 (Frontend Agent 핸드오프)

> AI 에이전트는 `strange_front`를 수정하지 않음.

| alertKind / type | UI |
|------------------|-----|
| `new_fall` + faint/fall | 기존 쓰러짐 배지/사운드 |
| `unrecovered` + FAINT_SUSPECTED | “낙상 후 미회복/실신 의심” (주황/지속 위험) |
| `unrecovered` + FALL_UNRECOVERED | “낙상 후 계속 누워 있음” |
| 동일 track 재 fall (`new_fall` after RECOVERED) | 새 이벤트로 표시 |

---

## 6. AI 구현 위치

| 구성요소 | 경로 |
|----------|------|
| 상태머신 | `ai/action/fall_event_state.py` |
| Posture | `ai/action/posture_estimator.py` |
| evaluate / emit | `ai/action/faint_post_processing.py` |
| MQTT payload | `ai/publishers/mqtt_payloads.py` |
| 발행 루프 | `scripts/run_rtsp_inference.py`, `scripts/serve_ai_overlay.py` |

---

## 7. 파라미터

| 이름 | 기본 | 의미 |
|------|------|------|
| `camera_cooldown_seconds` | 10 | NEW_FALL 카메라 보조 쿨다운 |
| `unrecovered_after_seconds` | = cooldown | 미회복 이벤트 최초 발행 지연 |
| `unrecovered_repeat_seconds` | 30 | 미회복 재발행 간격 (0=1회만) |
| `require_upright_to_lying` | true | 처음부터 누워 있음 NEW_FALL 억제 |

---

## 8. 검증 체크리스트

1. NEW_FALL 1회 후 10s 내: 추가 `faint` 없음.
2. 10s 이후 lying 유지: `FAINT_SUSPECTED` (또는 FALL_UNRECOVERED) 1회+, `originalEventId` = 최초 Fall.
3. payload 에 `postureLabel`, `movementLevel`, `state`, `durationSec` 존재.
4. upright 회복 후 재 fall: 새 `eventId` 의 `new_fall`.
5. 프론트/백엔드가 unknown type 으로 drop 하지 않는지 확인 (핸드오프).
