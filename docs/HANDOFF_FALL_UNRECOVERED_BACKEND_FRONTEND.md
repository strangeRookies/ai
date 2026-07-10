# 핸드오프 이슈: 낙상 후 미회복 / 실신 의심 이벤트 연동

**From:** AI Agent (`strange_ai`)  
**To:** Backend Agent · Frontend Agent  
**Priority:** High (운영 알림 UX / 오탐 재알림 방지)  
**Status:** AI 발행 구현 완료 · Back/Front 소비 구현 대기  
**Related:** [FALL_EVENT_LIFECYCLE_MQTT_CONTRACT.md](./FALL_EVENT_LIFECYCLE_MQTT_CONTRACT.md), [MQTT_TOPIC_SPEC.md](./MQTT_TOPIC_SPEC.md)

---

## 1. 배경 / 왜 필요한가

기존 낙상 알림은 LSTM Faint + **10초 camera cooldown** 으로 중복을 줄였습니다.  
하지만 cooldown이 끝나면 사람이 **계속 누워 있어도** 같은 상황을 **새 Fall** 로 다시 올릴 수 있었습니다.

AI 쪽에서는 아래처럼 분리했습니다.

| 구분 | 의미 | MQTT `type` / `event_type` | `alertKind` |
|------|------|---------------------------|-------------|
| 신규 낙상 | upright→lying 등 조건으로 **처음 확정** 1회 | `faint` / `fall` | `new_fall` |
| 미회복 위험 | 확정 후 cooldown(기본 10s) 지나도 **계속 lying / Faint** | **`FAINT_SUSPECTED`** 또는 **`FALL_UNRECOVERED`** | **`unrecovered`** |

- `new_fall` 재발행은 `POST_FALL_LYING` 동안 **하지 않음**
- 미회복은 **별도 event type** 으로 발행 (무시하면 안 됨)
- 회복(`RECOVERED`) 후에는 다시 `new_fall` 가능

---

## 2. 범위

### AI (완료)

- 상태머신 + posture + MQTT payload 필드 발행
- 토픽: 기존과 동일 **`safety/events`**

### Backend (요청)

- 구독/파싱 시 새 type 허용 및 저장
- 프론트로 WebSocket/STOMP 전달 시 type·문구 유지
- (권장) `originalEventId` 로 최초 낙상과 미회복 알림 연결

### Frontend (요청)

- 알림/대시보드에서 **NEW_FALL 과 다른 UI 문구·스타일**
- unknown type drop 금지

### Out of scope (이번 이슈)

- AI 추론 파라미터 튜닝
- MediaMTX / RTSP 송출 변경

---

## 3. MQTT 계약 (소비 측 필수)

- **Topic:** `safety/events`
- **messageType:** `event` (schemaVersion `1.1`)

### 3.1 NEW_FALL 예시

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

### 3.2 미회복 예시 (`FAINT_SUSPECTED`)

```json
{
  "schemaVersion": "1.1",
  "messageType": "event",
  "eventId": "…",
  "cameraLoginId": "cam_01",
  "type": "FAINT_SUSPECTED",
  "event_type": "FAINT_SUSPECTED",
  "alertKind": "unrecovered",
  "memoText": "낙상 후 계속 누워 있음",
  "originalEventId": "…-first-new-fall-id",
  "durationSec": 12.4,
  "trackingId": 7,
  "state": "POST_FALL_LYING",
  "postureLabel": "lying_like",
  "movementLevel": "still",
  "confidence": 0.77
}
```

`FALL_UNRECOVERED` 도 동일 스키마 (`type` 만 다름, Fall 라벨 경로).

### 3.3 필드 체크리스트

| 필드 | NEW_FALL | 미회복 | 비고 |
|------|:--------:|:------:|------|
| `type` / `event_type` | ✓ | ✓ | 대소문자 구분 (`FAINT_SUSPECTED`) |
| `alertKind` | `new_fall` | `unrecovered` | 분기 키로 권장 |
| `memoText` / `message` | 쓰러짐 의심! | 낙상 후 미회복… | UI 기본 문구 |
| `cameraLoginId` | ✓ | ✓ | |
| `trackingId` / `track_id` | ✓ | ✓ | |
| `originalEventId` | – | ✓ | 최초 Fall 연결 |
| `durationSec` | – | ✓ | 확정 후 경과 초 |
| `postureLabel` | 권장 | 권장 | |
| `movementLevel` | 권장 | 권장 | still/low/high/unknown |
| `state` | 권장 | 권장 | e.g. POST_FALL_LYING |

상세: `docs/FALL_EVENT_LIFECYCLE_MQTT_CONTRACT.md`

---

## 4. Backend 작업 항목 (Acceptance)

- [ ] `MqttSafetyEventSubscriber` / DTO 가 `FAINT_SUSPECTED`, `FALL_UNRECOVERED` 를 **reject/ignore 하지 않음**
- [ ] DB 또는 JSON 확장에 최소 저장:  
      `event_type`, `alert_kind`, `original_event_id`, `duration_sec`, `posture_label`, `movement_level`, `lifecycle_state`, `track_id`, `camera_login_id`, `memo_text`
- [ ] REST/WebSocket 응답에 동일 필드 노출 (프론트가 구분 표시 가능)
- [ ] 기존 `faint` / `fall` 동작 회귀 없음
- [ ] (권장) 알림 테이블에서 `original_event_id` 로 타임라인 조회 API 또는 조인 가능

**참고 코드 (repo 구조 가정):**

- `strange_back/.../MqttSafetyEventSubscriber.java`
- `strange_back/.../SafetyEventDto.java` (또는 동등 DTO)

---

## 5. Frontend 작업 항목 (Acceptance)

- [ ] 이벤트 카드/토스트/사이드바에서 type 분기:

| 조건 | 표시 문구 (권장) | UI 톤 |
|------|------------------|--------|
| `alertKind === "new_fall"` 또는 type `faint`/`fall` | 쓰러짐 의심! | 기존 긴급 알림 |
| `type === "FAINT_SUSPECTED"` 또는 memo 해당 | **낙상 후 미회복/실신 의심** | 지속 위험 (주황 등) |
| `type === "FALL_UNRECOVERED"` | **낙상 후 계속 누워 있음** | 지속 위험 |
| 회복 후 새 `new_fall` | 새 쓰러짐 이벤트 | 신규와 동일 스타일 |

- [ ] `durationSec` 있으면 “N초 지속” 등 보조 표시
- [ ] `originalEventId` 있으면 최초 낙상과 링크/그룹 (가능 시)
- [ ] unknown type 을 조용히 drop 하지 말 것 (최소 raw type 표시)

---

## 6. 검증 시나리오 (E2E)

1. 사람이 넘어짐 → **1회** `faint` + `alertKind=new_fall`
2. 10초 이내 계속 누워 있음 → **추가 faint 없음**
3. 10초 이후 계속 누워 있음 → **`FAINT_SUSPECTED`** (또는 FALL_UNRECOVERED),  
   `originalEventId` = 1번 eventId, `durationSec` ≥ 10
4. 일어남(회복) 후 다시 넘어짐 → **새** `new_fall` (다른 eventId)
5. 프론트: 1번과 3번 문구/색이 **다름**
6. 백엔드 DB: 3번 row 에 type·original_event_id 저장 확인

---

## 7. 비기능 / 주의

- 미회복 재발행 간격 AI 기본 **30초** (`unrecovered_repeat_seconds`) — 스팸 방지
- NEW_FALL camera cooldown(10s) 은 **신규 Fall 전용 보조 락**; 미회복 이벤트는 이 락으로 막지 않음
- 필드명 camelCase / snake_case 둘 다 올 수 있음 → DTO 는 둘 다 수용 권장

---

## 8. AI 쪽 구현 위치 (참고만)

| 역할 | 경로 |
|------|------|
| 상태머신 | `ai/action/fall_event_state.py` |
| Posture | `ai/action/posture_estimator.py` |
| evaluate / emit | `ai/action/faint_post_processing.py` |
| Payload | `ai/publishers/mqtt_payloads.py` |
| 발행 | `scripts/run_rtsp_inference.py`, `scripts/serve_ai_overlay.py` |

---

## 9. 이슈 타이틀 제안 (복붙용)

**Backend**

```text
[AI Handoff] safety/events: FAINT_SUSPECTED / FALL_UNRECOVERED 수신·저장·STOMP 전달
```

**Frontend**

```text
[AI Handoff] 낙상 알림 UI: new_fall vs FAINT_SUSPECTED/FALL_UNRECOVERED 문구·스타일 분리
```

---

## 10. Definition of Done

- [ ] Backend: 새 type 저장·푸시 E2E 통과
- [ ] Frontend: 문구 분리 QA 통과 (위 시나리오 1–5)
- [ ] 기존 faint 알림 회귀 없음
- [ ] (선택) Integration: GPU PC 실스트림 1회 미회복 이벤트 스크린샷/로그 첨부

---

**문의 시 참고 로그 (AI)**

```text
[yolo-runtime] …          # 추론 backend
memoText / type in safety/events payload
summary unrecovered_events_generated
```
