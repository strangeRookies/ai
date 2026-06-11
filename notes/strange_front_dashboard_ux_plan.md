# 대시보드 UX 패치 계획

날짜: 2026-06-09
대상 저장소: `strange_front`, `strange_back`

## 현재 대시보드가 신뢰하기 어려운 이유

1. 프론트엔드에 실시간 알림 경로가 두 개 있다.
   - `src/app/App.tsx`가 `useAlertWebSocket()`를 전역으로 mount한다.
   - `src/hooks/useAiAlertActions.ts`도 내부에서 `useAiEvents()`로 다시 구독한다.
   - 결과: 백엔드에서 온 동일 이벤트가 프론트 안에서 두 번 surface될 수 있다.

2. 이벤트 식별자가 너무 불안정하다.
   - `aiEventKey()`는 `camera_id`, `event_type`, `timestamp`, `track_id`를 함께 쓴다.
   - 백엔드가 같은 위험 상황을 약 2초 간격으로 새 timestamp와 함께 다시 publish하면, 프론트는 그것을 새 경고로 판단한다.
   - 결과: 사용자가 확인한 뒤에도 같은 알림이 계속 다시 나타난다.

3. 실제 이벤트가 없어도 UI가 처음부터 위험 상태처럼 보이는 부분이 있다.
   - `src/features/dashboard/data/cameras.ts`는 camera 2를 기본 `eventStatus: 'danger'`로 하드코딩한다.
   - `src/features/dashboard/pages/UserDashboard.tsx`는 `INITIAL_ALERTS`에 mock 감지 데이터를 미리 채운다.
   - 결과: MQTT/STOMP payload가 없어도 CCTV 그리드와 알림 위젯이 위험 상황처럼 보인다.

4. 재생 모달과 사이드바 일부 문구가 하드코딩돼 있다.
   - 예: `Detected`, `이상 거동 감지 (CRITICAL)`, 고정 타임스탬프
   - 결과: 실제 feed 상태와 무관하게 “감지됨”처럼 읽힌다.

## 변경해야 할 파일

### `strange_front`

- `src/app/App.tsx`
  - 전역 `useAlertWebSocket()` mount 제거
  - 실시간 사건 소비자는 대시보드 하나만 남기기

- `src/shared/types/aiEvents.ts` 신규
  - `AiEvent`, feed 상태, 연결 상태 타입 정리

- `src/shared/utils/aiEventFeed.ts` 신규
  - stable fingerprint 생성
  - `cameraId + eventType + trackId + bbox` 기준 반복 이벤트 dedupe
  - 예: 15초 stale window 안에서만 활성 이벤트 유지

- `src/shared/utils/aiAlerts.ts`
  - helper 추가
    - `aiEventFingerprint()`
    - `formatAiEventLabel()`
    - `getSeverityTone()`
  - 깨진 한글 라벨도 함께 정리

- `src/hooks/useAiEvents.ts`
  - bare array 대신 구조화된 feed state 반환
  - `connectionState`, `lastEventAt`, deduplicated `events` 추적
  - STOMP와 SSE payload를 하나의 parser로 정규화

- `src/hooks/useAiAlertActions.ts`
  - timestamp 포함 key 대신 stable fingerprint로 acknowledge
  - 이벤트가 active window에서 사라지면 acknowledged fingerprint 제거
  - repeating alarm은 아직 확인되지 않은 active event에만 적용

- `src/components/dashboard/AiAlertCard.tsx`
  - `Detected` / `Acknowledged` 같은 하드코딩 영문 문구를 상태 기반 문구로 교체
  - 실제 이벤트 라벨, 시간, confidence, 확인 상태 표시

- `src/components/dashboard/AiDangerPanel.tsx`
  - 빈 상태를 “알림 없음”이 아니라 모니터링/대기 상태로 표현

- `src/features/dashboard/data/cameras.ts`
  - seeded fake danger 상태 제거

- `src/features/dashboard/components/LiveCameraGrid.tsx`
  - 카메라 카드별 fullscreen 버튼 추가
  - 표시 개선
    - 최근 이벤트 시간
    - 확인됨 vs 모니터링 상태
    - 연결 상태 badge

- `src/features/dashboard/pages/UserDashboard.tsx`
  - `INITIAL_ALERTS` mock 활성 경고 제거 또는 demo 전용으로 명확히 분리
  - alert sync를 raw timestamp key가 아니라 stable fingerprint 기준으로 변경
  - header 상태를 `connectionState`에서 파생
    - monitoring / connecting / disconnected / no recent event
  - AI sidebar fallback을 진짜 empty state로 교체
  - playback modal overlay text와 시간도 실제 incident 기반으로 표시
  - `LiveCameraGrid`에 카메라별 event summary 전달

- `scripts/verify-ai-acknowledge-contract.mjs`
  - 대시보드 경로 수정
    - 기존: `src/app/pages/NurseDashboard.tsx`
    - 변경: `src/features/dashboard/pages/UserDashboard.tsx`
  - local confirm check도 fingerprint 기반 acknowledge 로직에 맞게 수정

### `strange_back`

- 이번 범위에서는 문서 위주
  - `PROJECT_SUMMARY.md`
  - 또는 신규 `docs/TROUBLESHOOTING.md`
  - 설명할 내용
    - 프론트 가정과 STOMP 이벤트 동작이 어긋났던 지점
    - timestamp 기반 재발행이 왜 반복 경고로 보였는지
    - 현재 프론트 suppression 정책이 어떻게 바뀌는지

## 권장 suppression 정책

1. 아래 필드로 stable fingerprint 생성
   - 정규화된 `camera_id`
   - 정규화된 `event_type`
   - 있으면 `track_id`
   - 있으면 `bbox`

2. 같은 fingerprint의 반복 이벤트는 일정 active window, 예: 15초 안에서는 같은 사건으로 취급

3. 사용자가 확인을 누르면
   - raw timestamp key가 아니라 fingerprint를 acknowledge
   - 그 fingerprint에 대해서는 반복 렌더링과 반복 알람음 억제
   - 단, 이벤트는 acknowledged 상태로 보이도록 유지

4. 해당 fingerprint가 active set에서 사라지면
   - acknowledgement suppression도 제거
   - 나중에 다시 나타나면 새 사건으로 다시 알림 가능

## 쓰기 권한이 열렸을 때의 검증 계획

1. `scripts/verify-ai-acknowledge-contract.mjs` 경로 수정 후 `npm.cmd run test:ai-ack`
2. `vite.config.ts`를 읽을 수 있는 환경에서 `npm.cmd run build`
3. 브라우저 수동 QA
   - 이벤트 없음 -> CCTV 그리드는 monitoring/normal 상태
   - 첫 live event 수신 -> alert card 1회 표시
   - confirm 클릭 -> 같은 이벤트가 2초마다 다시 뜨지 않음
   - 이후 새 event/변경된 event 수신 -> 새 alert 다시 표시
   - fullscreen 버튼으로 카메라 카드 전체화면 진입/종료 가능

## 현재 블로커

이 Codex 세션은 `strange_front`와 `strange_back`를 읽을 수는 있지만 쓸 수는 없다. 따라서 `Set-Content`나 직접 patch 같은 실제 구현은 access denied로 막히고, 최종 적용은 해당 저장소가 writable root로 열린 세션에서 진행해야 한다.
