# strange_front 대시보드 패치 블루프린트

날짜: 2026-06-09
상태: 현재 세션에서는 `strange_front`가 쓰기 불가라 실제 적용은 막혀 있음

이 문서는 쓰기 가능한 세션이 열렸을 때 그대로 반영할 수 있도록, 필요한 수정 사항을 파일 단위로 정리한 적용 가이드다.

## 1. 전역 실시간 리스너 중복 제거

### 대상 파일
- `C:\Users\user\Documents\최종 쉴더스\strange_front\src\app\App.tsx`

### 현재 근거
- 현재 `App.tsx`는 `useAlertWebSocket`를 import 해서 전역으로 mount하고 있다.
- 동시에 대시보드 내부에서도 `useAiAlertActions()` -> `useAiEvents()`를 통해 같은 실시간 이벤트를 구독한다.

### 수정 내용
- 아래 두 줄 제거
  - `import { useAlertWebSocket } from '../hooks/useAlertWebSocket';`
  - `useAlertWebSocket();`

### 왜 중요한가
- 동일한 STOMP 이벤트가 두 경로로 소비되면서 중복 알림과 toast 스팸이 발생하는 것을 막는다.

## 2. 불안정한 이벤트 식별자 수정

### 대상 파일
- `src/hooks/useAiAlertActions.ts`
- `src/shared/utils/aiAlerts.ts`
- 필요하면 신규 파일 `src/shared/utils/aiEventFeed.ts`

### 현재 문제
- `aiEventKey()`가 현재 아래 조합을 쓴다.
  - `${camera_id}:${event_type}:${timestamp}:${track_id}`
- 백엔드가 같은 위험 이벤트를 약 2초마다 새 `timestamp`로 다시 보내면, 프론트는 그것을 완전히 새로운 이벤트로 본다.

### 수정 방향
- 안정적인 fingerprint helper를 추가한다.
- 조합 기준:
  - 정규화된 `camera_id`
  - 정규화된 `event_type`
  - `track_id`가 있으면 사용, 없으면 `no-track`
  - `bbox`가 있으면 사용, 없으면 `no-bbox`

예시:

```ts
export function aiEventFingerprint(event: AiEvent) {
  const normalizedType = event.event_type.trim().toUpperCase();
  const normalizedCamera = event.camera_id.trim().toLowerCase();
  const normalizedTrack = event.track_id?.trim() || 'no-track';
  const bboxKey = Array.isArray(event.bbox) ? event.bbox.join(',') : 'no-bbox';
  return `${normalizedCamera}:${normalizedType}:${normalizedTrack}:${bboxKey}`;
}
```

### `useAiAlertActions.ts`에서 바꿀 부분
- acknowledged 상태를 timestamp 기반 key가 아니라 stable fingerprint 기반으로 변경
- 아래 코드 대체
  - `acknowledgedAiEventIds.has(aiEventKey(event))`
  - `next.add(aiEventKey(event))`

### 만료 정책도 추가
- 현재 활성 `dangerAiEvents` 목록에 더 이상 없는 fingerprint는 acknowledged set에서 제거
- 그래야 나중에 진짜 새 사건이 다시 발생했을 때 정상적으로 다시 알릴 수 있다.

### 왜 중요한가
- 사용자가 확인한 뒤에도 같은 사건이 2초마다 다시 뜨는 문제를 직접 막는다.

## 3. 수신 이벤트를 무조건 쌓지 말고 dedupe 처리

### 대상 파일
- `src/hooks/useAiEvents.ts`

### 현재 문제
- 현재는 모든 정규화 이벤트를 아래처럼 무조건 앞에 추가한다.

```ts
setEvents((prev) => [normalized, ...prev].slice(0, 50));
```

- 같은 활성 사건이 반복 들어와도 묶지 않는다.

### 수정 방향
- reducer를 두고 다음 규칙으로 축약한다.
  - 짧은 활성 윈도우, 예: 15초보다 오래된 이벤트는 제거
  - 동일 fingerprint의 기존 이벤트가 있으면 교체
  - 최신순 정렬 유지

예시 정책:

```ts
const STALE_EVENT_WINDOW_MS = 15000;

function eventTimestampMs(event: AiEvent) {
  return event.timestamp * 1000;
}

function pruneExpiredAiEvents(events: readonly AiEvent[], nowMs: number) {
  return events.filter((event) => nowMs - eventTimestampMs(event) <= STALE_EVENT_WINDOW_MS);
}

function reduceAiEventFeed(events: readonly AiEvent[], incomingEvent: AiEvent, nowMs: number) {
  const activeEvents = pruneExpiredAiEvents(events, nowMs);
  const incomingFingerprint = aiEventFingerprint(incomingEvent);
  const nextEvents = activeEvents.filter(
    (event) => aiEventFingerprint(event) !== incomingFingerprint,
  );
  return [incomingEvent, ...nextEvents].slice(0, 12);
}
```

그 후 setter를 `reduceAiEventFeed` 기반으로 바꾼다.

### 왜 중요한가
- 같은 라이브 사건이 여러 프레임으로 반복 들어와도 “새 사건 연속 발생”처럼 보이지 않게 한다.

## 4. 감지 상태를 기본 표시하지 말고 연결 상태를 분리

### 대상 파일
- `src/hooks/useAiEvents.ts`

### 추가할 상태
- 구조화된 feed state
  - `events`
  - `connectionState`
  - `lastEventAt`
  - `source`

### 필요한 상태값
- `idle`
- `connecting`
- `connected`
- `disconnected`
- `error`

### 왜 중요한가
- 실제 이벤트가 없어도 “감지됨”처럼 보이는 대신, 모니터링/연결중/끊김 상태를 정확히 렌더링할 수 있다.

## 5. 카메라 기본 danger 상태 제거

### 대상 파일
- `src/features/dashboard/data/cameras.ts`

### 현재 문제
- `camera-2`가 기본값부터 아래 상태다.

```ts
eventStatus: 'danger',
eventLabel: 'FALL',
```

### 수정 내용
- 기본값을 아래처럼 변경

```ts
eventStatus: 'normal',
```

### 왜 중요한가
- 실제 이벤트가 하나도 없는데도 CCTV 그리드가 위험 상황처럼 보이는 문제를 없앤다.

## 6. 경고 카드 문구를 상태 기반으로 변경

### 대상 파일
- `src/components/dashboard/AiAlertCard.tsx`

### 현재 문제 문구
- `{event.event_type} Detected`
- `Acknowledged`
- `Unacknowledged`
- `Confirm`

### 수정 방향
- 제목:
  - `FALL (낙상) 감지`처럼 포맷된 실제 이벤트 라벨 사용
- 부제:
  - acknowledged면 `조치 확인됨`
  - 새 활성 이벤트면 `새 위험 이벤트 수신`
- 버튼:
  - `확인`

### 왜 중요한가
- 실제 상태와 관계없는 일반적인 “Detected” 문구를 없애고, 운영자가 이해하기 쉬운 문구로 바꾼다.

## 7. AI 사이드바의 빈 상태 문구 개선

### 대상 파일
- `src/components/dashboard/AiDangerPanel.tsx`

### 현재 문제
- 빈 상태가 사실상 “AI 위험 알림 없음”이라고 말하면서도, fallback에서는 mock `alerts` 기반 카드가 계속 보일 수 있다.

### 수정 방향
- 빈 상태 문구를 아래처럼 변경
  - `실시간 위험 이벤트 대기 중`
  - `새 MQTT/STOMP 이벤트가 들어오면 여기에 표시됩니다.`
- 기본 empty-state fallback에서 seeded mock alert 카드 사용 중단

### 왜 중요한가
- “이벤트 없음”은 정상 모니터링 상태여야지, 가짜 과거 위험 상황처럼 보이면 안 된다.

## 8. `UserDashboard.tsx` 동기화 로직 수정

### 대상 파일
- `src/features/dashboard/pages/UserDashboard.tsx`

### 현재 문제
- `INITIAL_ALERTS`가 가짜 활성 사건을 seed한다.
- 이벤트 동기화가 여전히 `aiEventKey(event)`를 써서 timestamp에 민감하다.
- 홈 화면 헤더 상태 배지가 사실상 하드코딩이다.
- 재생 모달 overlay 텍스트와 시간도 하드코딩이다.

### 필요한 수정

1. `INITIAL_ALERTS`를 아래 둘 중 하나로 바꾸기
   - 빈 배열
   - 또는 명시적인 mock flag 뒤에 숨긴 demo 전용 데이터

2. 동기화 `useEffect`에서
   - stable fingerprint 기반 중복 억제
   - fingerprint 기준으로 acknowledged면 `status: 'resolved'`

3. feed state 기반 UI 상태 추가
   - `connecting` -> `이벤트 채널 연결 중`
   - `connected` + live event 없음 -> `실시간 모니터링 중`
   - `disconnected` -> `이벤트 채널 재연결 대기`

4. 홈 CCTV 영역
   - 해당 카메라에 실제 이벤트가 없으면 감지 문구를 보여주지 않기

5. 재생 모달
   - 아래 하드코딩 문구 제거
     - `이상 거동 감지 (CRITICAL)`
     - 고정 `2026-05-26`
     - 고정 `감지 타임스탬프 (00:30)`
   - 대신 선택된 incident에서 직접 렌더링
     - incident label
     - 실제 `selectedIncident.time`
     - acknowledged/new 상태

### 왜 중요한가
- 사용자가 “작동은 하지만 믿기 어렵다”고 느끼는 핵심 지점이 바로 이 파일이다.

## 9. 카메라 카드 전체화면 버튼 추가

### 대상 파일
- `src/features/dashboard/components/LiveCameraGrid.tsx`

### 현재 문제
- CCTV 카드별 전체화면 진입 UI가 없다.

### 수정 방향
- 각 카드에 `Expand` 아이콘 버튼 추가
- 브라우저 Fullscreen API 사용

예시:

```ts
const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

async function handleFullscreen(cameraId: string) {
  const target = cardRefs.current[cameraId];
  if (!target || !target.requestFullscreen) return;
  await target.requestFullscreen();
}
```

### 함께 개선할 부분
- 카드 하단에 표시
  - 최근 이벤트 시간
  - 확인됨 / 모니터링 상태
  - 연결 상태

### 왜 중요한가
- 전체화면 요구사항을 충족하고, 과도한 경고 표현 없이도 운영 정보가 더 잘 보이게 한다.

## 10. 로컬 검증 스크립트 경로 수정

### 대상 파일
- `scripts/verify-ai-acknowledge-contract.mjs`

### 현재 문제
- 아직도 아래 경로를 읽는다.

```ts
readFileSync('src/app/pages/NurseDashboard.tsx', 'utf8');
```

### 수정 내용

```ts
readFileSync('src/features/dashboard/pages/UserDashboard.tsx', 'utf8');
```

### 추가 수정
- 아래 체크도 함께 변경
  - `next.add(aiEventKey(event))`
- 새 fingerprint 기반 확인 로직에 맞게 바꾸기

### 왜 중요한가
- 현재 로컬 확인 스크립트 중 acknowledge 흐름을 직접 보는 유일한 스크립트인데, 경로가 이미 낡아 있다.

## 11. 문서 업데이트

### 대상 파일
- `strange_back/PROJECT_SUMMARY.md`
- 또는 신규 `strange_back/docs/TROUBLESHOOTING.md`

### 포함할 내용
- 무엇이 동작하지 않았는지
  - 프론트가 동일한 라이브 사건을 중복 채널로 소비
  - timestamp 기반 식별자로 같은 사건이 새 사건처럼 반복 표시
  - seeded fake danger UI 때문에 실제 payload가 없어도 감지처럼 보임
- 추정 원인
- 무엇을 바꿨는지
- 어떻게 검증했는지
- 남은 제한 사항

### 제외할 내용
- 비밀값
- 카메라 비밀번호
- 민감한 IP/계정 정보

## 12. 적용 후 검증 체크리스트

### 프론트
- `npm.cmd run test:ai-ack`
- `npm.cmd run build`
  - 샌드박스 때문에 `vite.config.ts` 접근이 계속 막히면, 일반 로컬 셸에서 한 번 더 확인

### 수동 QA
1. 이벤트 미수신
   - 대시보드가 monitoring/normal 상태로 보일 것
   - 첫 진입 시 fake danger 카드가 없을 것
2. 이벤트 수신
   - 경고가 1회 나타날 것
3. 확인 클릭
   - 같은 반복 이벤트가 2초마다 다시 알리지 않을 것
4. 새 이벤트 또는 식별자가 달라진 이벤트 수신
   - 새 경고가 다시 나타날 것
5. 전체화면
   - 카메라 카드 버튼으로 진입/종료 가능할 것

## 현재 세션 블로커 근거

- 이 Codex 세션은 `strange_front`를 읽을 수는 있지만 쓸 수는 없다.
- shell write probe는 실행 전 정책에서 차단된다.
- `apply_patch`는 현재 writable workspace에만 직접 적용 가능해서, 이 세션에서는 `strange_front`를 바로 패치할 수 없다.
