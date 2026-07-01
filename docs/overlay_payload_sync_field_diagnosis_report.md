# Overlay Sync 필드 누락 진단 보고서

## 1. 요약

브라우저 콘솔의 `overlay-sync` 로그에서 아래처럼 프레임 동기화 필드가 `n/a`로 표시되는 문제가 있었다.

```text
frameId=n/a capturedAtMs=n/a publishedAtMs=n/a networkLatencyMs=n/a endToEndLatencyMs=n/a bufferSize=1
```

이 문제는 `overlaySyncDelayMs`나 `requestVideoFrameCallback`만 조정해서 해결할 수 없다. 먼저 AI → MQTT → Backend → STOMP/WebSocket → Frontend parser 전체 경로에서 `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`가 어디서 사라지는지 확인해야 한다.

이번 AI 작업에서는 `strange_ai/` 범위 안에서 다음을 조치했다.

- AI가 실제 추론한 프레임 기준으로 evidence chain을 고정했다.
- overlay/event/frame_sync payload에 frame sync 필드를 additive 방식으로 추가했다.
- raw AI payload 샘플을 재현하는 진단 스크립트를 추가했다.
- backend/frontend 담당자가 이어서 확인할 handoff 문서를 작성했다.

백엔드/프론트 파일은 권한상 읽기 전용으로만 확인했다. 후속 작업 상세는 `docs/overlay_sync_backend_frontend_handoff.md`에 정리했다.

## 2. 현재 결론

AI 쪽 payload builder 기준으로는 이제 raw overlay/event/frame_sync payload에 필요한 필드가 들어간다.

필수 필드:

- `cameraLoginId`
- `frameId`
- `timestampMs`
- `capturedAtMs`
- `processedAtMs`
- `publishedAtMs`
- `frameWidth`
- `frameHeight`
- `evidenceId`
- `traceId`
- `evidence.droppedFrameCount`
- `evidence.latency.aiLatencyMs`
- `evidence.latency.publishLatencyMs`
- `evidence.latencyOrderValid`

따라서 다음 단계의 판단 기준은 단순하다.

| 확인 지점 | 결과 | 담당 원인 |
| --- | --- | --- |
| raw STOMP payload에 필드 없음 | 프론트 도달 전 이미 누락 | AI publish 또는 Backend relay |
| raw STOMP payload에는 있음 | normalized object에서 누락 | Frontend parser/normalizer |
| 일반 overlay에는 있음 | clear/empty overlay에서만 누락 | Backend clear overlay 생성 또는 Frontend clear 처리 |
| 필드는 모두 있음 | bufferSize가 계속 1 | 수신 빈도, pruning, 테스트 조건, 실제 AI publish rate |

## 3. RTSP latest-frame buffer와 evidence 기준

현재 AI는 실시간성을 위해 latest-frame queue를 사용한다.

- RTSP reader는 프레임을 계속 읽는다.
- AI 추론이 늦으면 queue에 쌓인 오래된 프레임은 버려진다.
- `CameraFrameQueue.get_latest()`는 최신 프레임 1장을 AI 추론에 넘긴다.
- 이 과정에서 `frameId`가 건너뛰는 것은 정상이다.

따라서 evidence는 "reader가 마지막으로 읽은 프레임"이 아니라 "YOLO Pose + LSTM이 실제 처리한 프레임"에 묶여야 한다.

현재 evidence key:

```text
evidenceId = {cameraLoginId}-{frameId}-{capturedAtMs}
traceId    = evidenceId
```

이 key를 overlay payload, event payload, frame_sync payload, clip metadata에 동일하게 붙인다.

## 4. AI에서 수정한 내용

### 4.1 payload 필드 보강

수정 위치:

- `ai/publishers/mqtt_payloads.py`
- `ai/publishers/evidence_fields.py`
- `ai/inference/rtsp_runtime.py`
- `scripts/serve_ai_overlay.py`
- `scripts/run_rtsp_inference.py`

AI payload는 기존 schema/topic을 깨지 않고 optional/additive 필드만 추가한다.

예상 raw overlay payload:

```json
{
  "schemaVersion": "1.1",
  "messageType": "overlay",
  "timestampMs": 1782180000123,
  "streamId": "qa_lobby_01",
  "cameraLoginId": "qa_lobby_01",
  "frameWidth": 640,
  "frameHeight": 360,
  "events": [],
  "frameId": 4,
  "capturedAtMs": 1782180000100,
  "processedAtMs": 1782180000120,
  "publishedAtMs": 1782180000123,
  "aiLatencyMs": 20,
  "publishLatencyMs": 23,
  "evidenceId": "qa_lobby_01-4-1782180000100",
  "traceId": "qa_lobby_01-4-1782180000100"
}
```

### 4.2 latency 순서 검증

정상 순서:

```text
capturedAtMs <= processedAtMs <= publishedAtMs
```

순서가 깨지면 payload에는 아래 값이 들어간다.

```json
{"latencyOrderValid": false}
```

overlay worker 로그에도 warning을 남긴다.

```text
[frame-sync] warning camera latency_order_invalid=true latency_order_valid=false ...
```

### 4.3 metadata evict fallback

`FrameMetadataBuffer`에서 metadata가 evict되더라도, 실제 처리 중인 `FramePacket`에 `cameraLoginId`, `frameId`, `capturedAtMs`가 남아 있으면 그 값으로 evidence를 만든다.

중요한 점:

- metadata가 없다고 `get_latest()`로 최신 metadata를 가져오지 않는다.
- 최신 metadata로 보정하면 다른 프레임에 feedback/evidence가 붙을 수 있다.
- fallback도 processed packet 기준으로만 만든다.

## 5. AI-side 진단 스크립트

실제 STOMP를 띄우지 않아도 AI payload builder가 어떤 raw payload를 만드는지 재현할 수 있도록 스크립트를 추가했다.

실행:

```powershell
cd strange_ai
uv run --with numpy python scripts/diagnose_overlay_payload_sync.py --camera-login-id qa_lobby_01
```

출력 항목:

- `rawOverlay`
- `rawEvent`
- `rawFrameSync`
- `expectedNormalizedOverlay`
- `classification`
- `expectedOverlaySyncLog`

정상 classification:

```text
sync_fields_present
```

classification 의미:

- `sync_fields_present`: AI raw payload와 expected normalized payload 모두 필수 sync 필드를 가진다.
- `upstream_raw_payload_missing_fields`: raw payload에서 필수 sync 필드가 빠졌다.
- `frontend_normalization_missing_fields`: raw payload에는 있는데 normalized object에서 빠졌다.

## 6. 프론트 읽기 결과

읽은 파일:

- `strange_front/src/shared/utils/stomp.ts`
- `strange_front/src/hooks/useAiEvents.ts`
- `strange_front/src/shared/utils/aiEventParsing.ts`
- `strange_front/src/features/dashboard/overlays/overlayTypes.ts`

확인한 내용:

- `aiEventParsing.ts`는 이미 `frameId/frame_id`, `timestampMs/timestamp`, `capturedAtMs/captured_at_ms`, `processedAtMs/processed_at_ms`, `publishedAtMs/published_at_ms` alias를 처리한다.
- `overlayTypes.ts`도 overlay message의 `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `queueLagMs`, `droppedFrameCount`를 읽는다.
- `useAiEvents.ts`의 `overlay-sync` 로그는 normalized event object에 값이 없을 때 `n/a`를 출력한다.

따라서 프론트에서 반드시 추가 확인해야 할 것은 두 가지다.

1. `SimpleStompClient.handleMessage()`에서 JSON parse 직전 raw `body`를 debug log로 찍기
2. `parseToAiEvent(raw)` 직후 normalized object를 debug log로 찍기

권장 로그:

```text
[overlay-sync:raw-stomp] {...}
[overlay-sync:normalized] camera=... frameId=... capturedAtMs=... processedAtMs=... publishedAtMs=...
```

## 7. 백엔드 읽기 결과

읽은 파일:

- `strange_back/src/main/java/com/strange/safety/camera/overlay/OverlayMessage.java`
- `strange_back/src/main/java/com/strange/safety/camera/overlay/OverlayRelayService.java`

확인한 내용:

- `OverlayMessage.java`는 이미 `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `queueLagMs`, `droppedFrameCount` 필드를 가진다.
- `OverlayRelayService`는 `frame_sync` message를 즉시 broadcast한다.
- 다만 `OverlayRelayService.clearSnapshot()`에서 빈 overlay를 만들 때 short constructor를 사용한다.

현재 clear overlay 생성 방식:

```java
new OverlayMessage(
    schemaVersion,
    MESSAGE_TYPE,
    timestampMs,
    streamId,
    cameraLoginId,
    frameWidth,
    frameHeight,
    List.of()
)
```

이 생성자는 아래 필드를 `null`로 만든다.

- `frameId`
- `capturedAtMs`
- `processedAtMs`
- `publishedAtMs`
- `queueLagMs`
- `droppedFrameCount`

그래서 일반 overlay에는 sync field가 있어도, clear/empty overlay에서는 프론트 로그가 `n/a`로 나올 수 있다.

백엔드 권장 수정:

- `clearSnapshot()`에서 full constructor를 사용한다.
- 기존 latest overlay의 sync field를 clear overlay에도 복사한다.
- 또는 clear overlay에는 별도 clear flag를 붙이고 프론트에서 sync 진단 로그를 생략한다.

가장 작은 수정은 full constructor로 sync field를 복사하는 방식이다.

## 8. bufferSize=1 원인 후보

프론트 overlay buffer 자체가 1개만 저장하도록 고정된 것은 아니다.

읽기 기준 기본값:

- max age: 5초
- max size: 300
- match threshold: 200ms

따라서 `bufferSize=1`의 가능성 높은 원인은 다음이다.

- 실제로 최근 5초 안에 overlay payload가 1개만 들어오고 있다.
- backend clear/empty overlay가 latest overlay를 대체하면서 sync field를 잃고 있다.
- payload에 `capturedAtMs/timestampMs`가 없어 receive-time fallback이 발생하고 pruning된다.
- 보고 있는 로그가 dashboard overlay buffer가 아니라 alert/event feed 쪽 buffer일 수 있다.
- AI 추론 속도가 낮아 latest-frame queue가 stale frame을 많이 버리고 overlay publish 빈도가 낮다.

## 9. 기대되는 정상 로그

정상 payload가 프론트까지 도달하고 normalized object에 유지되면 로그는 아래처럼 나와야 한다.

```text
[overlay-sync] camera=qa_lobby_01 frameId=4 capturedAtMs=1782180000100 publishedAtMs=1782180000123 receivedAtMs=1782180000158 networkLatencyMs=35 endToEndLatencyMs=58 selectedOverlayAgeMs=120 overlayTimestampDeltaMs=45 bufferSize=2
```

핵심은 다음 값들이 더 이상 `n/a`가 아니어야 한다.

- `frameId`
- `capturedAtMs`
- `publishedAtMs`
- `networkLatencyMs`
- `endToEndLatencyMs`

## 10. Self-Improving AI와의 연결

FP/FN 피드백은 단순 `frameId`가 아니라 `evidenceId` 기준으로 저장해야 한다.

이유:

- RTSP latest-frame queue 때문에 frameId가 건너뛸 수 있다.
- browser/backend/AI host 간 clock drift가 있을 수 있다.
- operator feedback이 이웃 프레임에 잘못 붙으면 retraining data가 오염된다.

후속 Self-Improving AI에서 저장해야 할 기준 필드:

- `evidenceId`
- `cameraLoginId`
- `frameId`
- `capturedAtMs`
- `processedAtMs`
- `publishedAtMs`
- `bbox`
- `keypoints`
- `confidence`
- `latency`
- optional `snapshotPath`
- optional `clipPath`

## 11. 남은 작업

### Frontend Agent

- `stomp.ts`에서 raw STOMP body debug log 추가
- `useAiEvents.ts`에서 normalized payload debug log 추가
- raw에는 있는데 normalized에서 빠지는지 확인
- `npm run build`
- 가능하면 `npm run test:overlay-sync`

### Backend Agent

- `OverlayRelayService.clearSnapshot()`에서 clear overlay 생성 시 sync fields 보존
- `OverlayRelayServiceTest`에 clear overlay sync field 보존 테스트 추가
- `./gradlew test`

### Integration 확인

- 브라우저에서 `VITE_FRONT_OVERLAY_SYNC_DEBUG=true`로 실행
- raw STOMP payload와 normalized payload 비교
- 실제 overlay-sync 로그에서 `frameId/capturedAtMs/publishedAtMs`가 숫자로 나오는지 확인
- `bufferSize`가 계속 1이면 실제 overlay publish rate와 pruning 조건을 추가 확인

## 12. 이번 AI 작업 검증

추가한 테스트:

```powershell
uv run --with numpy python -m unittest discover -s tests -p "test_overlay_payload_sync_diagnosis.py"
```

결과:

```text
Ran 2 tests
OK
```

추가한 진단 스크립트:

```powershell
uv run --with numpy python scripts/diagnose_overlay_payload_sync.py --camera-login-id qa_lobby_01
```

결과 요약:

- `rawOverlay.frameId = 4`
- `rawOverlay.capturedAtMs = 1782180000100`
- `rawOverlay.publishedAtMs = 1782180000123`
- `rawOverlay.evidenceId = qa_lobby_01-4-1782180000100`
- `classification = sync_fields_present`
- expected log의 `bufferSize = 2`
