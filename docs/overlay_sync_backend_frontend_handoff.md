# Overlay Sync Backend/Frontend Handoff

This handoff is for the Backend and Frontend agents. The AI agent inspected these paths read-only and did not edit `strange_back/` or `strange_front/` because the workspace ownership rules reserve those repositories for their assigned agents.

## Current AI Evidence State

The AI runtime now emits additive sync/evidence fields on processed-frame payloads:

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
- nested `evidence`
- `droppedFrameCount`
- `latencyOrderValid`

The local AI-side raw payload can be reproduced with:

```powershell
cd strange_ai
uv run --with numpy python scripts/diagnose_overlay_payload_sync.py --camera-login-id qa_lobby_01
```

Expected classification from the AI-side diagnostic is:

```text
sync_fields_present
```

## Frontend Agent Handoff

### Goal

Capture raw STOMP payloads immediately after WebSocket receipt and before normalization. Then log the normalized object after parsing. This proves whether `frameId/capturedAtMs/publishedAtMs` disappear before or after frontend parsing.

### Files To Inspect/Edit

- `strange_front/src/shared/utils/stomp.ts`
- `strange_front/src/hooks/useAiEvents.ts`
- `strange_front/src/shared/utils/aiEventParsing.ts`
- `strange_front/src/features/dashboard/overlays/overlayTypes.ts`

### Recommended Minimal Change

In `SimpleStompClient.handleMessage()`, after `const body = ...` and before `JSON.parse(body)`, add a debug-only callback or logger guarded by an env flag such as `VITE_FRONT_OVERLAY_SYNC_DEBUG === 'true'`.

The log should include the raw body and should not run unless debug is enabled:

```text
[overlay-sync:raw-stomp] <raw JSON string>
```

Then in `useAiEvents.handleIncoming()`, after `parseToAiEvent(raw)`, log the normalized object under the same flag:

```text
[overlay-sync:normalized] camera=... frameId=... capturedAtMs=... processedAtMs=... publishedAtMs=...
```

### Acceptance Checks

- If raw STOMP has the sync fields but normalized log has `n/a`, fix `aiEventParsing.ts` or overlay parsing.
- If raw STOMP lacks the sync fields, classify the issue as upstream AI/backend relay.
- `npm run build` should pass.
- If available, `npm run test:overlay-sync` should pass.

## Backend Agent Handoff

### Goal

Preserve sync fields when backend emits empty/clear overlay messages, or mark those clear messages so frontend overlay-sync diagnostics do not treat them as processed-frame overlays.

### Files To Inspect/Edit

- `strange_back/src/main/java/com/strange/safety/camera/overlay/OverlayMessage.java`
- `strange_back/src/main/java/com/strange/safety/camera/overlay/OverlayRelayService.java`
- `strange_back/src/test/java/com/strange/safety/camera/overlay/OverlayRelayServiceTest.java`

### Observed Field Loss Point

`OverlayRelayService.clearSnapshot()` currently creates a new empty overlay with the short constructor:

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

That constructor sets `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `queueLagMs`, and `droppedFrameCount` to `null`. This can produce frontend logs like:

```text
frameId=n/a capturedAtMs=n/a publishedAtMs=n/a networkLatencyMs=n/a endToEndLatencyMs=n/a
```

### Recommended Minimal Change

Use the full constructor when creating the clear overlay and copy the sync fields from the current message:

```java
new OverlayMessage(
    schemaVersion,
    MESSAGE_TYPE,
    timestampMs,
    current.message().streamId(),
    current.message().cameraLoginId(),
    current.message().frameWidth(),
    current.message().frameHeight(),
    List.of(),
    current.message().frameId(),
    current.message().capturedAtMs(),
    current.message().processedAtMs(),
    current.message().publishedAtMs(),
    current.message().queueLagMs(),
    current.message().droppedFrameCount()
)
```

Alternative: add an explicit clear flag and have the frontend suppress frame evidence diagnostics for clear overlays. Copying the fields is the smallest compatibility-preserving change.

### Acceptance Checks

- Add or update `OverlayRelayServiceTest` to assert a cleared overlay preserves `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `queueLagMs`, and `droppedFrameCount`.
- `./gradlew test` should pass.
- Existing MQTT/STOMP topics must not change.

## End-To-End Classification Matrix

| Raw STOMP | Normalized Frontend | Likely Owner |
| --- | --- | --- |
| missing sync fields | missing sync fields | AI/backend relay |
| has sync fields | missing sync fields | Frontend parser/normalizer |
| has sync fields | has sync fields but clear overlay logs n/a | Backend clear overlay or frontend clear handling |
| has sync fields | has sync fields and bufferSize stays 1 | Publish rate, pruning window, or stream/test setup |

## Final Expected Log

```text
[overlay-sync] camera=qa_lobby_01 frameId=4 capturedAtMs=1782180000100 publishedAtMs=1782180000123 receivedAtMs=1782180000158 networkLatencyMs=35 endToEndLatencyMs=58 selectedOverlayAgeMs=120 overlayTimestampDeltaMs=45 bufferSize=2
```
