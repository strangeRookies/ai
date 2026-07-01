# Frame Sync Debug

AI overlay/event payloads now carry per-camera frame synchronization metadata.

## Latest-Frame Queue Evidence Rule

RTSP readers can receive frames faster than YOLO Pose plus tracking plus LSTM can process them. The AI worker therefore uses a bounded latest-frame queue:

- `CameraFrameQueue.put_latest()` drops the oldest queued frame when the queue is full.
- `CameraFrameQueue.get_latest()` keeps only the newest queued frame and drops stale queued frames before inference.
- `droppedFrameCount` is cumulative per worker queue and is diagnostic only.

Because of that, `frameId` must not mean "the last frame the reader saw". It must mean "the `FramePacket` that actually entered AI inference". Evidence is anchored to the processed packet and its metadata:

```text
evidenceId = {cameraLoginId}-{frameId}-{capturedAtMs}
traceId    = evidenceId
```

Overlay payloads, confirmed event payloads, frame_sync payloads, event clip metadata, and future snapshot metadata should all use this same evidence key. `timestampMs` for evidence identity is `capturedAtMs`; `publishedAtMs` is the publish time and must not mint a second evidence key.

## Contract

- `frameId` is a camera-local increasing counter.
- `capturedAtMs` is recorded immediately after a frame is read.
- `processedAtMs` is recorded after detection/tracking/LSTM processing.
- `publishedAtMs` is recorded immediately before MQTT payload creation.
- `aiLatencyMs = processedAtMs - capturedAtMs`.
- `publishLatencyMs = publishedAtMs - capturedAtMs`.
- `latencyOrderValid` is false when `capturedAtMs <= processedAtMs <= publishedAtMs` is violated.
- `evidenceId`, `traceId`, and nested `evidence` fields are additive and backward compatible.

LSTM events also include a `sequence` object:

- `sequenceLength`
- `sequenceStride`
- `sequenceStartFrameId`
- `sequenceEndFrameId`
- `sequenceStartAtMs`
- `sequenceEndAtMs`

Existing payload fields are preserved. The schema version is now `1.1`.

## Runtime

Enable local MJPEG frame sync overlay:

```bash
python scripts/serve_ai_overlay.py \
  --rtsp-url rtsp://127.0.0.1:8554/cam_04 \
  --camera-id cam_04 \
  --camera-login-id cam_04 \
  --mjpeg-debug \
  --frame-sync-debug
```

The debug panel shows:

- frame id
- captured timestamp in epoch milliseconds
- AI latency
- publish latency
- sequence length / stride

## Logs

Periodic frame synchronization logs:

```text
[frame-sync] cam_04 frame_id=120 captured_at_ms=1782180000123 ai_latency_ms=55 publish_latency_ms=62 dropped_frame_count=3 evidence_id=cam_04-120-1782180000123 buffer_size=60
```

Invalid timestamp order logs:

```text
[frame-sync] warning cam_04 latency_order_invalid=true latency_order_valid=false frame_id=120 captured_at_ms=1782180000123 processed_at_ms=1782180000110 published_at_ms=1782180000130
```

LSTM event sequence logs:

```text
[lstm-event] cam_04 sequence=91-120 length=30 stride=15 event=Faint prob=0.87
```

## Smoke Checks

1. Start the overlay worker with `--frame-sync-debug`.
2. Confirm `frame_id` increases per `camera_login_id`.
3. Confirm overlay MQTT payload includes `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `aiLatencyMs`, and `publishLatencyMs`.
4. Confirm overlay, event, and frame_sync payloads share `evidenceId` for the same processed frame.
5. Trigger an LSTM event and confirm the event payload includes `sequenceStartFrameId` and `sequenceEndFrameId`.
6. Keep `FRAME_SYNC_BUFFER_SIZE` bounded. The default is `60`.
7. Tune delayed overlay warnings with `FRAME_SYNC_DELAY_WARNING_MS`. The default is `300`.

## Self-Improving AI Dependency

Operator FP/FN/TP/TN feedback should be linked by `evidenceId`, not by `frameId` alone. A dropped queue can skip frame ids, and clocks can drift across the browser, backend, and AI host. The stable feedback key is:

- `cameraLoginId`
- `frameId`
- `capturedAtMs`
- `evidenceId`

Only after those fields are stable can false positives become hard-negative candidates and false negatives become faint/fall reinforcement candidates without mixing evidence from a neighboring frame.
