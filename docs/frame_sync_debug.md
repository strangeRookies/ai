# Frame Sync Debug

AI overlay/event payloads now carry per-camera frame synchronization metadata.

## Contract

- `frameId` is a camera-local increasing counter.
- `capturedAtMs` is recorded immediately after a frame is read.
- `processedAtMs` is recorded after detection/tracking/LSTM processing.
- `publishedAtMs` is recorded immediately before MQTT payload creation.
- `aiLatencyMs = processedAtMs - capturedAtMs`.
- `publishLatencyMs = publishedAtMs - capturedAtMs`.

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
[frame-sync] cam_04 frame_id=120 captured_at_ms=1782180000123 ai_latency_ms=55 publish_latency_ms=62 buffer_size=60
```

LSTM event sequence logs:

```text
[lstm-event] cam_04 sequence=91-120 length=30 stride=15 event=Faint prob=0.87
```

## Smoke Checks

1. Start the overlay worker with `--frame-sync-debug`.
2. Confirm `frame_id` increases per `camera_login_id`.
3. Confirm overlay MQTT payload includes `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `aiLatencyMs`, and `publishLatencyMs`.
4. Trigger an LSTM event and confirm the event payload includes `sequenceStartFrameId` and `sequenceEndFrameId`.
5. Keep `FRAME_SYNC_BUFFER_SIZE` bounded. The default is `60`.
6. Tune delayed overlay warnings with `FRAME_SYNC_DELAY_WARNING_MS`. The default is `300`.
