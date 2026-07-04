# MJPEG Demo Stability Strategy

## Decision

For the demo deadline, browser video display uses bounded MJPEG streams while AI decisions continue to come from MQTT metadata and event payloads.

The real-time AI path remains:

```text
YOLO Pose -> Supervision ByteTrack -> LSTM -> MQTT overlay/event metadata
```

MJPEG is a demo/debug viewing surface, not the source of truth for AI events.

## Runtime Settings

Recommended demo defaults:

```env
MJPEG_ENABLED=true
MJPEG_HOST=0.0.0.0
MJPEG_PORT=8010
MJPEG_BASE_PATH=/mjpeg
MJPEG_FPS=8
MJPEG_WIDTH=640
MJPEG_HEIGHT=360
MJPEG_JPEG_QUALITY=70
MJPEG_ENABLE_OVERLAY=false
```

The per-worker stream URL is:

```text
http://<host>:<port>/mjpeg/{cameraLoginId}
```

For registered cameras, the first worker starts at `MJPEG_PORT`, and later workers use the next available ports. For example:

```text
cam_01 -> http://localhost:8010/mjpeg/cam_01
cam_02 -> http://localhost:8011/mjpeg/cam_02
```

## Port Separation

| Port | Role |
| --- | --- |
| `18080` | Backend camera/event API |
| `1883` | MQTT |
| `8554` | RTSP |
| `8888` | HLS |
| `8889` | WebRTC/WHEP |
| `8010-8020` | AI MJPEG demo streams |

## Stability Rules

- MJPEG encoding is capped by FPS, width, height, and JPEG quality.
- Slow MJPEG clients only affect their own HTTP response loop.
- Client disconnects are treated as normal stream termination.
- Encoding failure skips the current frame and continues.
- `MJPEG_ENABLE_OVERLAY=false` avoids extra annotation drawing in the AI worker; frontend overlays can still use MQTT metadata.

## VLM Scope

VLM must stay off the per-frame real-time loop.

Recommended future settings:

```env
VLM_ENABLED=false
VLM_TRIGGER_CLASSES=faint,fall
VLM_COOLDOWN_SECONDS=30
VLM_MAX_IMAGES_PER_EVENT=3
VLM_TIMEOUT_SECONDS=10
```

VLM should run only after a YOLO/LSTM event and should read snapshots or short clips. Its output can be appended as optional fields such as:

- `vlmSummary`
- `vlmConfidence`
- `vlmReason`
- `vlmProcessedAtMs`

VLM timeout, API failure, VRAM pressure, or model failure must not block the original YOLO/LSTM event publish.

## WebRTC And GStreamer Follow-Up

WebRTC/GStreamer remains the better long-term direction for lower latency and timestamp-aware media handling. It is deferred because the deadline risk is higher than MJPEG.

Future work:

- GStreamer appsink reader for frame timestamps.
- RTCP Sender Report / RTP timestamp mapping.
- WebRTC output path once timestamp and reconnect behavior are stable.
- Replace OpenCV `VideoCapture` timestamps with an explicit `timestampSource`.

Current limitation:

```text
capturedAtMs = AI PC local receive time
```

It is not the original camera capture time.
