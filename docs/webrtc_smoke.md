# WebRTC WHEP Smoke Test

This is the first gate before AI, backend, frontend, Docker, or CI/CD integration.

Required path:

```text
sample video or RTSP input
-> MediaMTX
-> WebRTC WHEP
-> browser video playback
```

Success criteria:

| Check | Required value |
| --- | --- |
| WHEP POST | HTTP 2xx |
| ICE `connectionState` | `connected` or `completed` |
| `ontrack` | fired |
| `bytesReceived` | increases |
| `framesDecoded` | increases |
| video event | `playing` fired |

## Path Standard: `cameraLoginId`

> **Important**: The project's camera path standard is `cameraLoginId` as registered in the backend DB.
> All RTSP publish paths, HLS URLs, WebRTC WHEP URLs, and AI runner inputs must use the **same** `cameraLoginId`.
>
> The `cam1` path (without underscore) was used as a **temporary smoke path** during initial WebRTC validation.
> It is **not** the operational standard. Do not use `cam1` in production or registered-camera flows.

| Layer | Operational standard |
| --- | --- |
| RTSP publish | `rtsp://<host>:8554/cam_01` |
| HLS | `http://<host>:8888/cam_01/index.m3u8` |
| WebRTC WHEP | `http://<host>:8889/cam_01/whep` |
| AI runner input | `rtsp://<host>:8554/cam_01` |
| AI Overlay | `:8010` (cam_01), `:8011` (cam_02), `:8012` (cam_03), `:8013` (cam_04) |

## GPU PC Manual Smoke

On the GPU PC:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
bash scripts/run_rtsp_server.sh
```

The default `stream/mediamtx.yml` is tuned for same-host or SSH-tunnel playback because its ICE candidate is `127.0.0.1`.

For direct browser access to the GPU PC IP, start MediaMTX with:

```bash
MEDIAMTX_CONFIG=stream/mediamtx.webrtc.direct.yml bash scripts/run_rtsp_server.sh
```

Then use `http://<GPU_PC_IP>:8889/cam_01/whep` as the WHEP URL and make sure inbound TCP `8889` and `8189` are open.

Publish a sample video to MediaMTX:

```bash
ffmpeg -re -stream_loop -1 \
  -i /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos/101-2_cam01_swoon01_place03_day_winter.mp4 \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/cam_01
```

Open the smoke page from the same machine or through SSH forwarding:

```text
benchmark/webrtc_whep_smoke.html?url=http://localhost:8889/cam_01/whep
```

The page exposes the machine-readable result at:

```js
window.webrtcSmokeResult
```

## Windows SSH Tunnel

Forward WebRTC HTTP and ICE TCP ports in addition to HLS and overlay ports:

```powershell
ssh -N `
  -L 8888:127.0.0.1:8888 `
  -L 8889:127.0.0.1:8889 `
  -L 8189:127.0.0.1:8189 `
  -L 8010:127.0.0.1:8010 `
  -L 8011:127.0.0.1:8011 `
  -L 8012:127.0.0.1:8012 `
  -L 8013:127.0.0.1:8013 `
  -R 18080:127.0.0.1:18080 `
  welabs@58.127.241.84
```

Then open:

```text
http://127.0.0.1:8099/benchmark/webrtc_whep_smoke.html?url=http%3A%2F%2Flocalhost%3A8889%2Fcam_01%2Fwhep
```

### Historical Note (cam1 temporary path)

The initial WebRTC smoke validation was run against the `cam1` path (no underscore), which was published
by the then-running ffmpeg session on the GPU PC. The passing result was:

- WHEP URL: `http://localhost:8889/cam1/whep`
- WHEP POST: 201
- ICE connected, ontrack fired, video playing
- bytesReceived=396948, framesDecoded=44

This validated the MediaMTX/WHEP/browser pipeline end-to-end.
The path has since been aligned to the project standard `cam_01`.

## Playwright Smoke

Install Playwright in the environment where the browser can reach MediaMTX:

```bash
npm install --no-save @playwright/test
npx playwright install chromium
```

Run (standard `cam_01` path):

```bash
WHEP_URL=http://localhost:8889/cam_01/whep \
npx playwright test -c benchmark/playwright.whep.config.mjs --reporter=line
```

On Windows PowerShell:

```powershell
$env:WHEP_URL="http://localhost:8889/cam_01/whep"
npx.cmd playwright test -c benchmark\playwright.whep.config.mjs --reporter=line
```

The `WHEP_STREAM_PATH` env var can also be used to override just the path portion:

```powershell
$env:WHEP_STREAM_PATH="cam_01"
npx.cmd playwright test -c benchmark\playwright.whep.config.mjs --reporter=line
```

Default values in `webrtc_whep_smoke.spec.mjs`:

```js
const streamPath = process.env.WHEP_STREAM_PATH || 'cam_01';
const whepUrl = process.env.WHEP_URL || `http://localhost:8889/${streamPath}/whep`;
```

## Docker/Compose Gate

After bare GPU PC playback passes, repeat the same smoke with MediaMTX in Docker/Compose. Do not move to React player integration until the same six checks pass through the container network and published ports.

> **Port conflict warning**: The existing `ai-tunnel` container (`docker-compose.tunnel.yml`) occupies ports
> `8888`, `8889`, and `8189`. Before running the Docker smoke, either:
>
> 1. Stop the tunnel first: `docker compose -f docker-compose.tunnel.yml down`
> 2. Or run the smoke on alternate ports and adjust `WHEP_URL` accordingly.
>
> Confirm with: `netstat -ano | findstr ":8889"` (Windows) or `ss -tlnp | grep 8889` (Linux)

Start MediaMTX:

```bash
docker compose -f docker-compose.webrtc-smoke.yml up -d
```

Publish to it:

```bash
ffmpeg -re -stream_loop -1 \
  -i /path/to/sample.mp4 \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/cam_01
```

Run the same browser smoke against:

```text
http://localhost:8889/cam_01/whep
```

Both HLS and WebRTC WHEP must pass for the same path:

```bash
# HLS check
curl -I http://localhost:8888/cam_01/index.m3u8

# WebRTC WHEP check (via Playwright)
$env:WHEP_URL="http://localhost:8889/cam_01/whep"
npx.cmd playwright test -c benchmark\playwright.whep.config.mjs --reporter=line
```

## Next Gates

1. Connect React/Vite WebRTC player to the passing WHEP URL.
2. Add AI inference and MQTT `safety/events`.
3. Confirm Spring Boot subscribes, persists, and broadcasts over WebSocket.
4. Confirm frontend alert appears while the WebRTC video keeps playing.
5. Only then apply GitHub Actions image build/test/deploy automation.
