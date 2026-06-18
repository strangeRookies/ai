# RTSP to Web Streaming

The frontend does not use RTSP URLs directly. Keep RTSP on the AI/GPU PC, then expose browser-safe streams to the dashboard.

## Path Standard: `cameraLoginId`

All camera path identifiers are based on `cameraLoginId` as registered in the backend DB.
Do **not** use bare `cam1`, `cam2` paths — use `cam_01`, `cam_02`, etc.

| Layer | URL pattern |
| --- | --- |
| RTSP publish | `rtsp://<host>:8554/cam_01` |
| HLS | `http://<host>:8888/cam_01/index.m3u8` |
| WebRTC WHEP | `http://<host>:8889/cam_01/whep` |
| AI Overlay (MJPEG) | `http://<host>:8010` (cam_01), `:8011` (cam_02), … |
| AI runner input | same RTSP path as publish |

## Architecture

```text
Camera or video file
  -> MediaMTX RTSP server :8554 (path = cameraLoginId, e.g. cam_01)
  -> HLS :8888/{cameraLoginId}/index.m3u8
  -> WebRTC WHEP :8889/{cameraLoginId}/whep
  -> AI overlay MJPEG :8010~8013 (per camera port)
  -> frontend video player / canvas
```

`serve_mjpeg.py` does not shell out to ffmpeg. It reads RTSP through OpenCV, which commonly uses FFmpeg internally. FFmpeg is used here to publish test webcams or recorded videos into the RTSP server.

## 1. Start RTSP Server

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
chmod +x scripts/*.sh
./scripts/run_rtsp_server.sh
```

This serves any publisher path. Active camera paths are registered in the backend DB as `cameraLoginId`.

## 2. Publish Test Video or Webcam

Recorded video (publish to `cam_01`):

```bash
ffmpeg -re -stream_loop -1 \
  -i /path/to/sample.mp4 \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/cam_01
```

Linux webcam:

```bash
ls /dev/video*
ffmpeg -f v4l2 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://localhost:8554/cam_01
```

Windows webcam to GPU PC:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -f dshow -i video="YOUR WEBCAM NAME" -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://GPU_PC_IP:8554/cam_01
```

## 3. Start Web Stream Bridge (MJPEG legacy mode)

In another terminal:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate
./scripts/run_stream_bridge.sh
```

Health check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/cameras
```

Open in a browser:

```text
http://GPU_PC_IP:8000/stream/camera-1
```

## Real Cameras

Configure real RTSP URLs only on the AI server. Paths must match cameraLoginId:

```bash
export CAMERA_1_RTSP_URL='rtsp://localhost:8554/cam_01'
export CAMERA_2_RTSP_URL='rtsp://localhost:8554/cam_02'
export CAMERA_3_RTSP_URL='rtsp://192.168.0.10:8554/cam_03'
export CAMERA_4_RTSP_URL='rtsp://192.168.0.11:8554/cam_04'
./scripts/run_stream_bridge.sh
```

On the frontend PC, set:

```text
VITE_STREAM_BASE_URL=http://GPU_PC_IP:8000
```

## Troubleshooting

If `rtsp://localhost:8554` says connection refused, the RTSP server is not running.

If `No route to host` appears for `192.168.x.x`, the IP is wrong, unreachable, or blocked by firewall.

If Linux blocks ports:

```bash
sudo ufw allow 8554/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 8888/tcp
sudo ufw allow 8889/tcp
sudo ufw allow 8189/tcp
```

Do not put RTSP usernames or passwords in frontend code.
