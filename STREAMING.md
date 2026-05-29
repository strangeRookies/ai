# RTSP to Web Streaming

The frontend does not use RTSP URLs directly. Keep RTSP on the AI/GPU PC, then expose browser-safe MJPEG URLs to the dashboard:

```text
http://localhost:8000/stream/camera-1
http://localhost:8000/stream/camera-2
http://localhost:8000/stream/camera-3
http://localhost:8000/stream/camera-4
```

## Architecture

```text
Camera or video file
  -> RTSP server, MediaMTX on :8554
  -> serve_mjpeg.py, OpenCV frame decoder and buffer on :8000
  -> frontend <img> MJPEG stream
```

`serve_mjpeg.py` does not shell out to ffmpeg. It reads RTSP through OpenCV, which commonly uses FFmpeg internally. FFmpeg is used here to publish test webcams or recorded videos into the RTSP server.

## 1. Start RTSP Server

```bash
cd /home/welabs/yolo_training/strange_ai
chmod +x scripts/*.sh
./scripts/run_rtsp_server.sh
```

This serves these RTSP paths:

```text
rtsp://localhost:8554/cam1
rtsp://localhost:8554/cam2
rtsp://localhost:8554/cam3
rtsp://localhost:8554/cam4
```

## 2. Publish Test Video or Webcam

Recorded video:

```bash
./scripts/publish_sample_video.sh sample_videos/example.mp4 cam1
```

Linux webcam:

```bash
ls /dev/video*
./scripts/publish_webcam_linux.sh /dev/video0 cam1
```

Windows webcam to GPU PC:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -f dshow -i video="YOUR WEBCAM NAME" -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://GPU_PC_IP:8554/cam1
```

## 3. Start Web Stream Bridge

In another terminal:

```bash
cd /home/welabs/yolo_training/strange_ai
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

Configure real RTSP URLs only on the AI server:

```bash
export CAMERA_1_RTSP_URL='rtsp://localhost:8554/cam1'
export CAMERA_2_RTSP_URL='rtsp://localhost:8554/cam2'
export CAMERA_3_RTSP_URL='rtsp://192.168.0.10:8554/cam3'
export CAMERA_4_RTSP_URL='rtsp://192.168.0.11:8554/cam4'
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
```

Do not put RTSP usernames or passwords in frontend code.

## CPU Inference Feasibility Check

Before moving inference into a backend-side CPU deployment, measure real latency/FPS with the same stream shape used by the dashboard.

Run against the current `cam1` RTSP stream:

```bash
cd /home/welabs/yolo_training/strange_ai
source .venv/bin/activate
python benchmark/cpu_inference_benchmark.py \
  --input rtsp://localhost:8554/cam1 \
  --model yolov8n-pose.pt \
  --device cpu \
  --imgsz 320 \
  --max-frames 300 \
  --target-fps-per-camera 5 \
  --camera-count 4
```

Run against a recorded file:

```bash
python benchmark/cpu_inference_benchmark.py \
  --input sample_videos/example.mp4 \
  --model yolov8n-pose.pt \
  --device cpu \
  --imgsz 320 \
  --max-frames 300 \
  --target-fps-per-camera 5 \
  --camera-count 4
```

Read the `recommendation` field:

```text
cpu_ok_for_configured_camera_count
  CPU is likely enough for the configured camera count and AI FPS.

cpu_ok_for_fewer_cameras_or_lower_fps
  CPU can work, but reduce camera count, AI FPS, or image size.

cpu_not_recommended_without_optimization
  Keep AI worker separate or optimize with ONNX/OpenVINO, lower FPS, or a smaller model.
```

This benchmark should guide whether `strange_ai` can run beside `strange_back` on a CPU server or should stay on a GPU/AI worker host.
