# AI Docker Deployment

## Analyzed Runtime Flow

The Docker entrypoint is `scripts/run_registered_cameras.py`.

1. `scripts/run_registered_cameras.py` polls `BACKEND_BASE_URL/api/cameras/active`.
2. `ai/registered_cameras.py` parses backend camera records into `RegisteredCamera`. `cameraLoginId` is used as the external camera key for dynamic IDs such as `cam_01`.
3. `REAL_RTSP` cameras use the backend-provided `rtspUrl`.
4. `SIMULATED_RTSP` cameras use `MEDIAMTX_RTSP_BASE_URL/<cameraLoginId>` and can stream a local video file to MediaMTX through ffmpeg.
5. `ai/registered_camera_workers.py` starts one `scripts/serve_ai_overlay.py` worker per camera.
6. Each worker reads RTSP frames, runs YOLO pose detection, tracking, per-track keypoint buffering, and the LSTM action classifier.
7. Faint events are published through `ai/publishers/event_publisher.py` to the default MQTT topic `safety/events`.
8. Camera status events use the same MQTT connection path through `ai/publishers/camera_status_publisher.py`.

## Local Python

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Dry-run the registered camera runner without probing RTSP streams:

```bash
python scripts/run_registered_cameras.py --dry-run --skip-rtsp-probe
```

Example with explicit runtime settings:

```bash
python scripts/run_registered_cameras.py \
  --backend-base-url http://localhost:8080 \
  --rtsp-base-url rtsp://localhost:8554 \
  --mqtt-host localhost \
  --mqtt-topic safety/events \
  --yolo-model ./models/yolo26n-pose.pt \
  --action-model ./models/lstm.pt
```

## Docker Build And Run

Build:

```bash
docker build -t strange-ai:local .
```

Run:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/sample_videos:/data/videos:ro" \
  -v strange_ai_runs:/app/runs \
  strange-ai:local
```

PowerShell dry-run example:

```powershell
docker run --rm `
  -e BACKEND_BASE_URL=http://host.docker.internal:8080 `
  -e MEDIAMTX_RTSP_BASE_URL=rtsp://host.docker.internal:8554 `
  -e MQTT_HOST=host.docker.internal `
  -e YOLO_MODEL_PATH=/models/yolo26n-pose.pt `
  -e MODEL_CHECKPOINT_PATH=/models/lstm.pt `
  -v ${PWD}/models:/models:ro `
  -v ${PWD}/sample_videos:/data/videos:ro `
  -v strange_ai_runs:/app/runs `
  strange-ai:local python scripts/run_registered_cameras.py --dry-run --skip-rtsp-probe
```

## Docker Compose

Create a local `.env` from the checked-in example, then start the AI service:

```bash
cp .env.example .env
docker compose -f docker-compose.ai.yml up --build
```

The default compose values use `host.docker.internal` so the container can reach backend, MQTT, and MediaMTX services running on the host. If all services run in one compose network, use service names instead, for example:

```dotenv
BACKEND_BASE_URL=http://strange-back:8080
MQTT_HOST=mqtt
MEDIAMTX_RTSP_BASE_URL=rtsp://mediamtx:8554
```

## GPU

Install NVIDIA Container Toolkit on the Docker host, then run:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -e DEVICE=0 \
  -e ACTION_DEVICE=0 \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/sample_videos:/data/videos:ro" \
  -v strange_ai_runs:/app/runs \
  strange-ai:local
```

`docker-compose.ai.yml` includes a commented `gpus: all` example. For CPU fallback, set `DEVICE=cpu` and `ACTION_DEVICE=cpu`, or keep the default `auto`.

## Environment Variables

| Name | Default | Purpose |
| --- | --- | --- |
| `BACKEND_BASE_URL` | `http://host.docker.internal:8080` | Backend API base URL for `/api/cameras/active` |
| `BACKEND_TOKEN` | empty | Optional backend bearer token |
| `BACKEND_TIMEOUT_SECONDS` | `10` | Backend request timeout |
| `MQTT_HOST` | `host.docker.internal` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `safety/events` | AI event publish topic |
| `MQTT_CLIENT_ID_PREFIX` | `strange-ai` | MQTT client ID prefix per camera |
| `MEDIAMTX_RTSP_BASE_URL` | `rtsp://host.docker.internal:8554` | RTSP base URL for simulated cameras |
| `YOLO_MODEL_PATH` | `/models/yolo26n-pose.pt` | YOLO pose model path |
| `MODEL_CHECKPOINT_PATH` | `/models/lstm.pt` | LSTM checkpoint path |
| `SEQUENCE_LENGTH` | `8` | Per-track sequence length |
| `SEQUENCE_STRIDE` | `4` | Sequence stride |
| `CAMERA_POLL_INTERVAL_SECONDS` | `30` | Active camera refresh interval |
| `DEVICE` | `auto` | YOLO device: `auto`, `cpu`, `0`, or `cuda:0` |
| `ACTION_DEVICE` | `auto` | LSTM device |
| `VIDEO_POOL_DIR` | `video_pool` | Local video pool for `SIMULATED_RTSP` |
| `EVENT_CLIP_OUTPUT_DIR` | `clips` | Event clip output directory |
| `LOG_LEVEL` | `INFO` | Runtime logging level |

## Volumes

| Container path | Purpose |
| --- | --- |
| `/models` | Inject YOLO and LSTM model files |
| `/data/videos` | Inject simulated RTSP sample videos |
| `/app/runs` | Persist worker logs, event clips, and runtime outputs |

Model files, sample videos, logs, generated runs, datasets, and raw data are excluded from the image and should be supplied through volumes or environment variables. The compose file uses the named volume `strange_ai_runs` for `/app/runs` so the non-root container user can write logs consistently on Linux hosts.

## Connection Checks

Backend:

```bash
curl http://localhost:8080/api/cameras/active
```

MQTT:

```bash
python scripts/publish_test_mqtt_event.py
python scripts/subscribe_mqtt_events.py
```

MediaMTX/RTSP:

```bash
ffprobe rtsp://localhost:8554/cam_01
```

AI preflight:

```bash
python scripts/run_rtsp_inference.py --preflight-only --detector-mode mock --dry-run --action-model=
```

## Troubleshooting

RTSP connection failed: Check `MEDIAMTX_RTSP_BASE_URL` and backend `rtspUrl` values. From Docker, a host MediaMTX instance usually needs `rtsp://host.docker.internal:8554`.

MQTT connection failed: Check `MQTT_HOST`, `MQTT_PORT`, broker listener configuration, and firewall rules. If container logs show an MQTT connection error, verify the broker address first.

Model path error: Confirm `YOLO_MODEL_PATH` and `MODEL_CHECKPOINT_PATH` are paths inside the container. The usual setup mounts host `./models` to container `/models`.

CUDA unavailable: Confirm `--gpus all` or compose `gpus: all` is enabled and NVIDIA Container Toolkit is installed. Otherwise run with `DEVICE=cpu` and `ACTION_DEVICE=cpu`.

OpenCV `libGL` error: The Dockerfile installs `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, and `libxrender1`. For local Python, install equivalent OS packages or use a headless OpenCV environment.

Backend API connection failed: Inside a container, `localhost` points to the container itself. Use `http://host.docker.internal:8080` for a host backend, or `http://<service-name>:8080` for another compose service.
