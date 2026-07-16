# Queue Lag & MQTT E2E (Ubuntu / Bash)

GPU PC repo root 예: `/home/welabs/yolo_training/strange_ai_lstm` → 아래 `ROOT`는 `strange_ai` 루트.

Payload에는 이미 `capturedAtMs`, `processedAtMs`, `mqttPublishStartedAtMs`가 있으므로 **별도 E2E 스키마 추가 불필요**.

## 0. 사전 패치 (레포에 반영됨)

- `ai/runtime_metrics.py`: `add_queue_lag_ms`, summary에 `avg/p50/p95/max_queue_lag_ms`
- `scripts/run_rtsp_inference.py`: dequeue 시 queue lag·`max_queue_depth` 기록
- `scripts/measure_mqtt_alert_latency.py`: MQTT subscriber E2E 집계
- `scripts/run_queue_lag_and_mqtt_e2e.sh`: 2-cam AB + E2E 래퍼

검증:

```bash
cd "$ROOT"
source .venv/bin/activate
python -m py_compile ai/runtime_metrics.py scripts/run_rtsp_inference.py scripts/measure_mqtt_alert_latency.py
python -m unittest tests.test_runtime_metrics_queue_lag -q
```

## 1. Queue Lag 2-cam AB

```bash
export ROOT="$PWD"
export OUTPUT_DIR="$ROOT/runs/wiki_metrics/$(date +%Y%m%d_%H%M%S)/08_queue_latency"

CAMERA_IDS="cam_03,cam_04" \
MANAGE_STREAMS=never \
MAX_FRAMES=1800 \
FRAME_QUEUE_MAXSIZE=3 \
bash scripts/run_queue_lag_and_mqtt_e2e.sh queue
```

또는 직접:

```bash
CAMERA_IDS="cam_03,cam_04" MANAGE_STREAMS=never MAX_FRAMES=1800 FRAME_QUEUE_MAXSIZE=3 \
  bash scripts/run_2cam_rtsp_ab_metrics.sh
```

확인 지표: `p50/p95/max_queue_lag_ms`, `max_queue_depth`, `latest_dropped_frame_count`.

## 2. MQTT E2E (AI capture → subscriber)

```bash
export ROOT="$PWD"
export KILL_EXISTING_WORKERS=1   # 실험 전용일 때만
bash scripts/run_queue_lag_and_mqtt_e2e.sh e2e
```

결과: `$E2E_OUT/alert_latency_summary.json`

- `ai_latency`: captured → processed
- `processed_to_mqtt`: processed → publish start
- `mqtt_transport`: publish start → subscriber
- `e2e_subscriber`: captured → subscriber
- `within_1_second_rate`

## 3. 브라우저까지

프론트 WebSocket 수신 시 `Date.now() - capturedAtMs` 로그 추가 필요 (본 문서 범위 밖).