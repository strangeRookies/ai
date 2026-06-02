# Project Summary

## YOLO Pose Model Benchmark

This project includes a benchmarking pipeline for selecting a YOLO pose model for a smart safety monitoring system.

The benchmark compares these configured model groups on the same video inputs, image size, and device:

- YOLOv8n-pose (`yolo8n-pose.pt`, fallback `yolov8n-pose.pt`)
- YOLOv11n-pose (`yolo11n-pose.pt`)
- YOLO26n-pose (`yolo26n-pose.pt`)
- YOLOv8s-pose (`yolo8s-pose.pt`, fallback `yolov8s-pose.pt`)
- YOLOv11s-pose (`yolo11s-pose.pt`)
- YOLO26s-pose (`yolo26s-pose.pt`)

YOLO model names can vary by installed Ultralytics version. The benchmark tries each configured candidate in order. If a model cannot be loaded or downloaded, that model is marked as `FAILED` or `SKIPPED`, and the remaining models continue.

### Input Videos

Place benchmark videos in:

```bash
sample_videos/
```

Supported extensions are configured in:

```bash
configs/model_benchmark.yaml
```

If the folder is empty, the script prints a guidance message and writes skipped result reports.

### Run

```bash
python benchmark/benchmark_models.py --video-dir sample_videos --imgsz 640 --device 0
```

Use automatic device selection:

```bash
python benchmark/benchmark_models.py --video-dir sample_videos --imgsz 640 --device auto
```

If CUDA is unavailable, the benchmark falls back to CPU and records `device=cpu`.

### Results

The benchmark writes:

```bash
benchmark/results/model_benchmark.csv
benchmark/results/model_benchmark.md
```

Recorded metrics:

- `model_name`
- `video_name`
- `device`
- `imgsz`
- `total_frames`
- `processed_frames`
- `avg_fps`
- `avg_latency_ms`
- `p95_latency_ms`
- `gpu_memory_mb`
- `avg_person_confidence`
- `avg_keypoint_confidence`
- `keypoint_missing_rate`
- `fall_candidate_count`
- `error_or_status`

GPU memory is measured with `torch.cuda.max_memory_allocated()` when CUDA is used. The script performs warm-up inference before measured frames, does not save raw frames, and does not persist private or sensitive video data.

### Fall Rule

The optional fall candidate rule uses COCO pose shoulder and hip keypoints. A person is counted as a `FALL_DOWN` candidate when the shoulder-to-hip torso vector is much more horizontal than vertical and required keypoints are above the configured confidence threshold.

This is a simple screening rule for model comparison, not a final safety decision engine.

### Fall Candidate Diagnostics

If one model reports unexpected `fall_candidate_count` values, inspect the rule inputs on the same sampled frames:

```bash
python benchmark/diagnose_fall_candidates.py \
  --video sample_videos/full_demo.mp4 \
  --models YOLOv8s-pose,YOLOv11n-pose,YOLO26n-pose \
  --start-frame 7800 \
  --end-frame 8200 \
  --samples 10 \
  --imgsz 640 \
  --device 0
```

The diagnostic report writes:

```text
benchmark/results/fall_candidate_diagnostics/fall_candidate_diagnostics.csv
benchmark/results/fall_candidate_diagnostics/fall_candidate_diagnostics.json
benchmark/results/fall_candidate_diagnostics/overlays/<model>/frame_*.jpg
```

Each row includes bbox `xyxy`, bbox width/height ratio, keypoint coordinates, keypoint confidence, fall-rule torso values, final candidate decision, and a result-format report for checking pixel vs normalized coordinates.

### LSTM Pose Extractor Comparison

Do not eliminate YOLO26n-pose only because `fall_candidate_count` is zero. That value is a rule-based diagnostic. Compare pose models as downstream LSTM keypoint extractors instead:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --detector-mode real \
  --models YOLOv11n-pose:yolo11n-pose.pt,YOLO26n-pose:yolo26n-pose.pt,YOLOv8s-pose:yolov8s-pose.pt \
  --device 0 \
  --imgsz 640 \
  --max-rows-per-split 3 \
  --max-frames 300 \
  --epochs 1
```

For a preprocessing-only smoke test, add `--dry-run`. Results are written under:

```text
benchmark/results/lstm_extractor_comparison/
```

The comparison prioritizes Faint recall and stable sequence generation over temporary fall-candidate rule counts. It reports clips processed, person detections, keypoints extracted, generated sequences, zero-sequence clips, keypoint missing rate, fallback usage, LSTM accuracy, precision, recall, F1-score, and confusion matrix.

### Final LSTM Benchmark: YOLOv11n-pose vs YOLO26n-pose

The final LSTM extractor benchmark is now scoped to the two remaining pose backbones only:

- YOLOv11n-pose (`yolo11n-pose.pt`)
- YOLO26n-pose (`yolo26n-pose.pt`)

Older YOLOv8 and `s` variants are previous pose-model candidates and should not be rerun for the final LSTM comparison unless a specific regression check requires it.

Run the GPU-PC smoke test first:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --detector-mode real \
  --models YOLOv11n-pose:yolo11n-pose.pt,YOLO26n-pose:yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --output-dir benchmark/results/lstm_final_11n_vs_26n \
  --max-rows-per-split 1 \
  --max-frames 120 \
  --epochs 1
```

If the smoke test completes, run the full final benchmark:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --detector-mode real \
  --models YOLOv11n-pose:yolo11n-pose.pt,YOLO26n-pose:yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --output-dir benchmark/results/lstm_final_11n_vs_26n \
  --max-rows-per-split 30 \
  --max-frames 0 \
  --epochs 10
```

Required outputs are written under:

```text
benchmark/results/lstm_final_11n_vs_26n/
benchmark/results/lstm_final_11n_vs_26n/summary.csv
benchmark/results/lstm_final_11n_vs_26n/summary.json
benchmark/results/lstm_final_11n_vs_26n/report.md
benchmark/results/lstm_final_11n_vs_26n/<model>/summary.json
benchmark/results/lstm_final_11n_vs_26n/<model>/confusion_matrix.csv
benchmark/results/lstm_final_11n_vs_26n/<model>/history.json
benchmark/results/lstm_final_11n_vs_26n/<model>/best.pt
```

Interpret the final result in this order: Faint recall, F1-score, false alarm tendency from the confusion matrix, sequence stability, then runtime feasibility. The report separates pose-only context, sequence generation metrics, and LSTM classification metrics. The current local Codex environment verified the CLI/report workflow and unit tests, but the real smoke/full benchmark must run on the GPU PC because the local workspace does not contain `../ai_fall_experiments/data/metadata/metadata.csv` or the real YOLO/Torch runtime.

## Mock Edge AI MQTT Publisher

`mock_edge_ai.py` publishes random safety event JSON messages to the MQTT topic used by the local development pipeline.

```text
Python Edge AI -> MQTT Broker (Mosquitto) -> Spring Boot MQTT Subscriber -> WebSocket -> React Frontend
```

This script is a mock publisher for integration testing before OpenCV, YOLOv8-Pose, and RTSP inference are connected.

### MQTT Environment

Set these environment variables when you need values other than the defaults:

```text
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=safety/events
MQTT_CLIENT_ID=edge-ai-mock-001
PUBLISH_INTERVAL_SECONDS=3
```

### Install

```bash
pip install -r requirements.txt
```

### Run

Make sure Mosquitto MQTT Broker is running, then start the mock publisher:

```bash
python mock_edge_ai.py
```

The script publishes events to:

```text
safety/events
```

Each published event is also printed to the console.

### Verify MQTT Publish

In another terminal, subscribe to the MQTT topic:

```bash
mosquitto_sub -h localhost -p 1883 -t safety/events
```

If `mosquitto_sub` is not installed locally, use the Mosquitto Docker container:

```bash
docker exec -it strange-mosquitto mosquitto_sub -h localhost -p 1883 -t safety/events
```

Expected event shape:

```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-05-26T10:00:00Z",
  "severity": "HIGH",
  "message": "Fall detected from mock edge AI",
  "source": "edge-ai-mock"
}
```

### Scope

- This mock does not implement YOLO, OpenCV, or RTSP processing.
- It does not include real videos, personal data, API keys, or passwords.
- It can be replaced later by the real edge AI inference pipeline while keeping the MQTT topic contract.

## Edge AI Pipeline MVP

`main.py` provides the initial runtime shape for the real CCTV Edge AI server.

```text
CCTV / Sample Video
-> RTSP Stream
-> OpenCV RTSP Reader
-> Latest-frame Queue
-> YOLO Pose Detector or Mock Detector
-> Track Assigner
-> Per-track Sequence Buffer
-> Fall Rule State Machine
-> MQTT safety/events
-> Spring Boot Backend
```

### Current Structure

```text
main.py
config.py
stream/frame_queue.py
stream/rtsp_reader.py
detector/mock_detector.py
detector/yolo_pose_detector.py
tracking/simple_tracker.py
rules/fall_rule.py
rules/track_sequence.py
messaging/event_schema.py
messaging/mqtt_publisher.py
tests/test_fall_rule.py
tests/test_event_schema.py
```

The frame queue is intentionally small and drops old frames so inference latency does not grow when processing is slower than the RTSP input.

### Pipeline Environment

```text
RTSP_URL=rtsp://localhost:8554/cam01
CAMERA_ID=cam_01
DETECTOR_MODE=mock
YOLO_MODEL=yolov8n-pose.pt
YOLO_DEVICE=auto
FRAME_QUEUE_SIZE=2
ALLOW_MOCK_FALLBACK=true
FALL_MIN_DURATION_SECONDS=1.5
FALL_DEBOUNCE_SECONDS=10
FALL_CANDIDATE_THRESHOLD=0.7
FALL_DECISION_WINDOW=3
FALL_DECISION_REQUIRED=2
TRACK_IOU_THRESHOLD=0.3
TRACK_MAX_MISSING_SECONDS=2
SEQUENCE_LENGTH=30
SEQUENCE_MAX_TRACK_AGE_SECONDS=5
MAX_FRAMES=0
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=safety/events
MQTT_CLIENT_ID=edge-ai-001
```

### Run The Pipeline

Mock detector mode, no RTSP required:

```bash
python main.py --dry-run --once
```

Publish to MQTT with mock detector:

```bash
python main.py --once
```

Run with RTSP and YOLO pose model:

```bash
DETECTOR_MODE=yolo RTSP_URL=rtsp://localhost:8554/cam01 YOLO_MODEL=yolov8n-pose.pt python main.py
```

On Windows PowerShell:

```powershell
$env:DETECTOR_MODE="yolo"
$env:RTSP_URL="rtsp://localhost:8554/cam01"
$env:YOLO_MODEL="yolov8n-pose.pt"
python main.py
```

### MQTT Verification

Start Mosquitto in `strange_infra`, then subscribe:

```bash
mosquitto_sub -h localhost -p 1883 -t safety/events
```

Run the AI server:

```bash
python main.py --once
```

Expected event shape:

```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-05-26T10:00:00Z",
  "severity": "HIGH",
  "message": "쓰러짐 의심 상황이 감지되었습니다.",
  "source": "edge-ai",
  "track_id": 1,
  "metadata": {
    "bbox": [100, 150, 280, 390],
    "confidence": 0.87,
    "rule_score": 0.91,
    "pose_state": "LYING",
    "model_name": "yolov8n-pose"
  }
}
```

### Rule Engine Notes

The first stabilized rule is `fall_detected`. It combines bbox aspect ratio, pose-horizontal signal, detector confidence, and a minimum duration threshold before emitting an event. Repeated events for the same track are debounced.

The runtime now follows a track-aware decision flow:

```text
Detection
-> SimpleTrackAssigner
-> PerTrackSequenceBuffer
-> FallRuleEngine
-> Event schema v1.0
```

`SimpleTrackAssigner` is a lightweight IoU-based adapter so detections without native tracker IDs still get stable-enough `track_id` values in local testing. It is intentionally isolated behind `tracking/simple_tracker.py` so ByteTrack can replace it later without changing the event rule or MQTT schema.

`FallRuleEngine` confirms an event only after the candidate score passes the threshold in at least `FALL_DECISION_REQUIRED` of the last `FALL_DECISION_WINDOW` observations and remains active for `FALL_MIN_DURATION_SECONDS`. This keeps the diagram's "2 out of recent 3" decision rule while preserving duration and cooldown safeguards.

TODO:

- Replace `SimpleTrackAssigner` with ByteTrack or another production tracker for stable `track_id` across crowded real streams.
- Expand rule modules for unconscious, bed fall, unauthorized exit, and violence detection.
- Add RTSP benchmark tooling in `benchmark/benchmark_rtsp.py`.
- Calibrate thresholds with real non-sensitive sample videos.

## Event-Frame Action Recognition Pipeline

The dataset JSON files contain event-level labels such as `annotations.event_class` and `annotations.event_frame`. They do not contain bbox ground truth, so this project does not convert them to YOLO txt labels or train YOLO from them.

For this dataset, YOLO is used as a pretrained person detector only. The action pipeline is:

```text
RTSP/local mp4
-> pretrained YOLO person bbox
-> largest person crop sequence
-> ActionClassifier
-> event_frame evaluation
-> bbox + event payload output/visualization
```

Run on a local mp4 and JSON label:

```bash
python -m ai.main --input path/to/video.mp4 --label path/to/label.json --detector-mode mock --max-frames 120
```

Run with pretrained YOLO person detector:

```bash
python -m ai.main --input path/to/video.mp4 --label path/to/label.json --detector-mode yolo --yolo-model yolov8n.pt
```

Save visualization:

```bash
python -m ai.main --input path/to/video.mp4 --label path/to/label.json --output-video outputs/annotated.mp4
```

Use a CSV containing `video_path,label_path,split`:

```bash
python -m ai.main --dataset-csv datasets/processed/clips_train.csv --split train --detector-mode mock
```

Train the LSTM action classifier from the same CSV:

```bash
bash scripts/run_yolov8n_vs_yolo11n_lstm.sh
```

For a quick demo run, limit rows, frames, and epochs:

```bash
DATASET_CSV=../ai_fall_experiments/data/metadata/metadata.csv MAX_ROWS_PER_SPLIT=3 MAX_FRAMES=120 EPOCHS=1 bash scripts/run_yolov8n_vs_yolo11n_lstm.sh
```

If YOLO does not find a person in a short demo clip, the training script uses the whole frame as a fallback crop by default. Disable that behavior with `FALLBACK_FULL_FRAME=false`.
The preprocessing step uses annotation `event_frame` ranges first when available, retries YOLO once with a lower confidence threshold, and writes detector/fallback metadata for every generated sequence.

This writes checkpoints and a comparison table:

```text
runs/action_lstm/yolov8n/best.pt
runs/action_lstm/yolo11n/best.pt
runs/action_lstm/summary.csv
runs/action_lstm/<model>/preprocess_sequences_train.csv
runs/action_lstm/<model>/preprocess_summary_train.json
```

Or run one model manually:

```bash
python -m ai.action.train_lstm \
  --dataset-csv datasets/processed/clips_train.csv \
  --train-split train \
  --val-split val \
  --detector-mode yolo \
  --yolo-model yolov8n.pt \
  --output-dir runs/action_lstm/yolov8n
```

Compare another YOLO detector backbone by changing `--yolo-model` and output directory:

```bash
python -m ai.action.train_lstm \
  --dataset-csv datasets/processed/clips_train.csv \
  --train-split train \
  --val-split val \
  --detector-mode yolo \
  --yolo-model yolo11n.pt \
  --output-dir runs/action_lstm/yolo11n
```

Run inference with a trained checkpoint:

```bash
python -m ai.main \
  --input path/to/video.mp4 \
  --label path/to/label.json \
  --detector-mode yolo \
  --yolo-model yolov8n.pt \
  --action-model runs/action_lstm/yolov8n/best.pt
```

If `--action-model` is not provided, the pipeline still falls back to `MockActionClassifier` for integration smoke tests.

## Dataset Split And 4-Camera Demo Checks

Verify the local fall dataset split ratio and source-video leakage:

```bash
python scripts/check_dataset_split.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv
```

Run split verification and the 4-camera dry-run together:

```bash
python scripts/run_dataset_rtsp_verification.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --config configs/demo_4cams.yaml \
  --detector-mode mock \
  --max-frames 60 \
  --write-fixed-split runs/dataset_split/metadata_stratified.csv
```

The combined report is written to `runs/verification/final_summary.json`.

Run a dataset pose/keypoint sequence dry-run without training:

```bash
python scripts/run_dataset_evaluation.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --detector-mode mock \
  --max-rows-per-split 2 \
  --max-frames 60 \
  --output runs/verification/dataset_evaluation_summary.json
```

This reports selected-row class counts, person bbox detections, keypoint extraction count, generated keypoint sequence count, zero-sequence clips, and fallback crop usage ratio.

If the ratio or leakage is wrong, write a safe candidate split without overwriting production metadata:

```bash
python scripts/check_dataset_split.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --write-fixed runs/dataset_split/metadata_stratified.csv
```

Run the safe 4-camera local dataset dry-run. This uses local/demo RTSP URLs from `configs/demo_4cams.yaml`, reads local dataset videos, prints bbox/event payloads in the summary, and does not contact MQTT/EQMS:

```bash
python scripts/run_rtsp_demo.py \
  --config configs/demo_4cams.yaml \
  --dataset-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --dry-run \
  --detector-mode mock \
  --max-frames 60
```

Run a single-camera RTSP AI inference dry-run against the local MediaMTX cam1 stream:

```bash
python scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam1 \
  --detector-mode mock \
  --dry-run \
  --max-frames 60 \
  --output runs/verification/rtsp_cam1_inference.json
```

Use the real YOLO pose detector when the pose model and dependencies are available:

```bash
python scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam1 \
  --detector-mode real \
  --yolo-model yolov8n-pose.pt \
  --dry-run \
  --max-frames 60
```

`serve_mjpeg.py` remains a raw RTSP-to-MJPEG stream server. It does not run YOLO/LSTM or draw overlays. The AI dry-run scripts above are the current local inference/event-output path.

To view the actual cam1 video with AI overlays in a browser, run the separate local overlay server. This keeps the working raw MJPEG stream untouched:

```bash
python scripts/serve_ai_overlay.py \
  --rtsp-url rtsp://localhost:8554/cam1 \
  --detector-mode mock \
  --port 8010 \
  --print-events
```

Open:

```text
http://localhost:8010/stream
```

The overlay shows person bbox, skeleton/keypoints when present, current Normal/Faint-style action label, confidence, frame count, bbox count, keypoint count, sequence count, prediction count, and event count. Use `--detector-mode real --yolo-model yolov8n-pose.pt` when YOLO Pose dependencies and model files are available.

The dry-run prints an RTSP publish plan using local-only URLs. To actually publish the four local videos to MediaMTX on the GPU PC, start the local RTSP server first, then opt in explicitly:

```bash
./scripts/run_rtsp_server.sh
python scripts/run_rtsp_demo.py \
  --config configs/demo_4cams.yaml \
  --dataset-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --dry-run \
  --start-rtsp-publishers \
  --read-from-rtsp \
  --max-frames 60
```
