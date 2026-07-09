# TensorRT Adoption Evidence Guide

## Goal

TensorRT should be adopted only if local measurements show that YOLO pose inference is the bottleneck and the `.engine` backend improves latency enough to justify the extra deployment risk.

## Safe Comparison

This comparison does not change the running worker path. It benchmarks the current `.pt` model and an optional TensorRT `.engine` file against the same video on the GPU PC.

First, connect to the GPU PC and activate the virtual environment. Do not use `cd strange_ai` from `~`; the current GPU-PC checkout is under `/home/welabs/yolo_training/strange_ai_lstm`.

```bash
ssh welabs@58.151.205.220
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate
```

Confirm the files before running:

```bash
pwd
ls scripts/compare_tensorrt_candidate.py
ls yolo26n-pose.pt
ls sample_videos
```

Run the Torch baseline first:

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video sample_videos/sample.mp4 --max-frames 300
```

If the output says `DEFER: No comparable TensorRT result is available.`, that is expected before a valid `.engine` exists. The baseline still writes the Torch row to the report.

If an engine already exists, pass it explicitly:

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --engine yolo26n-pose.engine --video sample_videos/sample.mp4 --max-frames 300
```

Engine export is opt-in. Run it only after the Torch baseline succeeds:

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video sample_videos/sample.mp4 --max-frames 300 --export-engine
```

Ultralytics may install missing packages such as `onnx`, `onnxslim`, `onnxruntime-gpu`, or `tensorrt-cu13` during the first export. If it prints `Restart runtime or rerun command for updates to take effect`, rerun the same export command after it finishes. Then run the explicit engine comparison command again.

Reports are written to `benchmark/results/tensorrt_candidate/`.

Check the report:

```bash
cat benchmark/results/tensorrt_candidate/tensorrt_candidate_comparison.md
```

If `python` is not available but the venv is active, use:

```bash
python3 scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video sample_videos/sample.mp4 --max-frames 300
```

On bash/Linux paths, use `/`, not Windows `\`.

## Decision Rule

Adopt TensorRT only when one of these is true:

- Average inference latency improves by at least `1.4x` and at least `10 ms` per frame.
- Torch is below the target realtime budget, but TensorRT reaches the target FPS.

Defer TensorRT when:

- Speedup is below `1.15x`.
- RTSP frame read latency or frame queue pressure is the dominant bottleneck.
- LSTM/post-processing or MQTT/overlay publish latency dominates the end-to-end path.

## Rollout Shape

Keep Torch as the default fallback. Add TensorRT as a selectable backend only after the comparison report and a 4-camera runtime metrics run both support adoption.
