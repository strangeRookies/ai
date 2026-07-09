# TensorRT Adoption Evidence Guide

## Goal

TensorRT should be adopted only if local measurements show that YOLO pose inference is the bottleneck and the `.engine` backend improves latency enough to justify the extra deployment risk.

## Safe Comparison

This comparison does not change the running worker path. It benchmarks the current `.pt` model and an optional TensorRT `.engine` file against the same video.

```powershell
cd strange_ai
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video sample_videos\sample.mp4 --max-frames 300
```

If an engine already exists, pass it explicitly:

```powershell
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --engine yolo26n-pose.engine --video sample_videos\sample.mp4 --max-frames 300
```

Engine export is opt-in:

```powershell
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video sample_videos\sample.mp4 --max-frames 300 --export-engine
```

Reports are written to `benchmark/results/tensorrt_candidate/`.

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
