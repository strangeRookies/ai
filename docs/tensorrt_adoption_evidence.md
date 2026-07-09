# TensorRT Adoption Evidence Guide

## Goal

TensorRT should be adopted only if local measurements show that YOLO pose inference is the bottleneck and the `.engine` backend improves latency enough to justify the extra deployment risk.

## Safe Comparison

This comparison does not change the running worker path. It benchmarks the current `.pt` model and an optional TensorRT `.engine` file against the same video on the GPU PC.

### 1. Connect to GPU PC & Activate Environment

Do not run the benchmark locally or use `cd strange_ai` from `~`. Connect to the remote GPU PC and navigate to the project directory:

```bash
# SSH connection to GPU PC
ssh welabs@58.151.205.220

# Navigate to project checkout
cd /home/welabs/yolo_training/strange_ai_lstm

# Activate virtual environment
source .venv/bin/activate
```

*(Note: If `python` is not available after activating the venv, use `python3` instead.)*

### 2. Confirm Required Files

Before running the script, verify that the required files and scripts exist in the repository:

```bash
pwd
ls scripts/compare_tensorrt_candidate.py
ls yolo26n-pose.pt
ls sample_videos
```

### 3. Run Torch Baseline

Run the baseline benchmark to record PyTorch inference latency. 

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video /home/welabs/yolo_training/ai_fall_experiments/data/raw/outdoor_swoon/videos/outside_swoon_1.mp4 --imgsz 640 --max-frames 300
```

> [!NOTE]
> If the output says `DEFER: No comparable TensorRT result is available.`, this is expected before a valid `.engine` file exists. The baseline results will still be saved to the report.

### 4. Export TensorRT Engine

Exporting the model to a TensorRT `.engine` file is opt-in. Run this step only after the Torch baseline runs successfully.

#### Option A: FP32 Export (Recommended / Safer)
The default export is FP32. TensorRT 11.1.0.106 on the RTX 5080 host might fail on the FP16 path with an `AttributeError` (`BuilderFlag` has no attribute `FP16`). Run the safer FP32 path first:

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video /home/welabs/yolo_training/ai_fall_experiments/data/raw/outdoor_swoon/videos/outside_swoon_1.mp4 --imgsz 640 --max-frames 300 --export-engine
```

#### Option B: FP16 Export (Optional)
If FP32 succeeds and you want to test FP16 explicitly:

```bash
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --video /home/welabs/yolo_training/ai_fall_experiments/data/raw/outdoor_swoon/videos/outside_swoon_1.mp4 --imgsz 640 --max-frames 300 --export-engine --engine-half
```

> [!TIP]
> * During the first export, Ultralytics may automatically install missing packages (e.g., `onnx`, `onnxslim`, `onnxruntime-gpu`, `tensorrt-cu13`). 
> * If it prints `Restart runtime or rerun command for updates to take effect`, simply rerun the export command after the installation finishes.
> * Treat the FP16 `BuilderFlag.FP16` failure as a package compatibility issue, not as evidence that TensorRT is slower. The final decision should compare Torch against a successfully generated `.engine`.

### 5. Run Comparison Benchmark

Once the `.engine` file is generated, run the explicit comparison again:

```bash
# For FP32 comparison
python scripts/compare_tensorrt_candidate.py --model yolo26n-pose.pt --engine yolo26n-pose.engine --video /home/welabs/yolo_training/ai_fall_experiments/data/raw/outdoor_swoon/videos/outside_swoon_1.mp4 --imgsz 640 --max-frames 300
```

### 6. Verify Comparison Report

Reports are saved to `benchmark/results/tensorrt_candidate/`. Run the following command to check the results:

```bash
cat benchmark/results/tensorrt_candidate/tensorrt_candidate_comparison.md
```

*(Reminder: Always use forward slashes `/` for paths on Linux/Bash, not Windows backslashes `\`.)*

---

## Decision Rule

Adopt TensorRT only when one of these is true:

- Average inference latency improves by at least `1.4x` and at least `10 ms` per frame.
- Torch is below the target realtime budget, but TensorRT reaches the target FPS.

Defer TensorRT when:

- Speedup is below `1.15x`.
- RTSP frame read latency or frame queue pressure is the dominant bottleneck.
- LSTM/post-processing or MQTT/overlay publish latency dominates the end-to-end path.

## 4-Camera Follow-Up

The single-video TensorRT result is only the first gate. Run the RTSP metrics with separate output directories so Torch, TensorRT, stale, and failed rows do not mix:

```bash
YOLO_MODEL=yolo26n-pose.pt IMGSZ=640 MAX_FRAMES=3000 OUTPUT_DIR=runs/verification_torch bash scripts/run_4cam_rtsp_metrics.sh
python scripts/summarize_4cam_metrics.py --dir runs/verification_torch --target-fps 10
```

```bash
YOLO_MODEL=yolo26n-pose.engine IMGSZ=640 MAX_FRAMES=3000 OUTPUT_DIR=runs/verification_tensorrt bash scripts/run_4cam_rtsp_metrics.sh
python scripts/summarize_4cam_metrics.py --dir runs/verification_tensorrt --target-fps 10
```

`summarize_4cam_metrics.py` defaults to the latest valid row per camera and ignores zero-frame rows. Use `--all` only when diagnosing failed runs:

```bash
python scripts/summarize_4cam_metrics.py --dir runs/verification_tensorrt --all
```

## Rollout Shape

Keep Torch as the default fallback. Add TensorRT as a selectable backend only after the comparison report and a 4-camera runtime metrics run both support adoption.
