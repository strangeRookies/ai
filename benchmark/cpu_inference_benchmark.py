import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU inference latency/FPS for CCTV control-room deployment decisions."
    )
    parser.add_argument("--input", required=True, help="Input video path or RTSP/MJPEG URL.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path/name.")
    parser.add_argument("--device", default="cpu", help="Use 'cpu' for backend feasibility checks.")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size.")
    parser.add_argument("--max-frames", type=int, default=300, help="Measured frames to process.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames to warm up before measuring.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference every N decoded frames.")
    parser.add_argument("--target-fps-per-camera", type=float, default=5.0, help="Required AI FPS per camera.")
    parser.add_argument("--camera-count", type=int, default=4, help="Planned camera count for capacity estimate.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Output directory.")
    return parser.parse_args()


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def load_runtime():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"opencv-python is required: {exc}") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(f"ultralytics is required: {exc}") from exc

    return cv2, YOLO


def run_benchmark(args):
    cv2, YOLO = load_runtime()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {args.input}")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    decoded_frames = 0
    measured_frames = 0
    detections = []
    infer_latencies_ms = []
    decode_latencies_ms = []

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    try:
        while measured_frames < args.max_frames:
            decode_start = time.perf_counter()
            ok, frame = cap.read()
            decode_latencies_ms.append((time.perf_counter() - decode_start) * 1000.0)
            if not ok:
                break

            decoded_frames += 1
            if args.frame_stride > 1 and decoded_frames % args.frame_stride != 0:
                continue

            if measured_frames < args.warmup_frames:
                model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
                measured_frames += 1
                continue

            infer_start = time.perf_counter()
            results = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
            infer_latencies_ms.append((time.perf_counter() - infer_start) * 1000.0)

            person_count = 0
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is not None and boxes.xyxy is not None:
                    person_count += len(boxes.xyxy)
            detections.append(person_count)
            measured_frames += 1
    finally:
        cap.release()

    wall_elapsed = time.perf_counter() - wall_start
    process_cpu_elapsed = time.process_time() - cpu_start
    measured_infer_frames = len(infer_latencies_ms)
    effective_fps = measured_infer_frames / wall_elapsed if wall_elapsed > 0 else 0.0
    infer_only_fps = 1000.0 / mean(infer_latencies_ms) if infer_latencies_ms else 0.0
    cpu_percent_one_core = (process_cpu_elapsed / wall_elapsed * 100.0) if wall_elapsed > 0 else 0.0
    required_total_fps = args.target_fps_per_camera * args.camera_count
    capacity_cameras = effective_fps / args.target_fps_per_camera if args.target_fps_per_camera > 0 else 0.0

    if measured_infer_frames == 0:
        recommendation = "failed_no_measured_frames"
    elif effective_fps >= required_total_fps:
        recommendation = "cpu_ok_for_configured_camera_count"
    elif capacity_cameras >= 1:
        recommendation = "cpu_ok_for_fewer_cameras_or_lower_fps"
    else:
        recommendation = "cpu_not_recommended_without_optimization"

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "input_width": width,
        "input_height": height,
        "input_fps": input_fps,
        "frame_stride": args.frame_stride,
        "decoded_frames": decoded_frames,
        "measured_infer_frames": measured_infer_frames,
        "wall_elapsed_seconds": wall_elapsed,
        "process_cpu_elapsed_seconds": process_cpu_elapsed,
        "process_cpu_percent_of_one_core": cpu_percent_one_core,
        "avg_decode_latency_ms": mean(decode_latencies_ms),
        "avg_infer_latency_ms": mean(infer_latencies_ms),
        "p50_infer_latency_ms": percentile(infer_latencies_ms, 50),
        "p95_infer_latency_ms": percentile(infer_latencies_ms, 95),
        "p99_infer_latency_ms": percentile(infer_latencies_ms, 99),
        "effective_pipeline_fps": effective_fps,
        "infer_only_fps": infer_only_fps,
        "avg_person_count": mean(detections),
        "target_fps_per_camera": args.target_fps_per_camera,
        "camera_count": args.camera_count,
        "required_total_fps": required_total_fps,
        "estimated_camera_capacity_at_target_fps": capacity_cameras,
        "recommendation": recommendation,
    }


def save_report(result, results_dir):
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"cpu_inference_benchmark_{stamp}.json"
    md_path = output_dir / f"cpu_inference_benchmark_{stamp}.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# CPU Inference Benchmark",
        "",
        f"- Input: `{result['input']}`",
        f"- Model: `{result['model']}`",
        f"- Device: `{result['device']}`",
        f"- Image size: `{result['imgsz']}`",
        f"- Input size/FPS: `{result['input_width']}x{result['input_height']} @ {result['input_fps']:.2f}`",
        "",
        "## Results",
        "",
        f"- Measured inference frames: `{result['measured_infer_frames']}`",
        f"- Effective pipeline FPS: `{result['effective_pipeline_fps']:.2f}`",
        f"- Inference-only FPS: `{result['infer_only_fps']:.2f}`",
        f"- Average inference latency: `{result['avg_infer_latency_ms']:.2f} ms`",
        f"- P95 inference latency: `{result['p95_infer_latency_ms']:.2f} ms`",
        f"- Process CPU percent of one core: `{result['process_cpu_percent_of_one_core']:.1f}%`",
        f"- Estimated camera capacity at target FPS: `{result['estimated_camera_capacity_at_target_fps']:.2f}`",
        "",
        "## Decision",
        "",
        f"- Target: `{result['camera_count']}` cameras x `{result['target_fps_per_camera']:.2f}` AI FPS",
        f"- Required total FPS: `{result['required_total_fps']:.2f}`",
        f"- Recommendation: `{result['recommendation']}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    args = parse_args()
    result = run_benchmark(args)
    json_path, md_path = save_report(result, args.results_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
