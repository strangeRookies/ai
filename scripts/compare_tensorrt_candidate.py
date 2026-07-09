from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkArgs:
    model: Path
    engine: Path | None
    video: Path
    output_dir: Path
    imgsz: int
    device: str
    max_frames: int
    export_engine: bool
    engine_half: bool


@dataclass(frozen=True, slots=True)
class BackendResult:
    backend: str
    model_path: str
    status: str
    frames: int
    avg_latency_ms: float
    p95_latency_ms: float
    fps: float


@dataclass(frozen=True, slots=True)
class Comparison:
    torch: BackendResult
    tensorrt: BackendResult | None
    speedup: float
    latency_delta_ms: float
    recommendation: str


@dataclass(frozen=True, slots=True)
class DetectionEquivalence:
    frame_index: int
    torch_count: int
    tensorrt_count: int
    matched_count: int
    detection_count_diff: int
    avg_bbox_iou: float
    avg_keypoint_confidence_diff: float
    event_decision_diff: int


def parse_args(argv: list[str]) -> BenchmarkArgs:
    parser = argparse.ArgumentParser(
        description="Compare current YOLO .pt inference with an optional TensorRT .engine without changing runtime code."
    )
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--engine", default="")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="benchmark/results/tensorrt_candidate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--export-engine", action="store_true")
    parser.add_argument("--engine-half", action="store_true", help="Export FP16 TensorRT engine. Default is FP32 for compatibility.")
    parsed = parser.parse_args(argv)
    engine = Path(parsed.engine) if parsed.engine else Path(parsed.model).with_suffix(".engine")
    return BenchmarkArgs(
        model=Path(parsed.model),
        engine=engine,
        video=Path(parsed.video),
        output_dir=Path(parsed.output_dir),
        imgsz=parsed.imgsz,
        device=str(parsed.device),
        max_frames=parsed.max_frames,
        export_engine=bool(parsed.export_engine),
        engine_half=bool(parsed.engine_half),
    )


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile_value / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def benchmark_yolo(model_path: Path, backend: str, args: BenchmarkArgs) -> BackendResult:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        return failed_result(backend, model_path, f"FAILED: dependency import failed: {exc}")

    if not model_path.exists() and model_path.suffix in {".pt", ".engine"}:
        return failed_result(backend, model_path, f"SKIPPED: model file not found: {model_path}")
    if not args.video.exists():
        return failed_result(backend, model_path, f"FAILED: video file not found: {args.video}")

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        return failed_result(backend, model_path, f"FAILED: could not open video: {args.video}")

    latencies: list[float] = []
    started = time.perf_counter()
    frames = 0
    try:
        while frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_started = time.perf_counter()
            model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
            latencies.append((time.perf_counter() - frame_started) * 1000.0)
            frames += 1
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    avg_latency = statistics.fmean(latencies) if latencies else 0.0
    return BackendResult(
        backend=backend,
        model_path=str(model_path),
        status="OK" if frames > 0 else "FAILED: no frames processed",
        frames=frames,
        avg_latency_ms=float(avg_latency),
        p95_latency_ms=percentile(latencies, 95.0),
        fps=float(frames / elapsed) if elapsed > 0 else 0.0,
    )


def maybe_export_engine(args: BenchmarkArgs) -> tuple[Path | None, str]:
    if args.engine is not None and args.engine.exists():
        return args.engine, "OK"
    if not args.export_engine:
        return args.engine, "SKIPPED: TensorRT engine file does not exist and --export-engine was not set"
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return args.engine, f"EXPORT_FAILED: ultralytics import failed: {exc}"
    model = YOLO(str(args.model))
    try:
        exported = model.export(format="engine", half=args.engine_half, imgsz=args.imgsz, device=args.device)
    except (AttributeError, RuntimeError, ImportError, OSError) as exc:
        precision = "FP16" if args.engine_half else "FP32"
        return args.engine, f"EXPORT_FAILED: {precision} TensorRT export failed: {exc}"
    return Path(str(exported)), "OK"


def compare(torch_result: BackendResult, tensorrt_result: BackendResult | None) -> Comparison:
    if tensorrt_result is None or tensorrt_result.status != "OK" or torch_result.status != "OK":
        return Comparison(torch_result, tensorrt_result, 0.0, 0.0, non_comparable_recommendation(torch_result, tensorrt_result))
    speedup = torch_result.avg_latency_ms / max(tensorrt_result.avg_latency_ms, 0.001)
    latency_delta = torch_result.avg_latency_ms - tensorrt_result.avg_latency_ms
    recommendation = adoption_recommendation(speedup, latency_delta, torch_result.fps, tensorrt_result.fps)
    return Comparison(torch_result, tensorrt_result, speedup, latency_delta, recommendation)


def adoption_recommendation(speedup: float, latency_delta_ms: float, torch_fps: float, tensorrt_fps: float) -> str:
    if speedup >= 1.4 and latency_delta_ms >= 10.0:
        return "ADOPT_CANDIDATE: TensorRT shows enough latency gain. Keep Torch fallback and roll out gradually."
    if torch_fps < 10.0 and tensorrt_fps >= 10.0:
        return "ADOPT_CANDIDATE: TensorRT restores the target FPS budget."
    if speedup < 1.15:
        return "DEFER: Speedup is too small. Check RTSP decode, frame queue, and LSTM bottlenecks first."
    return "MEASURE_MORE: Gain exists, but run a long 4-camera test before runtime adoption."


def non_comparable_recommendation(torch_result: BackendResult, tensorrt_result: BackendResult | None) -> str:
    tensorrt_status = "MISSING: no TensorRT row" if tensorrt_result is None else tensorrt_result.status
    return (
        "DEFER: No comparable TensorRT result is available. "
        f"torch_status={torch_result.status}; tensorrt_status={tensorrt_status}"
    )


def failed_result(backend: str, model_path: Path, status: str) -> BackendResult:
    return BackendResult(backend, str(model_path), status, 0, 0.0, 0.0, 0.0)


def compare_detection_equivalence(
    frame_index: int,
    torch_detections: list[dict],
    tensorrt_detections: list[dict],
    iou_threshold: float = 0.50,
    torch_event_decision: bool = False,
    tensorrt_event_decision: bool = False,
) -> DetectionEquivalence:
    matches: list[tuple[int, int, float]] = []
    used_tensorrt: set[int] = set()
    for torch_index, torch_detection in enumerate(torch_detections):
        best_index: int | None = None
        best_iou = 0.0
        for tensorrt_index, tensorrt_detection in enumerate(tensorrt_detections):
            if tensorrt_index in used_tensorrt:
                continue
            iou = bbox_iou(torch_detection.get("bbox"), tensorrt_detection.get("bbox"))
            if iou > best_iou:
                best_iou = iou
                best_index = tensorrt_index
        if best_index is not None and best_iou >= iou_threshold:
            used_tensorrt.add(best_index)
            matches.append((torch_index, best_index, best_iou))
    keypoint_diffs = [
        _optional_float(tensorrt_detections[tensorrt_index].get("keypoint_confidence"), 0.0)
        - _optional_float(torch_detections[torch_index].get("keypoint_confidence"), 0.0)
        for torch_index, tensorrt_index, _iou in matches
    ]
    return DetectionEquivalence(
        frame_index=frame_index,
        torch_count=len(torch_detections),
        tensorrt_count=len(tensorrt_detections),
        matched_count=len(matches),
        detection_count_diff=len(tensorrt_detections) - len(torch_detections),
        avg_bbox_iou=statistics.fmean([iou for _torch_index, _tensorrt_index, iou in matches]) if matches else 0.0,
        avg_keypoint_confidence_diff=statistics.fmean(keypoint_diffs) if keypoint_diffs else 0.0,
        event_decision_diff=int(bool(torch_event_decision) != bool(tensorrt_event_decision)),
    )


def bbox_iou(left, right) -> float:
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _optional_float(value, default: float) -> float:
    if value is None:
        return default
    return float(value)


def write_reports(comparison: Comparison, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tensorrt_candidate_comparison.csv"
    md_path = output_dir / "tensorrt_candidate_comparison.md"
    rows = [comparison.torch]
    if comparison.tensorrt is not None:
        rows.append(comparison.tensorrt)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["backend", "model_path", "status", "frames", "avg_latency_ms", "p95_latency_ms", "fps"])
        for row in rows:
            writer.writerow([
                row.backend,
                row.model_path,
                row.status,
                row.frames,
                f"{row.avg_latency_ms:.3f}",
                f"{row.p95_latency_ms:.3f}",
                f"{row.fps:.3f}",
            ])
    md_path.write_text(markdown_report(comparison), encoding="utf-8")
    return csv_path, md_path


def markdown_report(comparison: Comparison) -> str:
    rows = [comparison.torch]
    if comparison.tensorrt is not None:
        rows.append(comparison.tensorrt)
    table_rows = "\n".join(
        f"| {row.backend} | {row.status} | {row.frames} | {row.avg_latency_ms:.3f} | {row.p95_latency_ms:.3f} | {row.fps:.3f} |"
        for row in rows
    )
    return (
        "# TensorRT Candidate Comparison\n\n"
        "| backend | status | frames | avg latency ms | p95 latency ms | fps |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        f"- speedup: {comparison.speedup:.3f}x\n"
        f"- latency_delta_ms: {comparison.latency_delta_ms:.3f}\n"
        f"- recommendation: {comparison.recommendation}\n"
    )


def markdown_equivalence_report(rows: list[DetectionEquivalence]) -> str:
    table_rows = "\n".join(
        "| "
        f"{row.frame_index} | "
        f"{row.torch_count} | "
        f"{row.tensorrt_count} | "
        f"{row.matched_count} | "
        f"{row.detection_count_diff} | "
        f"{row.avg_bbox_iou:.3f} | "
        f"{row.avg_keypoint_confidence_diff:.4f} | "
        f"{row.event_decision_diff} |"
        for row in rows
    )
    total_event_diff = sum(row.event_decision_diff for row in rows)
    avg_detection_diff = statistics.fmean([abs(row.detection_count_diff) for row in rows]) if rows else 0.0
    avg_keypoint_diff = statistics.fmean([row.avg_keypoint_confidence_diff for row in rows]) if rows else 0.0
    return (
        "# TensorRT Detection Equivalence Debug\n\n"
        "| frame | torch_count | tensorrt_count | matched | detection_count_diff | avg_bbox_iou | keypoint_confidence_diff | event_decision_diff |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        f"- frames_compared: {len(rows)}\n"
        f"- avg_abs_detection_count_diff: {avg_detection_diff:.3f}\n"
        f"- avg_keypoint_confidence_diff: {avg_keypoint_diff:.4f}\n"
        f"- event_decision_diff: {total_event_diff}\n"
    )


def get_yolo_detections(model_path: Path, args: BenchmarkArgs) -> list[list[dict]]:
    if not model_path.exists():
        return []
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ImportError in get_yolo_detections: {exc}", flush=True)
        return []

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        return []

    frames_detections: list[list[dict]] = []
    frames = 0
    try:
        while frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
            frame_detections: list[dict] = []
            if results and len(results) > 0:
                result = results[0]
                boxes = getattr(result, "boxes", None)
                keypoints = getattr(result, "keypoints", None)
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy().tolist()
                    conf = boxes.conf.cpu().numpy().tolist()
                    
                    kp_conf = None
                    if keypoints is not None and getattr(keypoints, "conf", None) is not None:
                        kp_conf = keypoints.conf.cpu().numpy().tolist()
                        
                    for idx in range(len(xyxy)):
                        bbox = xyxy[idx]
                        c_val = conf[idx]
                        
                        avg_kp_conf = 0.0
                        if kp_conf is not None and idx < len(kp_conf):
                            valid_kp = [float(v) for v in kp_conf[idx] if float(v) > 0.0]
                            if valid_kp:
                                avg_kp_conf = sum(valid_kp) / len(valid_kp)
                                
                        frame_detections.append({
                            "bbox": bbox,
                            "confidence": c_val,
                            "keypoint_confidence": avg_kp_conf,
                        })
            frames_detections.append(frame_detections)
            frames += 1
    finally:
        capture.release()
    return frames_detections


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    torch_result = benchmark_yolo(args.model, "torch", args)
    engine, export_status = maybe_export_engine(args)
    if export_status != "OK" and args.export_engine:
        tensorrt_result = failed_result("tensorrt", engine or Path(""), export_status)
    else:
        tensorrt_result = benchmark_yolo(engine, "tensorrt", args) if engine is not None else None
    comparison = compare(torch_result, tensorrt_result)
    csv_path, md_path = write_reports(comparison, args.output_dir)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print_backend_result(torch_result)
    if tensorrt_result is not None:
        print_backend_result(tensorrt_result)
        if torch_result.status == "OK" and tensorrt_result.status == "OK" and engine is not None and engine.exists():
            print("Running detection equivalence comparison...", flush=True)
            torch_dets = get_yolo_detections(args.model, args)
            trt_dets = get_yolo_detections(engine, args)
            
            eq_records = []
            min_len = min(len(torch_dets), len(trt_dets))
            for f_idx in range(min_len):
                eq = compare_detection_equivalence(
                    frame_index=f_idx,
                    torch_detections=torch_dets[f_idx],
                    tensorrt_detections=trt_dets[f_idx],
                    iou_threshold=0.50,
                    torch_event_decision=False,
                    tensorrt_event_decision=False,
                )
                eq_records.append(eq)
            
            if eq_records:
                eq_md = markdown_equivalence_report(eq_records)
                eq_md_path = args.output_dir / "tensorrt_equivalence_report.md"
                eq_md_path.write_text(eq_md, encoding="utf-8")
                print(f"Saved Equivalence Markdown: {eq_md_path}")
                
                eq_csv_path = args.output_dir / "tensorrt_equivalence_report.csv"
                with eq_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
                        "frame", "torch_count", "tensorrt_count", "matched",
                        "detection_count_diff", "avg_bbox_iou", "keypoint_confidence_diff", "event_decision_diff"
                    ])
                    for eq in eq_records:
                        writer.writerow([
                            eq.frame_index,
                            eq.torch_count,
                            eq.tensorrt_count,
                            eq.matched_count,
                            eq.detection_count_diff,
                            f"{eq.avg_bbox_iou:.3f}",
                            f"{eq.avg_keypoint_confidence_diff:.4f}",
                            eq.event_decision_diff
                        ])
                print(f"Saved Equivalence CSV: {eq_csv_path}")
    print(comparison.recommendation)
    return 0


def print_backend_result(result: BackendResult) -> None:
    print(
        f"{result.backend}: status={result.status}; frames={result.frames}; "
        f"avg_latency_ms={result.avg_latency_ms:.3f}; fps={result.fps:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
