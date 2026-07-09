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


def maybe_export_engine(args: BenchmarkArgs) -> Path | None:
    if args.engine is not None and args.engine.exists():
        return args.engine
    if not args.export_engine:
        return args.engine
    try:
        from ultralytics import YOLO
    except ImportError:
        return args.engine
    model = YOLO(str(args.model))
    exported = model.export(format="engine", half=True, imgsz=args.imgsz, device=args.device)
    return Path(str(exported))


def compare(torch_result: BackendResult, tensorrt_result: BackendResult | None) -> Comparison:
    if tensorrt_result is None or tensorrt_result.status != "OK" or torch_result.status != "OK":
        return Comparison(torch_result, tensorrt_result, 0.0, 0.0, "보류: 비교 가능한 TensorRT 결과가 없습니다.")
    speedup = torch_result.avg_latency_ms / max(tensorrt_result.avg_latency_ms, 0.001)
    latency_delta = torch_result.avg_latency_ms - tensorrt_result.avg_latency_ms
    recommendation = adoption_recommendation(speedup, latency_delta, torch_result.fps, tensorrt_result.fps)
    return Comparison(torch_result, tensorrt_result, speedup, latency_delta, recommendation)


def adoption_recommendation(speedup: float, latency_delta_ms: float, torch_fps: float, tensorrt_fps: float) -> str:
    if speedup >= 1.4 and latency_delta_ms >= 10.0:
        return "도입 후보: TensorRT가 충분한 latency 이득을 보입니다. Torch fallback을 유지하고 단계 적용하세요."
    if torch_fps < 10.0 and tensorrt_fps >= 10.0:
        return "도입 후보: 목표 FPS 미달을 TensorRT가 회복합니다."
    if speedup < 1.15:
        return "보류: 이득이 작아 RTSP decode/frame queue/LSTM 병목을 먼저 확인하세요."
    return "추가 측정: 이득은 있으나 운영 전 4-camera 장시간 테스트가 필요합니다."


def failed_result(backend: str, model_path: Path, status: str) -> BackendResult:
    return BackendResult(backend, str(model_path), status, 0, 0.0, 0.0, 0.0)


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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    torch_result = benchmark_yolo(args.model, "torch", args)
    engine = maybe_export_engine(args)
    tensorrt_result = benchmark_yolo(engine, "tensorrt", args) if engine is not None else None
    comparison = compare(torch_result, tensorrt_result)
    csv_path, md_path = write_reports(comparison, args.output_dir)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(comparison.recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
