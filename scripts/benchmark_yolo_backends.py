#!/usr/bin/env python3
"""Compare PyTorch vs TensorRT YOLO pose latency on the same frames.

Usage:
  python scripts/benchmark_yolo_backends.py --video path.mp4 --pt yolo26n-pose.pt --engine yolo26n-pose.engine
  python scripts/benchmark_yolo_backends.py --synthetic --frames 120 --warmup 20

Writes JSON (+ optional CSV) with avg/p50/p95/FPS. Gracefully records SKIP when
CUDA/engine/ultralytics are unavailable (no invented hardware numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.runtime_metrics import compute_latency_report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark YOLO .pt vs .engine inference latency.")
    p.add_argument("--video", default=None, help="Video path (optional if --synthetic).")
    p.add_argument("--pt", default="yolo26n-pose.pt")
    p.add_argument("--engine", default="yolo26n-pose.engine")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=120, help="Measured frames after warmup.")
    p.add_argument("--synthetic", action="store_true", help="Use random numpy frames (no video file).")
    p.add_argument("--output-json", default="benchmark/results/yolo_backend_latency.json")
    p.add_argument("--output-csv", default="benchmark/results/yolo_backend_latency.csv")
    p.add_argument("--skip-engine", action="store_true")
    p.add_argument("--skip-pt", action="store_true")
    return p.parse_args(argv)


def _load_frames(args) -> list:
    import numpy as np

    total = max(1, int(args.warmup) + int(args.frames))
    if args.synthetic or not args.video:
        return [np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8) for _ in range(total)]
    path = Path(args.video)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"cv2 required for video benchmark: {exc}") from exc
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < total:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    # loop-pad if short
    while len(frames) < total:
        frames.append(frames[len(frames) % max(1, len(frames))])
    return frames[:total]


def _time_model(model_path: str, frames: list, device: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, device=device, imgsz=imgsz, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {"latencies_ms": latencies, "status": "OK"}


def run_benchmark(args) -> dict:
    report = {
        "status": "OK",
        "warmup": int(args.warmup),
        "measure_frames": int(args.frames),
        "device": args.device,
        "imgsz": int(args.imgsz),
        "backends": {},
    }
    try:
        frames = _load_frames(args)
    except Exception as exc:
        report["status"] = f"SKIPPED: frame load failed: {exc}"
        return report

    for name, path, skip in (
        ("pytorch", args.pt, args.skip_pt),
        ("tensorrt", args.engine, args.skip_engine),
    ):
        if skip:
            report["backends"][name] = {"status": "SKIPPED: user flag", "model_path": path}
            continue
        p = Path(path)
        if not p.exists() and name == "tensorrt":
            report["backends"][name] = {
                "status": f"SKIPPED: engine not found: {path}",
                "model_path": path,
            }
            continue
        if not p.exists() and name == "pytorch":
            report["backends"][name] = {
                "status": f"SKIPPED: weights not found: {path}",
                "model_path": path,
            }
            continue
        try:
            timed = _time_model(path, frames, args.device, args.imgsz)
            stats = compute_latency_report(timed["latencies_ms"], warmup=int(args.warmup))
            report["backends"][name] = {
                "status": "OK",
                "model_path": path,
                **stats,
            }
        except Exception as exc:
            report["backends"][name] = {
                "status": f"FAILED: {exc}",
                "model_path": path,
            }
    return report


def write_outputs(report: dict, json_path: str, csv_path: str) -> None:
    jp = Path(json_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cp = Path(csv_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "status", "model_path", "frames", "avg_infer_ms", "p50_infer_ms", "p95_infer_ms", "fps"])
        for backend, row in (report.get("backends") or {}).items():
            w.writerow(
                [
                    backend,
                    row.get("status"),
                    row.get("model_path"),
                    row.get("frames"),
                    row.get("avg_infer_ms"),
                    row.get("p50_infer_ms"),
                    row.get("p95_infer_ms"),
                    row.get("fps"),
                ]
            )


def main(argv=None):
    args = parse_args(argv)
    report = run_benchmark(args)
    write_outputs(report, args.output_json, args.output_csv)
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "OK" else 0  # non-zero only on hard errors


if __name__ == "__main__":
    raise SystemExit(main())
