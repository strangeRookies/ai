"""Inference backend start logs and periodic metrics (mock/GPU-ready).

Does not claim TensorRT performance improvements. Offline tests use fake timers.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


Clock = Callable[[], float]


@dataclass
class BackendStartLog:
    requested_backend: str
    actual_backend: str
    fallback_reason: str | None
    model_path: str | None
    engine_path: str | None
    device: str
    precision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "fallback_reason": self.fallback_reason,
            "model_path": self.model_path,
            "engine_path": self.engine_path,
            "device": self.device,
            "precision": self.precision,
        }


@dataclass
class BackendMetricsCollector:
    clock: Clock = time.perf_counter
    inference_ms_samples: list[float] = field(default_factory=list)
    frame_drop_count: int = 0
    processed_frame_count: int = 0
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = float(self.clock())

    def record_inference_ms(self, value: float) -> None:
        self.inference_ms_samples.append(float(value))
        self.processed_frame_count += 1

    def record_frame_drop(self, count: int = 1) -> None:
        self.frame_drop_count += int(count)

    def periodic_metrics(self) -> dict[str, Any]:
        samples = list(self.inference_ms_samples)
        avg = (sum(samples) / len(samples)) if samples else None
        return {
            "avg_inference_ms": avg,
            "p50_inference_ms": _percentile(samples, 50),
            "p95_inference_ms": _percentile(samples, 95),
            "fps": (1000.0 / avg) if avg and avg > 0 else None,
            "frame_drop_count": self.frame_drop_count,
            "processed_frame_count": self.processed_frame_count,
            "elapsed_sec": max(0.0, float(self.clock()) - self.started_at),
        }


def select_backend_with_logging(
    *,
    requested_backend: str,
    model_path: str | None = None,
    engine_path: str | None = None,
    device: str = "cpu",
    precision: str | None = None,
    mock: bool = True,
) -> BackendStartLog:
    """Resolve backend for offline/mock or real later. No performance claims."""
    requested = (requested_backend or "pytorch").lower()
    if mock or requested in {"mock", "fake"}:
        return BackendStartLog(
            requested_backend=requested,
            actual_backend="mock",
            fallback_reason="mock_mode_or_no_gpu" if requested != "mock" else None,
            model_path=model_path,
            engine_path=engine_path,
            device=device,
            precision=precision or "n/a",
        )
    if requested == "tensorrt" and engine_path:
        # Real validation happens in tensorrt_runtime when GPU present.
        return BackendStartLog(
            requested_backend="tensorrt",
            actual_backend="tensorrt",
            fallback_reason=None,
            model_path=model_path,
            engine_path=engine_path,
            device=device,
            precision=precision or "fp16",
        )
    return BackendStartLog(
        requested_backend=requested,
        actual_backend="pytorch",
        fallback_reason=None if requested == "pytorch" else "tensorrt_not_selected_or_unavailable",
        model_path=model_path,
        engine_path=engine_path,
        device=device,
        precision=precision or "fp32",
    )


def run_mock_backend_benchmark(
    *,
    output_root: str | Path = "runs/backend_benchmark",
    iterations: int = 20,
    fake_inference_ms: float = 8.0,
    clock: Clock | None = None,
    requested_backend: str = "tensorrt",
) -> dict[str, Any]:
    """Write benchmark artifacts using fake timing (no GPU). Never claims TRT improvement."""
    clock = clock or time.perf_counter
    run_id = f"bench-{uuid.uuid4().hex[:10]}"
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=True)

    start = select_backend_with_logging(
        requested_backend=requested_backend,
        model_path="models/mock.pt",
        engine_path="models/mock.engine",
        device="cpu",
        mock=True,
    )
    collector = BackendMetricsCollector(clock=clock)
    for _ in range(max(1, int(iterations))):
        # Fake timer: record configured ms without sleeping for speed.
        collector.record_inference_ms(fake_inference_ms)

    metrics = {
        "disclaimer": "Mock/fake-timer benchmark only. Not a TensorRT performance result.",
        "start": start.to_dict(),
        "periodic": collector.periodic_metrics(),
        "iterations": iterations,
        "fake_inference_ms": fake_inference_ms,
    }
    manifest = {
        "runId": run_id,
        "disclaimer": metrics["disclaimer"],
        "requested_backend": start.requested_backend,
        "actual_backend": start.actual_backend,
        "outputDir": str(out).replace("\\", "/"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Backend Benchmark Report",
            "",
            f"> **{metrics['disclaimer']}**",
            "",
            f"- runId: `{run_id}`",
            f"- requested: `{start.requested_backend}` actual: `{start.actual_backend}`",
            f"- fallback_reason: `{start.fallback_reason}`",
            f"- device: `{start.device}` precision: `{start.precision}`",
            f"- processed_frame_count: {metrics['periodic']['processed_frame_count']}",
            f"- avg_inference_ms (fake): {metrics['periodic']['avg_inference_ms']}",
            "",
            "Do not interpret these numbers as TensorRT speedup or accuracy gains.",
            "",
        ]
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    return {"runId": run_id, "outputDir": str(out), "metrics": metrics, "manifest": manifest}


def _percentile(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)
