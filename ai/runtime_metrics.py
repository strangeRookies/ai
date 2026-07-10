import time
from statistics import fmean


class RuntimeMetrics:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.read_ms_total = 0.0
        self.yolo_ms_total = 0.0
        self.lstm_ms_total = 0.0
        self.total_frame_ms_total = 0.0
        self.lstm_samples = 0
        self.max_active_tracks = 0
        self.yolo_ms_samples: list[float] = []
        self.warmup_frames = 0
        self._warmup_skip = 0

    def set_warmup_skip(self, n: int) -> None:
        self._warmup_skip = max(0, int(n))

    def add_read_ms(self, value):
        self.read_ms_total += float(value)

    def add_yolo_ms(self, value):
        v = float(value)
        self.yolo_ms_total += v
        # Track all samples for percentile; optional warmup skip for reported stats
        if len(self.yolo_ms_samples) + self.warmup_frames >= self._warmup_skip:
            self.yolo_ms_samples.append(v)
        else:
            self.warmup_frames += 1

    def add_lstm_ms(self, value):
        self.lstm_ms_total += float(value)
        self.lstm_samples += 1

    def add_total_frame_ms(self, value):
        self.total_frame_ms_total += float(value)

    def observe_active_tracks(self, active_tracks):
        self.max_active_tracks = max(self.max_active_tracks, int(active_tracks))

    def runtime_seconds(self):
        return max(0.0, time.perf_counter() - self.started_at)

    def yolo_latency_stats(self) -> dict:
        samples = list(self.yolo_ms_samples)
        return {
            "frames": len(samples),
            "avg_infer_ms": _avg_list(samples),
            "p50_infer_ms": percentile(samples, 50),
            "p95_infer_ms": percentile(samples, 95),
            "fps": _fps_from_avg_ms(_avg_list(samples)),
        }

    def summary(self, frames_processed, bbox_detections, keypoints_extracted, generated_sequences, lstm_predictions):
        frames = max(0, int(frames_processed))
        runtime_seconds = self.runtime_seconds()
        latency = self.yolo_latency_stats()
        return {
            "runtime_seconds": round(runtime_seconds, 6),
            "effective_fps": _rate(frames, runtime_seconds),
            "avg_frame_read_ms": _avg(self.read_ms_total, frames),
            "avg_yolo_inference_ms": _avg(self.yolo_ms_total, frames),
            "p50_yolo_inference_ms": latency["p50_infer_ms"],
            "p95_yolo_inference_ms": latency["p95_infer_ms"],
            "avg_lstm_inference_ms": _avg(self.lstm_ms_total, self.lstm_samples),
            "avg_total_frame_ms": _avg(self.total_frame_ms_total, frames),
            "bbox_per_frame": _rate(int(bbox_detections), frames),
            "keypoints_per_frame": _rate(int(keypoints_extracted), frames),
            "sequence_per_frame": _rate(int(generated_sequences), frames),
            "prediction_per_frame": _rate(int(lstm_predictions), frames),
            "max_active_tracks": self.max_active_tracks,
            "gpu_memory": gpu_memory_snapshot(),
            "gpu_memory_warning": gpu_memory_warning(),
        }


def percentile(samples: list[float], p: float) -> float | None:
    """Inclusive linear percentile for inference latency samples."""
    if not samples:
        return None
    ordered = sorted(float(x) for x in samples)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (float(p) / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    value = ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return round(value, 6)


def compute_latency_report(
    latencies_ms: list[float],
    *,
    warmup: int = 0,
) -> dict:
    """Shared helper for worker logs and benchmark scripts."""
    warm = max(0, int(warmup))
    samples = [float(x) for x in latencies_ms[warm:]] if warm else [float(x) for x in latencies_ms]
    avg = _avg_list(samples)
    return {
        "frames": len(samples),
        "warmup_skipped": min(warm, len(latencies_ms)),
        "avg_infer_ms": avg,
        "p50_infer_ms": percentile(samples, 50),
        "p95_infer_ms": percentile(samples, 95),
        "fps": _fps_from_avg_ms(avg),
    }


def _avg_list(samples: list[float]) -> float | None:
    if not samples:
        return None
    return round(fmean(samples), 6)


def _fps_from_avg_ms(avg_ms: float | None) -> float | None:
    if avg_ms is None or avg_ms <= 0:
        return None
    return round(1000.0 / avg_ms, 6)


def _avg(total, count):
    if int(count) <= 0:
        return None
    return round(float(total) / int(count), 6)


def _rate(total, count):
    if float(count) <= 0:
        return 0.0
    return round(float(total) / float(count), 6)


def gpu_memory_snapshot():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    divisor = 1024 * 1024
    return {
        "device": int(device),
        "allocated_mb": round(float(torch.cuda.memory_allocated(device)) / divisor, 3),
        "reserved_mb": round(float(torch.cuda.memory_reserved(device)) / divisor, 3),
        "max_allocated_mb": round(float(torch.cuda.max_memory_allocated(device)) / divisor, 3),
    }


def gpu_memory_warning():
    try:
        import torch
    except ImportError:
        return "torch unavailable; GPU memory was not collected."
    if not torch.cuda.is_available():
        return "CUDA unavailable; GPU memory was not collected."
    return None
