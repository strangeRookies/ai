import time


class RuntimeMetrics:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.read_ms_total = 0.0
        self.yolo_ms_total = 0.0
        self.lstm_ms_total = 0.0
        self.total_frame_ms_total = 0.0
        self.lstm_samples = 0
        self.max_active_tracks = 0

    def add_read_ms(self, value):
        self.read_ms_total += float(value)

    def add_yolo_ms(self, value):
        self.yolo_ms_total += float(value)

    def add_lstm_ms(self, value):
        self.lstm_ms_total += float(value)
        self.lstm_samples += 1

    def add_total_frame_ms(self, value):
        self.total_frame_ms_total += float(value)

    def observe_active_tracks(self, active_tracks):
        self.max_active_tracks = max(self.max_active_tracks, int(active_tracks))

    def runtime_seconds(self):
        return max(0.0, time.perf_counter() - self.started_at)

    def summary(self, frames_processed, bbox_detections, keypoints_extracted, generated_sequences, lstm_predictions):
        frames = max(0, int(frames_processed))
        runtime_seconds = self.runtime_seconds()
        return {
            "runtime_seconds": round(runtime_seconds, 6),
            "effective_fps": _rate(frames, runtime_seconds),
            "avg_frame_read_ms": _avg(self.read_ms_total, frames),
            "avg_yolo_inference_ms": _avg(self.yolo_ms_total, frames),
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
