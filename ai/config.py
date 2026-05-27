import argparse
import os
from dataclasses import dataclass


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class AiPipelineConfig:
    input_uri: str
    label_path: str | None
    dataset_csv: str | None
    split: str
    camera_id: str
    detector_mode: str
    yolo_model: str
    yolo_conf: float
    yolo_iou: float
    sequence_length: int
    sequence_stride: int
    resize_size: int
    output_video: str | None
    event_clip_enabled: bool
    event_clip_pre_frames: int
    event_clip_post_frames: int
    event_clip_cooldown_seconds: float
    event_clip_output_dir: str
    event_clip_queue_size: int
    max_frames: int


def parse_config():
    parser = argparse.ArgumentParser(description="Run event-frame action-recognition inference.")
    parser.add_argument("--input", dest="input_uri", default=os.getenv("INPUT_URI", os.getenv("RTSP_URL", "")))
    parser.add_argument("--label", dest="label_path", default=os.getenv("LABEL_PATH"))
    parser.add_argument("--dataset-csv", default=os.getenv("DATASET_CSV"))
    parser.add_argument("--split", default=os.getenv("SPLIT", "train"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam_01"))
    parser.add_argument("--detector-mode", default=os.getenv("DETECTOR_MODE", "mock"), choices=["mock", "yolo"])
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--yolo-conf", type=float, default=env_float("YOLO_CONF", 0.35))
    parser.add_argument("--yolo-iou", type=float, default=env_float("YOLO_IOU", 0.5))
    parser.add_argument("--sequence-length", type=int, default=env_int("SEQUENCE_LENGTH", 16))
    parser.add_argument("--sequence-stride", type=int, default=env_int("SEQUENCE_STRIDE", 8))
    parser.add_argument("--resize-size", type=int, default=env_int("RESIZE_SIZE", 224))
    parser.add_argument("--output-video", default=os.getenv("OUTPUT_VIDEO"))
    parser.add_argument("--event-clip-enabled", action=argparse.BooleanOptionalAction, default=os.getenv("EVENT_CLIP_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"})
    parser.add_argument("--event-clip-pre-frames", type=int, default=env_int("EVENT_CLIP_PRE_FRAMES", 150))
    parser.add_argument("--event-clip-post-frames", type=int, default=env_int("EVENT_CLIP_POST_FRAMES", 150))
    parser.add_argument("--event-clip-cooldown-seconds", type=float, default=env_float("EVENT_CLIP_COOLDOWN_SECONDS", 10))
    parser.add_argument("--event-clip-output-dir", default=os.getenv("EVENT_CLIP_OUTPUT_DIR", "clips"))
    parser.add_argument("--event-clip-queue-size", type=int, default=env_int("EVENT_CLIP_QUEUE_SIZE", 8))
    parser.add_argument("--max-frames", type=int, default=env_int("MAX_FRAMES", 0))
    args = parser.parse_args()
    return AiPipelineConfig(**vars(args))
