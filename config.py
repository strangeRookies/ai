import os
from dataclasses import dataclass


def get_env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {value}")


def get_env_float(name, default):
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got: {value}")


def get_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    rtsp_url: str
    camera_id: str
    detector_mode: str
    yolo_model: str
    yolo_device: str
    frame_queue_size: int
    reconnect_delay_seconds: float
    mock_frame_interval_seconds: float
    allow_mock_fallback: bool
    mqtt_host: str
    mqtt_port: int
    mqtt_topic: str
    mqtt_client_id: str
    mqtt_username: str
    mqtt_password: str
    fall_min_duration_seconds: float
    fall_debounce_seconds: float
    fall_candidate_threshold: float
    fall_decision_window: int
    fall_decision_required: int
    track_iou_threshold: float
    track_max_missing_seconds: float
    sequence_length: int
    sequence_max_track_age_seconds: float
    event_clip_pre_frames: int
    event_clip_post_frames: int
    event_clip_cooldown_seconds: float
    event_clip_fps: float
    event_clip_output_dir: str
    event_clip_queue_size: int
    event_clip_enabled: bool
    max_frames: int


def load_settings():
    return Settings(
        rtsp_url=os.getenv("RTSP_URL", "rtsp://localhost:8554/cam01"),
        camera_id=os.getenv("CAMERA_ID", "cam_01"),
        detector_mode=os.getenv("DETECTOR_MODE", "mock").strip().lower(),
        yolo_model=os.getenv("YOLO_MODEL", "yolo26n-pose.pt"),
        yolo_device=os.getenv("YOLO_DEVICE", "auto"),
        frame_queue_size=get_env_int("FRAME_QUEUE_SIZE", 2),
        reconnect_delay_seconds=get_env_float("RTSP_RECONNECT_DELAY_SECONDS", 3),
        mock_frame_interval_seconds=get_env_float("MOCK_FRAME_INTERVAL_SECONDS", 0.2),
        allow_mock_fallback=get_env_bool("ALLOW_MOCK_FALLBACK", True),
        mqtt_host=os.getenv("MQTT_HOST", "localhost"),
        mqtt_port=get_env_int("MQTT_PORT", 1883),
        mqtt_topic=os.getenv("MQTT_TOPIC", "safety/events"),
        mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "edge-ai-001"),
        mqtt_username=os.getenv("MQTT_USERNAME", ""),
        mqtt_password=os.getenv("MQTT_PASSWORD", ""),
        fall_min_duration_seconds=get_env_float("FALL_MIN_DURATION_SECONDS", 1.5),
        fall_debounce_seconds=get_env_float("FALL_DEBOUNCE_SECONDS", 10),
        fall_candidate_threshold=get_env_float("FALL_CANDIDATE_THRESHOLD", 0.7),
        fall_decision_window=get_env_int("FALL_DECISION_WINDOW", 3),
        fall_decision_required=get_env_int("FALL_DECISION_REQUIRED", 2),
        track_iou_threshold=get_env_float("TRACK_IOU_THRESHOLD", 0.3),
        track_max_missing_seconds=get_env_float("TRACK_MAX_MISSING_SECONDS", 2),
        sequence_length=get_env_int("SEQUENCE_LENGTH", 30),
        sequence_max_track_age_seconds=get_env_float("SEQUENCE_MAX_TRACK_AGE_SECONDS", 5),
        event_clip_pre_frames=get_env_int("EVENT_CLIP_PRE_FRAMES", 150),
        event_clip_post_frames=get_env_int("EVENT_CLIP_POST_FRAMES", 150),
        event_clip_cooldown_seconds=get_env_float("EVENT_CLIP_COOLDOWN_SECONDS", 10),
        event_clip_fps=get_env_float("EVENT_CLIP_FPS", 30),
        event_clip_output_dir=os.getenv("EVENT_CLIP_OUTPUT_DIR", "clips"),
        event_clip_queue_size=get_env_int("EVENT_CLIP_QUEUE_SIZE", 8),
        event_clip_enabled=get_env_bool("EVENT_CLIP_ENABLED", True),
        max_frames=get_env_int("MAX_FRAMES", 0),
    )
