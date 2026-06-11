import argparse
import os

from ai.action.faint_post_processing import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
)


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run safe local RTSP YOLO Pose + LSTM inference dry-run.")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/cam1"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam_01"))
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Load configured detector/classifier and print settings without opening RTSP or MQTT.")
    parser.add_argument("--output", default=None, help="Write one run summary JSON containing counters and sample_event.")
    parser.add_argument("--event-log-dir", default=None, help="Write one debug event JSON file per emitted event.")
    parser.add_argument("--evaluation-log", default=None, help="Append one JSONL row per LSTM event candidate for offline evaluation.")
    parser.add_argument("--ground-truth", choices=["Normal", "Faint", "hard_negative", "normal_basic"], default=None)
    parser.add_argument("--source-id", default=None, help="Stable source ID used in evaluation logs. Defaults to rtsp_url.")
    parser.add_argument("--video-id", default=None, help="Stable video ID used in evaluation logs. Defaults to source_id or rtsp_url.")
    parser.add_argument("--publisher", choices=["console", "mqtt"], default=os.getenv("EVENT_PUBLISHER"), help="Event publisher. Default: console in --dry-run, mqtt otherwise.")
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST"), help="MQTT broker host. Defaults to MQTT_HOST or localhost.")
    parser.add_argument("--mqtt-port", type=int, default=None, help="MQTT broker port. Defaults to MQTT_PORT or 1883.")
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_TOPIC"), help="MQTT topic. Defaults to MQTT_TOPIC or safety/events.")
    parser.add_argument("--mqtt-client-id", default=os.getenv("MQTT_CLIENT_ID"), help="MQTT client ID. Defaults to MQTT_CLIENT_ID or strange-ai-local.")
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"), help="MQTT username. Defaults to MQTT_USERNAME.")
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"), help="MQTT password. Defaults to MQTT_PASSWORD and is never printed.")
    parser.add_argument("--overlay-output", default=None)
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL", "yolo26n-pose.pt"))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "auto"))
    parser.add_argument("--imgsz", type=int, default=env_int("YOLO_IMGSZ", 640))
    parser.add_argument("--detector-conf", type=float, default=env_float("YOLO_CONF", 0.10))
    parser.add_argument("--action-model", default=os.getenv("ACTION_MODEL") or DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default=os.getenv("ACTION_DEVICE", "auto"))
    parser.add_argument("--action-threshold", type=float, default=env_float("ACTION_THRESHOLD", DEFAULT_FAINT_THRESHOLD), help="Faint probability threshold for LSTM checkpoints with Normal/Faint classes.")
    parser.add_argument("--min-consecutive-faint", type=int, default=env_int("MIN_CONSECUTIVE_FAINT", DEFAULT_MIN_CONSECUTIVE_FAINT), help="Consecutive Faint sequences required before emitting an event.")
    parser.add_argument("--camera-cooldown-seconds", type=float, default=env_float("CAMERA_COOLDOWN_SECONDS", DEFAULT_CAMERA_COOLDOWN_SECONDS), help="Per-camera event cooldown after a Faint event.")
    parser.add_argument("--event-severity", default="HIGH")
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=env_int("SEQUENCE_LENGTH", 8))
    parser.add_argument("--sequence-stride", type=int, default=env_int("SEQUENCE_STRIDE", 4))
    parser.add_argument("--resize-size", type=int, default=env_int("RESIZE_SIZE", 224))
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default=os.getenv("TRACKING_MODE", "auto"))
    parser.add_argument("--track-thresh", type=float, default=env_float("TRACK_THRESH", 0.10))
    parser.add_argument("--match-thresh", "--tracker-iou-threshold", dest="match_thresh", type=float, default=env_float("TRACK_IOU_THRESHOLD", 0.20))
    parser.add_argument("--track-buffer", type=int, default=env_int("TRACK_BUFFER", 90))
    parser.add_argument("--min-box-area", type=float, default=env_float("MIN_BOX_AREA", 100.0))
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=env_float("BBOX_SMOOTHING_ALPHA", 0.60))
    parser.add_argument("--track-max-missing-seconds", type=float, default=env_float("TRACK_MAX_MISSING_SECONDS", 4.0))
    parser.add_argument("--center-match-ratio", type=float, default=env_float("CENTER_MATCH_RATIO", 0.70))
    return parser.parse_args(argv)
