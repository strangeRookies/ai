import argparse
import os

from ai.action.faint_post_processing import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
)
from ai.action.lstm_contract import DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run safe local RTSP YOLO Pose + LSTM inference dry-run.")
    parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/cam1"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam_01"))
    parser.add_argument("--camera-login-id", default=os.getenv("CAMERA_LOGIN_ID"))
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
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_TOPIC"), help="Legacy MQTT event topic alias.")
    parser.add_argument("--mqtt-camera-topic", default=os.getenv("MQTT_CAMERA_TOPIC"), help="MQTT overlay topic. Defaults to MQTT_CAMERA_TOPIC or camera.")
    parser.add_argument("--mqtt-event-topic", default=os.getenv("MQTT_EVENT_TOPIC"), help="MQTT confirmed event topic. Defaults to MQTT_EVENT_TOPIC, MQTT_TOPIC, or event.")
    parser.add_argument("--mqtt-client-id", default=os.getenv("MQTT_CLIENT_ID"), help="MQTT client ID. Defaults to MQTT_CLIENT_ID or strange-ai-local.")
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"), help="MQTT username. Defaults to MQTT_USERNAME.")
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"), help="MQTT password. Defaults to MQTT_PASSWORD and is never printed.")
    parser.add_argument("--overlay-output", default=None)
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL_PATH", os.getenv("YOLO_MODEL", "yolo26n-pose.pt")))
    parser.add_argument("--device", default=os.getenv("DEVICE", os.getenv("YOLO_DEVICE", "auto")))
    parser.add_argument("--imgsz", type=int, default=env_int("YOLO_IMGSZ", 640))
    parser.add_argument("--detector-conf", type=float, default=env_float("YOLO_CONF", 0.15))
    parser.add_argument("--action-model", default=os.getenv("MODEL_CHECKPOINT_PATH", os.getenv("ACTION_MODEL") or DEFAULT_ACTION_MODEL))
    parser.add_argument("--action-device", default=os.getenv("ACTION_DEVICE", "auto"))
    parser.add_argument("--action-threshold", type=float, default=env_float("ACTION_THRESHOLD", DEFAULT_FAINT_THRESHOLD), help="Faint probability threshold for LSTM checkpoints with Normal/Faint classes.")
    parser.add_argument("--min-consecutive-faint", type=int, default=env_int("MIN_CONSECUTIVE_FAINT", DEFAULT_MIN_CONSECUTIVE_FAINT), help="Consecutive Faint sequences required before emitting an event.")
    parser.add_argument("--camera-cooldown-seconds", type=float, default=env_float("CAMERA_COOLDOWN_SECONDS", DEFAULT_CAMERA_COOLDOWN_SECONDS), help="Per-camera event cooldown after a Faint event.")
    parser.add_argument("--event-severity", default="HIGH")
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--frame-queue-maxsize", type=int, default=env_int("FRAME_QUEUE_MAXSIZE", 3))
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=env_int("SEQUENCE_LENGTH", DEFAULT_LSTM_SEQUENCE_LENGTH))
    parser.add_argument("--sequence-stride", type=int, default=env_int("SEQUENCE_STRIDE", DEFAULT_LSTM_SEQUENCE_STRIDE))
    parser.add_argument("--cheap-filter-enabled", action=argparse.BooleanOptionalAction, default=os.getenv("CHEAP_FILTER_ENABLED", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--cheap-filter-slope-ratio", type=float, default=env_float("CHEAP_FILTER_SLOPE_RATIO", 1.3))
    parser.add_argument("--cheap-filter-min-keypoint-conf", type=float, default=env_float("CHEAP_FILTER_MIN_KEYPOINT_CONF", 0.25))
    parser.add_argument("--cheap-filter-min-bbox-area-ratio", type=float, default=env_float("CHEAP_FILTER_MIN_BBOX_AREA_RATIO", 0.005))
    parser.add_argument("--cheap-filter-min-center-drop-ratio", type=float, default=env_float("CHEAP_FILTER_MIN_CENTER_DROP_RATIO", 0.03))
    parser.add_argument("--cheap-filter-min-aspect-ratio-growth", type=float, default=env_float("CHEAP_FILTER_MIN_ASPECT_RATIO_GROWTH", 0.20))
    parser.add_argument("--cheap-filter-min-risk-score", type=float, default=env_float("CHEAP_FILTER_MIN_RISK_SCORE", 1.0))
    parser.add_argument("--resize-size", type=int, default=env_int("RESIZE_SIZE", 224))
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default=os.getenv("TRACKING_MODE", "auto"))
    parser.add_argument("--track-thresh", type=float, default=env_float("TRACK_THRESH", 0.10))
    parser.add_argument("--match-thresh", "--tracker-iou-threshold", dest="match_thresh", type=float, default=env_float("TRACK_IOU_THRESHOLD", 0.20))
    parser.add_argument("--track-buffer", type=int, default=env_int("TRACK_BUFFER", 90))
    parser.add_argument("--frame-rate", type=int, default=env_int("TRACK_FRAME_RATE", env_int("FRAME_RATE", 30)))
    parser.add_argument("--min-box-area", type=float, default=env_float("MIN_BOX_AREA", 100.0))
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=env_float("BBOX_SMOOTHING_ALPHA", 0.60))
    parser.add_argument("--track-max-missing-seconds", type=float, default=env_float("TRACK_MAX_MISSING_SECONDS", 4.0))
    parser.add_argument("--center-match-ratio", type=float, default=env_float("CENTER_MATCH_RATIO", 0.70))
    parser.add_argument("--tracking-grace-period-seconds", type=float, default=env_float("TRACKING_GRACE_PERIOD_SECONDS", env_float("TRACK_MAX_MISSING_SECONDS", 4.0)))
    parser.add_argument("--tracking-relink-iou-threshold", type=float, default=env_float("TRACKING_RELINK_IOU_THRESHOLD", 0.30))
    parser.add_argument("--tracking-relink-center-ratio", type=float, default=env_float("TRACKING_RELINK_CENTER_RATIO", 0.70))
    parser.add_argument("--tracking-relink-max-time-gap-seconds", type=float, default=env_float("TRACKING_RELINK_MAX_TIME_GAP_SECONDS", 2.0))
    parser.add_argument("--person-session-reconnect", action=argparse.BooleanOptionalAction, default=os.getenv("PERSON_SESSION_RECONNECT", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--person-session-reconnect-max-missing-seconds", type=float, default=env_float("PERSON_SESSION_RECONNECT_MAX_MISSING_SECONDS", 3.0))
    parser.add_argument("--pose-debug", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_DEBUG", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-debug-summary-every-n", type=int, default=env_int("POSE_DEBUG_SUMMARY_EVERY_N", 60))
    parser.add_argument("--pose-min-keypoint-confidence", type=float, default=env_float("POSE_MIN_KEYPOINT_CONFIDENCE", 0.25))
    parser.add_argument("--pose-debug-save-images", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_DEBUG_SAVE_IMAGES", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-debug-image-dir", default=os.getenv("POSE_DEBUG_IMAGE_DIR", "runs/pose_debug"))
    parser.add_argument("--pose-debug-image-every-n", type=int, default=env_int("POSE_DEBUG_IMAGE_EVERY_N", 300))
    parser.add_argument("--pose-tracking-diag-jsonl", action=argparse.BooleanOptionalAction, default=os.getenv("POSE_TRACKING_DIAG_JSONL", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--pose-tracking-diag-jsonl-path", default=os.getenv("POSE_TRACKING_DIAG_JSONL_PATH", "runs/diagnostics/pose_tracking_diag.jsonl"))
    return parser.parse_args(argv)
