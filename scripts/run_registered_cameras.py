from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.registered_cameras import (
    ACTIVE_CAMERAS_ENDPOINT,
    DEFAULT_BACKEND_BASE_URL,
    DEFAULT_RTSP_BASE_URL,
    DEFAULT_VIDEO_POOL,
    RegisteredCamera,
    RunnerConfig,
    active_cameras_url,
    load_active_cameras,
)
from ai.simulated_rtsp_sources import DEFAULT_STREAM_DOMAIN
from ai.action.lstm_contract import DEFAULT_LSTM_SEQUENCE_LENGTH, DEFAULT_LSTM_SEQUENCE_STRIDE, DEFAULT_KEYPOINT_INPUT_SIZE, log_lstm_config
from ai.registered_camera_workers import run_camera_sync_loop


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_int(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return int(value)
    return None


def env_optional_str(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())



def run_cameras(cameras: list[RegisteredCamera], config: RunnerConfig) -> None:
    run_camera_sync_loop(cameras, config)


def log_camera_api_config(config: RunnerConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "base_url": config.backend_base_url,
        "endpoint": ACTIVE_CAMERAS_ENDPOINT,
        "url": active_cameras_url(config.backend_base_url),
        "timeout_seconds": config.backend_timeout_seconds,
    }
    print(f"[camera-api-config] {json.dumps(payload, ensure_ascii=False)}", flush=True)
    return payload


def warn_if_multiple_registered_camera_runners(current_pid: int | None = None) -> dict | None:
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    try:
        completed = subprocess.run(
            ["pgrep", "-af", "run_registered_cameras.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if "run_registered_cameras.py" in line and "pgrep -af" not in line]
    if len(lines) <= 1:
        return None
    warning = {
        "messageType": "runner_duplicate_warning",
        "currentPid": current_pid,
        "runnerCount": len(lines),
        "runners": lines,
        "cleanupCommand": "pkill -f 'scripts/run_registered_cameras.py'",
    }
    print(f"[registered-cameras][warning] {json.dumps(warning, ensure_ascii=False)}", flush=True)
    return warning


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI overlay workers for backend-registered cameras.")
    parser.add_argument("--backend-base-url", default=os.getenv("BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL))
    parser.add_argument("--backend-token", default=os.getenv("BACKEND_TOKEN"))
    parser.add_argument("--backend-timeout-seconds", type=float, default=float(os.getenv("BACKEND_TIMEOUT_SECONDS", "10.0")))
    parser.add_argument("--rtsp-base-url", default=os.getenv("MEDIAMTX_RTSP_BASE_URL", os.getenv("RTSP_BASE_URL", DEFAULT_RTSP_BASE_URL)))
    parser.add_argument("--video-pool", default=os.getenv("VIDEO_POOL_DIR", DEFAULT_VIDEO_POOL))
    parser.add_argument("--overlay-host", default=os.getenv("OVERLAY_HOST", "0.0.0.0"))
    parser.add_argument("--overlay-base-port", type=int, default=int(os.getenv("OVERLAY_BASE_PORT", "8010")))
    parser.add_argument("--overlay-public-base-url", default=os.getenv("OVERLAY_PUBLIC_BASE_URL"))
    parser.add_argument(
        "--overlay-report-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("AI_OVERLAY_REPORT_ENABLED", True),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--publisher", choices=["mqtt", "console"], default=os.getenv("EVENT_PUBLISHER", "mqtt"))
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT")) if os.getenv("MQTT_PORT") else None)
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_TOPIC"))
    parser.add_argument("--mqtt-camera-topic", default=os.getenv("MQTT_CAMERA_TOPIC", "camera"))
    parser.add_argument("--mqtt-event-topic", default=os.getenv("MQTT_EVENT_TOPIC", os.getenv("MQTT_TOPIC", "event")))
    parser.add_argument("--mqtt-status-topic", default=os.getenv("MQTT_STATUS_TOPIC", "safety/cameras/status"))
    parser.add_argument("--mqtt-client-id-prefix", default=os.getenv("MQTT_CLIENT_ID_PREFIX", "strange-ai"))
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--detector-mode", choices=["real", "mock"], default=os.getenv("DETECTOR_MODE", "real"))
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL_PATH", os.getenv("YOLO_MODEL", "yolo26n-pose.pt")))
    parser.add_argument("--device", default=os.getenv("DEVICE", os.getenv("YOLO_DEVICE", "auto")))
    parser.add_argument("--detector-conf", type=float, default=float(os.getenv("DETECTOR_CONF", "0.15")))
    parser.add_argument("--action-model", default=os.getenv("MODEL_CHECKPOINT_PATH", os.getenv("ACTION_MODEL")))
    parser.add_argument("--action-device", default=os.getenv("ACTION_DEVICE", os.getenv("DEVICE", "auto")))
    parser.add_argument("--action-threshold", type=float, default=float(os.getenv("ACTION_THRESHOLD")) if os.getenv("ACTION_THRESHOLD") else None)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default=os.getenv("CLASSIFIER_INPUT", "keypoints"))
    parser.add_argument("--sequence-length", type=int, default=int(os.getenv("SEQUENCE_LENGTH", str(DEFAULT_LSTM_SEQUENCE_LENGTH))))
    parser.add_argument("--sequence-stride", type=int, default=int(os.getenv("SEQUENCE_STRIDE", str(DEFAULT_LSTM_SEQUENCE_STRIDE))))
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default=os.getenv("TRACKING_MODE", "supervision"))
    parser.add_argument("--track-thresh", type=float, default=float(os.getenv("TRACK_THRESH", "0.10")))
    parser.add_argument("--match-thresh", "--tracker-iou-threshold", dest="match_thresh", type=float, default=float(os.getenv("TRACK_IOU_THRESHOLD", "0.20")))
    parser.add_argument("--track-buffer", type=int, default=int(os.getenv("TRACK_BUFFER", "90")))
    parser.add_argument("--frame-rate", type=int, default=int(os.getenv("TRACK_FRAME_RATE", os.getenv("FRAME_RATE", "30"))))
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=float(os.getenv("BBOX_SMOOTHING_ALPHA", "0.60")))
    parser.add_argument("--tracking-grace-period-seconds", type=float, default=float(os.getenv("TRACKING_GRACE_PERIOD_SECONDS", os.getenv("TRACK_MAX_MISSING_SECONDS", "4.0"))))
    parser.add_argument("--tracking-relink-iou-threshold", type=float, default=float(os.getenv("TRACKING_RELINK_IOU_THRESHOLD", "0.30")))
    parser.add_argument("--tracking-relink-center-ratio", type=float, default=float(os.getenv("TRACKING_RELINK_CENTER_RATIO", "0.70")))
    parser.add_argument("--tracking-relink-max-time-gap-seconds", type=float, default=float(os.getenv("TRACKING_RELINK_MAX_TIME_GAP_SECONDS", "2.0")))
    parser.add_argument("--pose-debug", action=argparse.BooleanOptionalAction, default=env_bool("POSE_DEBUG", False))
    parser.add_argument("--pose-debug-summary-every-n", type=int, default=int(os.getenv("POSE_DEBUG_SUMMARY_EVERY_N", "60")))
    parser.add_argument("--pose-min-keypoint-confidence", type=float, default=float(os.getenv("POSE_MIN_KEYPOINT_CONFIDENCE", "0.25")))
    parser.add_argument("--pose-debug-save-images", action=argparse.BooleanOptionalAction, default=env_bool("POSE_DEBUG_SAVE_IMAGES", False))
    parser.add_argument("--pose-debug-image-dir", default=os.getenv("POSE_DEBUG_IMAGE_DIR", "runs/pose_debug"))
    parser.add_argument("--pose-debug-image-every-n", type=int, default=int(os.getenv("POSE_DEBUG_IMAGE_EVERY_N", "300")))
    parser.add_argument("--pose-tracking-diag-jsonl", action=argparse.BooleanOptionalAction, default=env_bool("POSE_TRACKING_DIAG_JSONL", False))
    parser.add_argument("--pose-tracking-diag-jsonl-path", default=os.getenv("POSE_TRACKING_DIAG_JSONL_PATH", "runs/diagnostics/pose_tracking_diag.jsonl"))
    parser.add_argument(
        "--tracking-stability-fallback",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACKING_STABILITY_FALLBACK", False),
        help="Use a lightweight bbox continuity tracker after supervision to stabilize final track_id values.",
    )
    parser.add_argument(
        "--tracking-stability-fallback-camera-ids",
        default=os.getenv("TRACKING_STABILITY_FALLBACK_CAMERA_IDS", ""),
        help="Comma-separated cameraLoginIds that should use tracking stability fallback without enabling it globally.",
    )
    parser.add_argument(
        "--mjpeg-debug",
        action=argparse.BooleanOptionalAction,
        default=env_bool("AI_MJPEG_DEBUG", False),
        help="Open per-worker debug MJPEG/health HTTP ports such as 8010-8013.",
    )
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rtsp-probe", action="store_true", help="Skip real RTSP preflight before starting AI workers.")
    parser.add_argument("--refresh-interval-seconds", type=float, default=float(os.getenv("CAMERA_POLL_INTERVAL_SECONDS", "30.0")))
    parser.add_argument("--skip-simulated-ffmpeg", action="store_true", help="Skip spawning internal ffmpeg for simulated cameras.")
    parser.add_argument("--domain", default=os.getenv("VIDEO_DOMAIN", DEFAULT_STREAM_DOMAIN))
    parser.add_argument("--label", default=os.getenv("VIDEO_LABEL"))
    parser.add_argument("--video-filter", default=os.getenv("VIDEO_FILTER"))
    parser.add_argument(
        "--selected-track-id",
        "--preferred-track-id",
        dest="selected_track_id",
        type=int,
        default=env_optional_int(
            "AI_SELECTED_TRACK_ID",
            "SELECTED_TRACK_ID",
            "AI_PREFERRED_TRACK_ID",
            "PREFERRED_TRACK_ID",
        ),
    )
    parser.add_argument(
        "--selected-track-mode",
        default=env_optional_str("AI_SELECTED_TRACK_MODE", "SELECTED_TRACK_MODE") or "strict",
    )
    parser.add_argument(
        "--selected-track-missing-frames",
        type=int,
        default=env_optional_int("AI_SELECTED_TRACK_MISSING_FRAMES", "SELECTED_TRACK_MISSING_FRAMES") or 5,
    )
    return parser.parse_args(argv)



def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        backend_base_url=args.backend_base_url,
        backend_token=args.backend_token,
        backend_timeout_seconds=args.backend_timeout_seconds,
        rtsp_base_url=args.rtsp_base_url,
        video_pool=Path(args.video_pool),
        overlay_host=args.overlay_host,
        overlay_base_port=args.overlay_base_port,
        python_executable=args.python_executable,
        publisher=args.publisher,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_camera_topic=args.mqtt_camera_topic,
        mqtt_event_topic=args.mqtt_event_topic,
        mqtt_client_id_prefix=args.mqtt_client_id_prefix,
        mqtt_username=args.mqtt_username,
        mqtt_password=args.mqtt_password,
        detector_mode=args.detector_mode,
        yolo_model=args.yolo_model,
        device=args.device,
        detector_conf=args.detector_conf,
        action_model=args.action_model,
        action_device=args.action_device,
        action_threshold=args.action_threshold,
        classifier_input=args.classifier_input,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        tracking_mode=args.tracking_mode,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.frame_rate,
        bbox_smoothing_alpha=args.bbox_smoothing_alpha,
        tracking_grace_period_seconds=args.tracking_grace_period_seconds,
        tracking_relink_iou_threshold=args.tracking_relink_iou_threshold,
        tracking_relink_center_ratio=args.tracking_relink_center_ratio,
        tracking_relink_max_time_gap_seconds=args.tracking_relink_max_time_gap_seconds,
        pose_debug=args.pose_debug,
        pose_debug_summary_every_n=args.pose_debug_summary_every_n,
        pose_min_keypoint_confidence=args.pose_min_keypoint_confidence,
        pose_debug_save_images=args.pose_debug_save_images,
        pose_debug_image_dir=args.pose_debug_image_dir,
        pose_debug_image_every_n=args.pose_debug_image_every_n,
        pose_tracking_diag_jsonl=args.pose_tracking_diag_jsonl,
        pose_tracking_diag_jsonl_path=args.pose_tracking_diag_jsonl_path,
        tracking_stability_fallback=args.tracking_stability_fallback,
        tracking_stability_fallback_camera_ids=split_csv(args.tracking_stability_fallback_camera_ids),
        mjpeg_debug=args.mjpeg_debug,
        print_events=args.print_events,
        dry_run=args.dry_run,
        rtsp_probe_enabled=not args.skip_rtsp_probe,
        refresh_interval_seconds=args.refresh_interval_seconds,
        skip_ffmpeg_spawn=args.skip_simulated_ffmpeg,
        overlay_public_base_url=args.overlay_public_base_url,
        overlay_report_enabled=args.overlay_report_enabled,
        domain=args.domain,
        label=args.label,
        video_filter=args.video_filter,
        selected_track_id=args.selected_track_id,
        selected_track_mode=args.selected_track_mode,
        selected_track_missing_frames=args.selected_track_missing_frames,
        mqtt_status_topic=args.mqtt_status_topic,
    )



def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    warn_if_multiple_registered_camera_runners()
    print(
        "[registered-cameras] sequence config: "
        f"sequence_length={config.sequence_length} "
        f"sequence_stride={config.sequence_stride} "
        "defaults=30/15 stride_is_sequence_start_interval",
        flush=True,
    )
    log_lstm_config("[lstm-config]", config.sequence_length, config.sequence_stride, DEFAULT_KEYPOINT_INPUT_SIZE, "config/cli")
    log_camera_api_config(config)
    cameras = []
    while True:
        try:
            cameras = load_active_cameras(
                config.backend_base_url,
                config.backend_token,
                timeout_seconds=config.backend_timeout_seconds,
            )
            break
        except RuntimeError as exc:
            if config.dry_run:
                print(
                    f"[registered-cameras][warning] failed to load active cameras from {config.backend_base_url}: {exc}. "
                    "Dry-run will continue without backend cameras.",
                    file=sys.stderr,
                    flush=True,
                )
                break
            print(
                f"[registered-cameras][warning] failed to load active cameras from {config.backend_base_url}: {exc}. "
                "Retrying in 5 seconds...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(5)
    if not cameras:
        print("[registered-cameras][warning] no active AI cameras returned by backend", flush=True)
    run_cameras(cameras, config)


if __name__ == "__main__":
    main()
