from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.registered_cameras import (
    DEFAULT_BACKEND_BASE_URL,
    DEFAULT_RTSP_BASE_URL,
    DEFAULT_VIDEO_POOL,
    RegisteredCamera,
    RunnerConfig,
    load_active_cameras,
)
from ai.registered_camera_workers import run_camera_sync_loop


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_cameras(cameras: list[RegisteredCamera], config: RunnerConfig) -> None:
    run_camera_sync_loop(cameras, config)


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
    parser.add_argument("--mqtt-client-id-prefix", default=os.getenv("MQTT_CLIENT_ID_PREFIX", "strange-ai"))
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--detector-mode", choices=["real", "mock"], default=os.getenv("DETECTOR_MODE", "real"))
    parser.add_argument("--yolo-model", default=os.getenv("YOLO_MODEL_PATH", os.getenv("YOLO_MODEL", "yolo26n-pose.pt")))
    parser.add_argument("--device", default=os.getenv("DEVICE", os.getenv("YOLO_DEVICE", "auto")))
    parser.add_argument("--action-model", default=os.getenv("MODEL_CHECKPOINT_PATH", os.getenv("ACTION_MODEL")))
    parser.add_argument("--action-device", default=os.getenv("ACTION_DEVICE", os.getenv("DEVICE", "auto")))
    parser.add_argument("--action-threshold", type=float, default=float(os.getenv("ACTION_THRESHOLD")) if os.getenv("ACTION_THRESHOLD") else None)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default=os.getenv("CLASSIFIER_INPUT", "keypoints"))
    parser.add_argument("--sequence-length", type=int, default=int(os.getenv("SEQUENCE_LENGTH", "8")))
    parser.add_argument("--sequence-stride", type=int, default=int(os.getenv("SEQUENCE_STRIDE", "4")))
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default=os.getenv("TRACKING_MODE", "supervision"))
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rtsp-probe", action="store_true", help="Skip real RTSP preflight before starting AI workers.")
    parser.add_argument("--refresh-interval-seconds", type=float, default=float(os.getenv("CAMERA_POLL_INTERVAL_SECONDS", "30.0")))
    parser.add_argument("--skip-simulated-ffmpeg", action="store_true", help="Skip spawning internal ffmpeg for simulated cameras.")
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
        action_model=args.action_model,
        action_device=args.action_device,
        action_threshold=args.action_threshold,
        classifier_input=args.classifier_input,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        tracking_mode=args.tracking_mode,
        print_events=args.print_events,
        dry_run=args.dry_run,
        rtsp_probe_enabled=not args.skip_rtsp_probe,
        refresh_interval_seconds=args.refresh_interval_seconds,
        skip_ffmpeg_spawn=args.skip_simulated_ffmpeg,
        overlay_public_base_url=args.overlay_public_base_url,
        overlay_report_enabled=args.overlay_report_enabled,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    print(
        "[registered-cameras] sequence config: "
        f"sequence_length={config.sequence_length} "
        f"sequence_stride={config.sequence_stride} "
        "defaults=8/4 stride_is_sequence_start_interval",
        flush=True,
    )
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
