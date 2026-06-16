from __future__ import annotations

import argparse
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


def run_cameras(cameras: list[RegisteredCamera], config: RunnerConfig) -> None:
    run_camera_sync_loop(cameras, config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI overlay workers for backend-registered cameras.")
    parser.add_argument("--backend-base-url", default=DEFAULT_BACKEND_BASE_URL)
    parser.add_argument("--backend-token", default=None)
    parser.add_argument("--backend-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--rtsp-base-url", default=DEFAULT_RTSP_BASE_URL)
    parser.add_argument("--video-pool", default=DEFAULT_VIDEO_POOL)
    parser.add_argument("--overlay-host", default="0.0.0.0")
    parser.add_argument("--overlay-base-port", type=int, default=8010)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--publisher", choices=["mqtt", "console"], default="mqtt")
    parser.add_argument("--mqtt-host", default=None)
    parser.add_argument("--mqtt-port", type=int, default=None)
    parser.add_argument("--mqtt-topic", default="safety/events")
    parser.add_argument("--mqtt-client-id-prefix", default="strange-ai")
    parser.add_argument("--mqtt-username", default=None)
    parser.add_argument("--mqtt-password", default=None)
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="real")
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-model", default=None)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=None)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--tracking-mode", choices=["auto", "simple", "supervision"], default="supervision")
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rtsp-probe", action="store_true", help="Skip real RTSP preflight before starting AI workers.")
    parser.add_argument("--refresh-interval-seconds", type=float, default=30.0)
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
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    cameras = []
    while True:
        try:
            cameras = load_active_cameras(
                config.backend_base_url,
                config.backend_token,
                timeout_seconds=config.backend_timeout_seconds,
            )
            break
        except Exception as exc:
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
