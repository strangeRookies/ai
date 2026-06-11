from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.registered_cameras import (
    DEFAULT_BACKEND_BASE_URL,
    DEFAULT_RTSP_BASE_URL,
    DEFAULT_VIDEO_POOL,
    REPO_ROOT,
    RegisteredCamera,
    RunnerConfig,
    build_overlay_command,
    input_rtsp_for_camera,
    load_active_cameras,
)


def spawn_process(command: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()


def stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_cameras(cameras: list[RegisteredCamera], config: RunnerConfig) -> None:
    processes: list[subprocess.Popen[str]] = []
    for index, camera in enumerate(cameras):
        rtsp_url, ffmpeg_command = input_rtsp_for_camera(camera, config)
        overlay_command = build_overlay_command(camera, rtsp_url, config.overlay_base_port + index, config)
        print(f"[registered-cameras] {camera.camera_login_id} input={rtsp_url}", flush=True)
        if ffmpeg_command is not None:
            print(f"[registered-cameras] ffmpeg: {' '.join(ffmpeg_command)}", flush=True)
        print(f"[registered-cameras] overlay: {' '.join(overlay_command)}", flush=True)

        if config.dry_run:
            continue
        if ffmpeg_command is not None:
            processes.append(
                spawn_process(
                    ffmpeg_command,
                    REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-ffmpeg.log",
                )
            )
            time.sleep(1)
        processes.append(
            spawn_process(
                overlay_command,
                REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-overlay.log",
            )
        )

    if config.dry_run:
        return

    def shutdown(_signum: int, _frame: object) -> NoReturn:
        stop_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    while True:
        time.sleep(2)
        for process in list(processes):
            if process.poll() is not None:
                stop_processes([item for item in processes if item.poll() is None])
                raise RuntimeError(f"child process exited unexpectedly with code {process.returncode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI overlay workers for backend-registered cameras.")
    parser.add_argument("--backend-base-url", default=DEFAULT_BACKEND_BASE_URL)
    parser.add_argument("--backend-token", default=None)
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
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        backend_base_url=args.backend_base_url,
        backend_token=args.backend_token,
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
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    cameras = load_active_cameras(config.backend_base_url, config.backend_token)
    if not cameras:
        print("[registered-cameras][warning] no active AI cameras returned by backend", flush=True)
        return
    run_cameras(cameras, config)


if __name__ == "__main__":
    main()
