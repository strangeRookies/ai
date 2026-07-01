from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NoReturn

from ai.command_logging import safe_command_text
from ai.overlay_ports import next_overlay_port
from ai.overlay_registry_client import report_overlay_status, report_overlay_stopped
from ai.publishers.camera_status_publisher import CameraStatusPublisher
from ai.publishers.event_publisher import create_event_publisher
from ai.registered_cameras import (
    REPO_ROOT,
    RegisteredCamera,
    RunnerConfig,
    build_overlay_command,
    camera_rtsp_url,
    input_rtsp_for_camera,
    load_active_cameras,
)
from ai.streams.video_reader import VideoReader
from stream.rtsp_reader import redact_url

CameraFailureStatus = Literal["DISCONNECTED", "ERROR"]


@dataclass(slots=True)
class CameraWorker:
    processes: list[subprocess.Popen[str]]
    overlay_port: int
    source_signature: str
    camera_login_id: str | None = None
    rtsp_url: str | None = None


def spawn_process(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
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


def worker_has_exited(worker: CameraWorker) -> bool:
    return any(process.poll() is not None for process in worker.processes)


def camera_source_signature(camera: RegisteredCamera, config: RunnerConfig) -> str:
    import json as _json
    roi_suffix = _json.dumps(list(camera.roi_configs), sort_keys=True)
    match camera.source_type:
        case "REAL_RTSP":
            return f"REAL_RTSP:{camera.rtsp_url or ''}|roi:{roi_suffix}"
        case "SIMULATED_RTSP":
            return (
                f"SIMULATED_RTSP:{camera_rtsp_url(config.rtsp_base_url, camera.camera_login_id)}:"
                f"{camera.assigned_video_path or ''}|roi:{roi_suffix}"
            )


def rtsp_has_readable_frame(rtsp_url: str) -> bool:
    try:
        with VideoReader(rtsp_url) as reader:
            return reader.read() is not None
    except RuntimeError as exc:
        print(
            f"[registered-cameras][warning] RTSP probe failed: {redact_url(rtsp_url)} ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return False


def publish_unavailable_camera_status(
    camera: RegisteredCamera,
    rtsp_url: str | None,
    config: RunnerConfig,
    status: CameraFailureStatus,
    reason: str,
) -> None:
    publisher, _publisher_mode = create_event_publisher(
        SimpleNamespace(
            dry_run=config.dry_run,
            publisher=config.publisher,
            mqtt_host=config.mqtt_host,
            mqtt_port=config.mqtt_port,
            mqtt_topic=config.mqtt_topic,
            mqtt_camera_topic=config.mqtt_camera_topic,
            mqtt_event_topic=config.mqtt_event_topic,
            mqtt_client_id=f"{config.mqtt_client_id_prefix}-{camera.camera_login_id}-status",
            mqtt_username=config.mqtt_username,
            mqtt_password=config.mqtt_password,
        )
    )
    status_publisher = CameraStatusPublisher(
        mqtt_publisher=publisher,
        camera_login_id=camera.camera_login_id,
        rtsp_url=rtsp_url,
    )
    match status:
        case "DISCONNECTED":
            status_publisher.notify_disconnected(reason=reason)
        case "ERROR":
            status_publisher.notify_error(reason=reason)
    close = getattr(publisher, "close", None)
    if close:
        close()


def start_camera_worker(camera: RegisteredCamera, config: RunnerConfig, port: int) -> CameraWorker | None:
    processes: list[subprocess.Popen[str]] = []
    try:
        rtsp_url, ffmpeg_command = input_rtsp_for_camera(camera, config)
    except RuntimeError as exc:
        print(
            f"[registered-cameras][warning] invalid camera configuration camera={camera.camera_login_id}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        publish_unavailable_camera_status(camera, camera.rtsp_url, config, "ERROR", "INVALID_CAMERA_CONFIG")
        return None
    if config.rtsp_probe_enabled and ffmpeg_command is None and not config.dry_run:
        if not rtsp_has_readable_frame(rtsp_url):
            publish_unavailable_camera_status(camera, rtsp_url, config, "ERROR", "RTSP_PROBE_FAILED")
            print(
                f"[registered-cameras][warning] skipping AI worker for unreachable RTSP camera={camera.camera_login_id}",
                file=sys.stderr,
                flush=True,
            )
            return None
    overlay_command = build_overlay_command(camera, rtsp_url, port, config)
    print(f"[registered-cameras] {camera.camera_login_id} input={redact_url(rtsp_url)}", flush=True)
    if ffmpeg_command is not None:
        print(f"[registered-cameras] ffmpeg: {safe_command_text(ffmpeg_command)}", flush=True)
    print(f"[registered-cameras] overlay: {safe_command_text(overlay_command)}", flush=True)

    if config.dry_run:
        return CameraWorker(
            processes=[],
            overlay_port=port,
            source_signature=camera_source_signature(camera, config),
            camera_login_id=camera.camera_login_id,
            rtsp_url=rtsp_url,
        )
    if ffmpeg_command is not None:
        processes.append(
            spawn_process(
                ffmpeg_command,
                REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-ffmpeg.log",
            )
        )
        time.sleep(1)
    overlay_env = os.environ.copy()
    overlay_env["RTSP_URL"] = rtsp_url
    if config.mqtt_password:
        overlay_env["MQTT_PASSWORD"] = config.mqtt_password
    processes.append(
        spawn_process(
            overlay_command,
            REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-overlay.log",
            env=overlay_env,
        )
    )
    report_overlay_status(camera, rtsp_url, port, config, "RUNNING", getattr(processes[-1], "pid", None))
    return CameraWorker(
        processes=processes,
        overlay_port=port,
        source_signature=camera_source_signature(camera, config),
        camera_login_id=camera.camera_login_id,
        rtsp_url=rtsp_url,
    )


def sync_camera_workers(
    workers: dict[str, CameraWorker],
    cameras: list[RegisteredCamera],
    config: RunnerConfig,
) -> None:
    active_cameras = {camera.camera_login_id: camera for camera in cameras}
    for camera_login_id in list(workers):
        if camera_login_id not in active_cameras:
            print(f"[registered-cameras] stopping inactive camera={camera_login_id}", flush=True)
            stop_processes(workers[camera_login_id].processes)
            report_overlay_stopped(camera_login_id, workers[camera_login_id].rtsp_url, workers[camera_login_id].overlay_port, config)
            del workers[camera_login_id]

    for camera in active_cameras.values():
        existing_worker = workers.get(camera.camera_login_id)
        if existing_worker is not None and existing_worker.source_signature == camera_source_signature(camera, config):
            continue
        if existing_worker is not None:
            print(f"[registered-cameras] restarting updated camera={camera.camera_login_id}", flush=True)
            preferred_port = existing_worker.overlay_port
            stop_processes(existing_worker.processes)
            report_overlay_stopped(camera.camera_login_id, existing_worker.rtsp_url, existing_worker.overlay_port, config)
            del workers[camera.camera_login_id]
        else:
            preferred_port = None
        worker = start_camera_worker(camera, config, next_overlay_port(workers, config, preferred_port=preferred_port))
        if worker is not None:
            workers[camera.camera_login_id] = worker


def stop_all_workers(workers: dict[str, CameraWorker], config: RunnerConfig) -> None:
    for camera_login_id, worker in workers.items():
        stop_processes(worker.processes)
        report_overlay_stopped(camera_login_id, worker.rtsp_url, worker.overlay_port, config)
    workers.clear()


def run_camera_sync_loop(cameras: list[RegisteredCamera], config: RunnerConfig) -> None:
    workers: dict[str, CameraWorker] = {}
    sync_camera_workers(workers, cameras, config)
    if config.dry_run:
        return

    def shutdown(_signum: int, _frame: object) -> NoReturn:
        stop_all_workers(workers, config)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    next_refresh_at = time.monotonic() + config.refresh_interval_seconds
    while True:
        time.sleep(2)
        for camera_login_id, worker in list(workers.items()):
            if worker_has_exited(worker):
                print(
                    f"[registered-cameras][warning] worker exited; stopping camera={camera_login_id}",
                    file=sys.stderr,
                    flush=True,
                )
                stop_processes(worker.processes)
                report_overlay_stopped(camera_login_id, worker.rtsp_url, worker.overlay_port, config)
                del workers[camera_login_id]
        if time.monotonic() < next_refresh_at:
            continue
        next_refresh_at = time.monotonic() + config.refresh_interval_seconds
        try:
            latest_cameras = load_active_cameras(
                config.backend_base_url,
                config.backend_token,
                timeout_seconds=config.backend_timeout_seconds,
            )
        except RuntimeError as exc:
            print(f"[registered-cameras][warning] active camera refresh failed: {exc}", file=sys.stderr, flush=True)
            continue
        sync_camera_workers(workers, latest_cameras, config)
