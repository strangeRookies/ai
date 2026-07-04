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
    tracking_stability_fallback_enabled,
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
    command: list[str] | None = None
    overlay_log_path: Path | None = None



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
    exit_roi_suffix = _json.dumps(list(camera.exit_roi_configs), sort_keys=True)
    runtime_suffix = _json.dumps(
        {
            "bbox_smoothing_alpha": config.bbox_smoothing_alpha,
            "classifier_input": config.classifier_input,
            "detector_conf": config.detector_conf,
            "detector_mode": config.detector_mode,
            "device": config.device,
            "match_thresh": config.match_thresh,
            "mjpeg_debug": config.mjpeg_debug,
            "mqtt_camera_topic": config.mqtt_camera_topic,
            "mqtt_event_topic": config.mqtt_event_topic,
            "mqtt_host": config.mqtt_host,
            "mqtt_port": config.mqtt_port,
            "mqtt_status_topic": config.mqtt_status_topic,
            "pose_debug": config.pose_debug,
            "pose_debug_image_every_n": config.pose_debug_image_every_n,
            "pose_debug_save_images": config.pose_debug_save_images,
            "pose_debug_summary_every_n": config.pose_debug_summary_every_n,
            "pose_min_keypoint_confidence": config.pose_min_keypoint_confidence,
            "pose_tracking_diag_jsonl": config.pose_tracking_diag_jsonl,
            "pose_tracking_diag_jsonl_path": config.pose_tracking_diag_jsonl_path,
            "selected_track_id": config.selected_track_id,
            "selected_track_missing_frames": config.selected_track_missing_frames,
            "selected_track_mode": config.selected_track_mode,
            "sequence_length": config.sequence_length,
            "sequence_stride": config.sequence_stride,
            "track_buffer": config.track_buffer,
            "track_thresh": config.track_thresh,
            "frame_rate": config.frame_rate,
            "tracking_grace_period_seconds": config.tracking_grace_period_seconds,
            "tracking_mode": config.tracking_mode,
            "tracking_relink_center_ratio": config.tracking_relink_center_ratio,
            "tracking_relink_iou_threshold": config.tracking_relink_iou_threshold,
            "tracking_relink_max_time_gap_seconds": config.tracking_relink_max_time_gap_seconds,
            "tracking_stability_fallback": tracking_stability_fallback_enabled(camera, config),
            "webrtc_sync_base_port": config.webrtc_sync_base_port,
            "webrtc_sync_enabled": config.webrtc_sync_enabled,
            "webrtc_sync_host": config.webrtc_sync_host,
            "yolo_model": config.yolo_model,
        },
        sort_keys=True,
    )
    match camera.source_type:
        case "REAL_RTSP":
            return f"REAL_RTSP:{camera.rtsp_url or ''}|roi:{roi_suffix}|exit_roi:{exit_roi_suffix}|runtime:{runtime_suffix}"
        case "SIMULATED_RTSP":
            return (
                f"SIMULATED_RTSP:{camera_rtsp_url(config.rtsp_base_url, camera.camera_login_id)}:"
                f"{camera.assigned_video_path or ''}|roi:{roi_suffix}|exit_roi:{exit_roi_suffix}|runtime:{runtime_suffix}"
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

 #FFmpeg 프로세스 제어
 #카메라 원본 스트림이나 시뮬레이션용 MP4 비디오 파일을 RTSP 스트림으로 변환해 MediaMTX(rtsp://localhost:8554/{cameraLoginId})에 공급
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

    # Ensure any existing duplicate worker process is terminated/killed first
    from ai.worker_registry import force_kill_existing_worker
    force_kill_existing_worker(camera.camera_login_id)

    if config.dry_run:
        return CameraWorker(
            processes=[],
            overlay_port=port,
            source_signature=camera_source_signature(camera, config),
            camera_login_id=camera.camera_login_id,
            rtsp_url=rtsp_url,
            command=overlay_command,
            overlay_log_path=REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-overlay.log",
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
    if config.webrtc_sync_token:
        overlay_env["AI_WEBRTC_SYNC_TOKEN"] = config.webrtc_sync_token
    overlay_log_path = REPO_ROOT / "runs" / "registered_cameras" / f"{camera.camera_login_id}-overlay.log"
    processes.append(
        spawn_process(
            overlay_command,
            overlay_log_path,
            env=overlay_env,
        )
    )
    if config.mjpeg_debug:
        report_overlay_status(camera, rtsp_url, port, config, "RUNNING", getattr(processes[-1], "pid", None))
    elif config.overlay_report_enabled:
        print(
            f"[registered-cameras] overlay HTTP disabled for camera={camera.camera_login_id}; "
            "metadata is published by MQTT and video stays on MediaMTX WebRTC/HLS",
            flush=True,
        )
    return CameraWorker(
        processes=processes,
        overlay_port=port,
        source_signature=camera_source_signature(camera, config),
        camera_login_id=camera.camera_login_id,
        rtsp_url=rtsp_url,
        command=overlay_command,
        overlay_log_path=overlay_log_path,
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
        from ai.worker_registry import get_active_worker_count, get_active_publisher_count
        print(
            f"[registered-cameras] Active camera workers: {get_active_worker_count()} "
            f"| Active simulated publishers: {get_active_publisher_count()}",
            flush=True
        )
        for camera_login_id, worker in list(workers.items()):
            if worker_has_exited(worker):
                cmd_text = safe_command_text(worker.command) if getattr(worker, "command", None) else "unknown"
                masked_url = redact_url(worker.rtsp_url) if worker.rtsp_url else "unknown"
                print(
                    f"[registered-cameras][warning] worker exited; stopping camera={camera_login_id} "
                    f"| cameraLoginId={camera_login_id} | streamId={camera_login_id} "
                    f"| rtsp_url={masked_url} | overlay_log={worker.overlay_log_path} | command={cmd_text}",
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
