from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, assert_never


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_BASE_URL: Final = "http://localhost:8080"
DEFAULT_RTSP_BASE_URL: Final = "rtsp://localhost:8554"
DEFAULT_VIDEO_POOL: Final = "video_pool"

CameraSourceType = Literal["REAL_RTSP", "SIMULATED_RTSP"]


class RawCamera(TypedDict, total=False):
    cameraId: int
    cameraLoginId: str
    rtspUrl: str | None
    sourceType: str | None
    assignedVideoPath: str | None
    aiEnabled: bool
    status: str


@dataclass(frozen=True, slots=True)
class RegisteredCamera:
    camera_id: str
    camera_login_id: str
    rtsp_url: str | None
    source_type: CameraSourceType
    assigned_video_path: str | None


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    backend_base_url: str
    backend_token: str | None
    rtsp_base_url: str
    video_pool: Path
    overlay_host: str
    overlay_base_port: int
    python_executable: str
    publisher: str
    mqtt_host: str | None
    mqtt_port: int | None
    mqtt_topic: str | None
    mqtt_client_id_prefix: str
    mqtt_username: str | None
    mqtt_password: str | None
    detector_mode: str
    yolo_model: str
    device: str
    action_model: str | None
    action_device: str
    action_threshold: float | None
    classifier_input: str
    sequence_length: int
    sequence_stride: int
    tracking_mode: str
    print_events: bool
    dry_run: bool


def camera_rtsp_url(rtsp_base_url: str, camera_login_id: str) -> str:
    return f"{rtsp_base_url.rstrip('/')}/{camera_login_id}"


def parse_camera(raw: RawCamera) -> RegisteredCamera | None:
    if not raw.get("aiEnabled", True):
        return None
    if raw.get("status") not in (None, "ACTIVE"):
        return None

    login_id = str(raw.get("cameraLoginId") or "").strip()
    if not login_id:
        return None

    raw_source_type = raw.get("sourceType") or "REAL_RTSP"
    match raw_source_type:
        case "REAL_RTSP" | "SIMULATED_RTSP":
            source_type: CameraSourceType = raw_source_type
        case _:
            print(
                f"[registered-cameras][warning] unsupported sourceType={raw_source_type!r} for {login_id}; skipping",
                file=sys.stderr,
                flush=True,
            )
            return None

    return RegisteredCamera(
        camera_id=str(raw.get("cameraId") or login_id),
        camera_login_id=login_id,
        rtsp_url=raw.get("rtspUrl"),
        source_type=source_type,
        assigned_video_path=raw.get("assignedVideoPath"),
    )


def load_active_cameras(backend_base_url: str, backend_token: str | None) -> list[RegisteredCamera]:
    url = f"{backend_base_url.rstrip('/')}/api/cameras/active"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if backend_token:
        request.add_header("Authorization", f"Bearer {backend_token}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch active cameras from {url}: {exc}") from exc

    payload = json.loads(body)
    if not isinstance(payload, list):
        raise RuntimeError(f"expected active camera list from {url}, got {type(payload).__name__}")

    cameras: list[RegisteredCamera] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        parsed = parse_camera(item)
        if parsed is not None:
            cameras.append(parsed)
    return cameras


def first_video_from_pool(video_pool: Path) -> Path | None:
    if not video_pool.exists():
        return None
    return next(iter(sorted(video_pool.glob("*.mp4"))), None)


def resolve_simulated_video(camera: RegisteredCamera, video_pool: Path) -> Path:
    if camera.assigned_video_path:
        assigned = Path(camera.assigned_video_path)
        if assigned.exists():
            return assigned
        repo_relative = REPO_ROOT / assigned
        if repo_relative.exists():
            return repo_relative

    fallback = first_video_from_pool(video_pool)
    if fallback is not None:
        return fallback

    raise RuntimeError(
        f"no assignedVideoPath or fallback mp4 found for camera_login_id={camera.camera_login_id}"
    )


def build_ffmpeg_command(video_path: Path, rtsp_url: str) -> list[str]:
    return [
        "ffmpeg",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-f",
        "rtsp",
        rtsp_url,
    ]


def build_overlay_command(
    camera: RegisteredCamera,
    rtsp_url: str,
    port: int,
    config: RunnerConfig,
) -> list[str]:
    command = [
        config.python_executable,
        "scripts/serve_ai_overlay.py",
        "--rtsp-url",
        rtsp_url,
        "--camera-id",
        camera.camera_login_id,
        "--camera-login-id",
        camera.camera_login_id,
        "--host",
        config.overlay_host,
        "--port",
        str(port),
        "--publisher",
        config.publisher,
        "--detector-mode",
        config.detector_mode,
        "--yolo-model",
        config.yolo_model,
        "--device",
        config.device,
        "--action-device",
        config.action_device,
        "--classifier-input",
        config.classifier_input,
        "--sequence-length",
        str(config.sequence_length),
        "--sequence-stride",
        str(config.sequence_stride),
        "--tracking-mode",
        config.tracking_mode,
    ]
    optional_pairs = [
        ("--mqtt-host", config.mqtt_host),
        ("--mqtt-port", str(config.mqtt_port) if config.mqtt_port is not None else None),
        ("--mqtt-topic", config.mqtt_topic),
        ("--mqtt-client-id", f"{config.mqtt_client_id_prefix}-{camera.camera_login_id}"),
        ("--mqtt-username", config.mqtt_username),
        ("--mqtt-password", config.mqtt_password),
        ("--action-model", config.action_model),
        ("--action-threshold", str(config.action_threshold) if config.action_threshold is not None else None),
    ]
    for key, value in optional_pairs:
        if value:
            command.extend([key, value])
    if config.print_events:
        command.append("--print-events")
    return command


def input_rtsp_for_camera(camera: RegisteredCamera, config: RunnerConfig) -> tuple[str, list[str] | None]:
    match camera.source_type:
        case "REAL_RTSP":
            if not camera.rtsp_url:
                raise RuntimeError(f"REAL_RTSP camera has no rtspUrl: {camera.camera_login_id}")
            return camera.rtsp_url, None
        case "SIMULATED_RTSP":
            rtsp_url = camera_rtsp_url(config.rtsp_base_url, camera.camera_login_id)
            video_path = resolve_simulated_video(camera, config.video_pool)
            return rtsp_url, build_ffmpeg_command(video_path, rtsp_url)
        case unreachable:
            assert_never(unreachable)
