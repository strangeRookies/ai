from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypedDict, assert_never

from ai.camera_input_safety import (
    assigned_video_candidates,
    is_path_under,
    is_safe_camera_login_id,
    resolved_video_pool,
)
from ai.ffmpeg_command import build_ffmpeg_command


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_BASE_URL: Final = "http://localhost:18080"
ACTIVE_CAMERAS_ENDPOINT: Final = "/api/cameras/active"
DEFAULT_RTSP_BASE_URL: Final = "rtsp://localhost:8554"
DEFAULT_VIDEO_POOL: Final = "video_pool"

FAINT_SCENARIO_TYPES: Final = {"FALL_BED", "COLLAPSE", "SYNCOPE"}
EXIT_SCENARIO_TYPES: Final = {"EXIT"}

CameraSourceType = Literal["REAL_RTSP", "SIMULATED_RTSP"]


class RawRoiConfig(TypedDict, total=False):
    roiConfigId: int
    scenarioId: int
    scenarioType: str
    polygonPoints: str


class RawCamera(TypedDict, total=False):
    cameraId: int
    cameraLoginId: str
    rtspUrl: str | None
    sourceType: str | None
    assignedVideoPath: str | None
    aiEnabled: bool
    status: str
    roiConfigs: list[RawRoiConfig]


class ApiEnvelope(TypedDict, total=False):
    success: bool
    message: str
    data: list[RawCamera]


@dataclass(frozen=True, slots=True)
class RegisteredCamera:
    """Backend camera DTO를 AI worker가 쓰는 안정적인 내부 모델로 정규화한 값.

    Backend의 `cameraId`는 DB PK이고, AI 파이프라인과 MQTT/frontend overlay의
    외부 식별자는 `cameraLoginId`다. 그래서 이 객체 안에서는
    `camera_login_id`를 worker key, stream path, log key로 일관되게 사용한다.
    """

    camera_id: str
    camera_login_id: str
    rtsp_url: str | None
    source_type: CameraSourceType
    assigned_video_path: str | None
    roi_configs: tuple[dict, ...] = field(default_factory=tuple)
    exit_roi_configs: tuple[dict, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """등록 카메라 runner가 하위 overlay worker에 전달하는 실행 계약.

    이 설정은 `scripts/run_registered_cameras.py`의 CLI/env에서 만들어지고,
    `build_overlay_command()`가 per-camera `serve_ai_overlay.py` 프로세스 인자로
    변환한다. 기존 MQTT payload/backend schema를 바꾸지 않기 위해, 새 진단
    옵션도 전부 이 config를 거쳐 worker 인자로만 전달한다.
    """

    backend_base_url: str
    backend_token: str | None
    backend_timeout_seconds: float
    rtsp_base_url: str
    video_pool: Path
    overlay_host: str
    overlay_base_port: int
    python_executable: str
    publisher: str
    mqtt_host: str | None
    mqtt_port: int | None
    mqtt_topic: str | None
    mqtt_camera_topic: str | None
    mqtt_event_topic: str | None
    mqtt_client_id_prefix: str
    mqtt_username: str | None
    mqtt_password: str | None
    detector_mode: str
    yolo_model: str
    device: str
    detector_conf: float
    action_model: str | None
    action_device: str
    action_threshold: float | None
    classifier_input: str
    sequence_length: int
    sequence_stride: int
    tracking_mode: str
    track_thresh: float
    match_thresh: float
    track_buffer: int
    bbox_smoothing_alpha: float
    print_events: bool
    dry_run: bool
    rtsp_probe_enabled: bool
    refresh_interval_seconds: float
    tracking_stability_fallback: bool = False
    tracking_stability_fallback_camera_ids: tuple[str, ...] = ()
    mjpeg_debug: bool = False
    webrtc_sync_enabled: bool = False
    webrtc_sync_host: str = "0.0.0.0"
    webrtc_sync_base_port: int = 8090
    webrtc_sync_token: str | None = None
    mjpeg_enabled: bool = False
    mjpeg_fps: float = 8.0
    mjpeg_width: int = 640
    mjpeg_height: int = 360
    mjpeg_jpeg_quality: int = 70
    mjpeg_base_path: str = "/mjpeg"
    mjpeg_enable_overlay: bool = True
    skip_ffmpeg_spawn: bool = False
    overlay_public_base_url: str | None = None
    overlay_report_enabled: bool = False
    domain: str | None = None
    label: str | None = None
    video_filter: str | None = None
    selected_track_id: int | None = None
    selected_track_mode: str = "strict"
    selected_track_missing_frames: int = 5
    mqtt_status_topic: str | None = None
    frame_rate: int = 30
    tracking_grace_period_seconds: float = 4.0
    tracking_relink_iou_threshold: float = 0.30
    tracking_relink_center_ratio: float = 0.70
    tracking_relink_max_time_gap_seconds: float = 2.0
    pose_debug: bool = False
    pose_debug_summary_every_n: int = 60
    pose_min_keypoint_confidence: float = 0.25
    pose_debug_save_images: bool = False
    pose_debug_image_dir: str = "runs/pose_debug"
    pose_debug_image_every_n: int = 300
    pose_tracking_diag_jsonl: bool = False
    pose_tracking_diag_jsonl_path: str = "runs/diagnostics/pose_tracking_diag.jsonl"



def normalize_camera_login_id(login_id: str) -> str:
    # Normalize cam1 -> cam_01, cam_1 -> cam_01, cam01 -> cam_01
    match = re.match(r'^cam_?(\d+)$', login_id, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        return f"cam_{num:02d}"
    return login_id


def camera_rtsp_url(rtsp_base_url: str, camera_login_id: str) -> str:
    normalized_id = normalize_camera_login_id(camera_login_id)
    return f"{rtsp_base_url.rstrip('/')}/{normalized_id}"


def active_cameras_url(backend_base_url: str) -> str:
    return f"{backend_base_url.rstrip('/')}{ACTIVE_CAMERAS_ENDPOINT}"


def parse_camera(raw: RawCamera) -> RegisteredCamera | None:
    """Backend `/api/cameras/active` 항목 하나를 실행 가능한 카메라로 변환한다.

    여기서 필터링되는 항목은 worker를 만들지 않는다. 즉 `aiEnabled=false`,
    비활성 status, 빈/위험한 `cameraLoginId`, 지원하지 않는 `sourceType`은
    조용한 오동작 대신 명시적으로 제외된다. ROI는 LSTM 이벤트 판단에 필요한
    faint 계열과 exit 계열만 분리해 downstream으로 넘긴다.
    """

    if not raw.get("aiEnabled", True):
        return None
    if raw.get("status") not in (None, "ACTIVE"):
        return None

    login_id = str(raw.get("cameraLoginId") or "").strip()
    if not login_id:
        return None
    
    # cam1, cam_1, cam01처럼 등록된 값을 cam_01 형태로 맞춰 stream path와
    # 로그 key가 카메라 수에 따라 동적으로 정렬되게 한다.
    login_id = normalize_camera_login_id(login_id)
    
    if not is_safe_camera_login_id(login_id):
        print(
            f"[registered-cameras][warning] unsafe cameraLoginId={login_id!r}; skipping",
            file=sys.stderr,
            flush=True,
        )
        return None

    raw_source_type = raw.get("sourceType") or "SIMULATED_RTSP"
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

    raw_rois = raw.get("roiConfigs") or []
    faint_rois = [r for r in raw_rois if isinstance(r, dict) and r.get("scenarioType") in FAINT_SCENARIO_TYPES]
    exit_rois  = [r for r in raw_rois if isinstance(r, dict) and r.get("scenarioType") in EXIT_SCENARIO_TYPES]

    def _make_roi_tuple(rois):
        return tuple(
            {
                "roiConfigId": r.get("roiConfigId"),
                "scenarioId": r.get("scenarioId"),
                "scenarioType": r.get("scenarioType"),
                "polygonPoints": r.get("polygonPoints", ""),
            }
            for r in rois
            if r.get("polygonPoints")
        )

    return RegisteredCamera(
        camera_id=str(raw.get("cameraId") or login_id),
        camera_login_id=login_id,
        rtsp_url=raw.get("rtspUrl"),
        source_type=source_type,
        assigned_video_path=raw.get("assignedVideoPath"),
        roi_configs=_make_roi_tuple(faint_rois),
        exit_roi_configs=_make_roi_tuple(exit_rois),
    )


def load_active_cameras(
    backend_base_url: str,
    backend_token: str | None,
    timeout_seconds: float = 10.0,
) -> list[RegisteredCamera]:
    """Backend active camera API를 호출해 현재 AI 대상 카메라 목록을 가져온다.

    실패 메시지에는 실제 URL, timeout, HTTP status/body 일부를 포함한다. 이 값은
    GPU PC에서 `localhost`, `host.docker.internal`, compose service name을 잘못
    고른 경우를 구분하는 1차 증거가 되므로, 호출 실패를 단순 "offline"으로
    뭉개지 않는다.
    """

    url = active_cameras_url(backend_base_url)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if backend_token:
        request.add_header("Authorization", f"Bearer {backend_token}")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_snippet = ""
        try:
            body_snippet = exc.read().decode("utf-8", errors="replace")[:500]
        except OSError:
            body_snippet = "<unreadable>"
        raise RuntimeError(
            f"failed to fetch active cameras from {url}: HTTP {exc.code} {exc.reason}; "
            f"timeout={timeout_seconds:g}s; response_body={body_snippet!r}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"timed out after {timeout_seconds:g}s while waiting for active cameras from {url}; "
            "check backend logs and database connectivity"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"failed to fetch active cameras from {url}: {exc}; timeout={timeout_seconds:g}s"
        ) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse active cameras JSON from {url}: {exc}; "
            f"timeout={timeout_seconds:g}s; response_body={body[:500]!r}"
        ) from exc
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
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


def resolve_simulated_video(camera: RegisteredCamera, video_pool: Path, config: RunnerConfig | None = None) -> Path:
    """SIMULATED_RTSP 카메라가 송출할 mp4 파일을 결정한다.

    우선순위는 backend가 내려준 `assignedVideoPath`다. 단, 경로 탈출을 막기 위해
    항상 `video_pool` 아래 실제 파일인지 확인한다. assigned video가 없으면
    cameraLoginId 기반 stable index로 pool에서 하나를 고르므로 cam_01~cam_N이
    하드코딩 없이 재시작 후에도 같은 영상을 바라본다.
    """

    if camera.assigned_video_path:
        pool_root = resolved_video_pool(video_pool, REPO_ROOT)
        for candidate in assigned_video_candidates(camera.assigned_video_path, video_pool, REPO_ROOT):
            if candidate.exists() and candidate.is_file():
                if is_path_under(candidate, pool_root):
                    return candidate
                raise RuntimeError(
                    f"assignedVideoPath must stay under video_pool for camera_login_id={camera.camera_login_id}"
                )

    # Load and filter pool videos
    if not video_pool.exists():
        raise RuntimeError(f"video_pool directory does not exist: {video_pool}")
    
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [p for p in video_pool.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    video_files.sort(key=lambda x: x.name)

    if config is not None:
        from ai.simulated_rtsp_sources import filter_video_files
        matching, _ = filter_video_files(video_files, config.domain, config.label, config.video_filter)
        video_files = matching

    if not video_files:
        raise RuntimeError(f"no matching videos found in pool {video_pool}")

    from ai.simulated_rtsp_sources import stable_video_index, estimate_video_metadata
    idx = stable_video_index(camera.camera_login_id, len(video_files))
    assigned = video_files[idx]

    meta = estimate_video_metadata(assigned)
    print(
        f"[registered-cameras] Mapping camera_login_id={camera.camera_login_id} "
        f"to video={assigned.name} (domain={meta['domain']}, label={meta['label']}, source={meta['source']})",
        flush=True
    )
    return assigned


def build_overlay_command(
    camera: RegisteredCamera,
    rtsp_url: str,
    port: int,
    config: RunnerConfig,
) -> list[str]:
    """카메라 하나를 처리할 `serve_ai_overlay.py` worker 명령을 구성한다.

    이 함수는 backend schema나 MQTT payload를 직접 만들지 않는다. 대신 tracking,
    pose diagnostics, LSTM sequence, MQTT topic 같은 런타임 옵션을 CLI 인자로
    worker에 전달한다. RTSP URL과 MQTT password는 민감 정보가 될 수 있어 argv에
    싣지 않고 `start_camera_worker()`에서 환경변수로 주입한다.
    """

    command = [
        config.python_executable,
        "scripts/serve_ai_overlay.py",
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
        "--detector-conf",
        str(config.detector_conf),
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
        "--track-thresh",
        str(config.track_thresh),
        "--match-thresh",
        str(config.match_thresh),
        "--track-buffer",
        str(config.track_buffer),
        "--frame-rate",
        str(config.frame_rate),
        "--bbox-smoothing-alpha",
        str(config.bbox_smoothing_alpha),
        "--tracking-grace-period-seconds",
        str(config.tracking_grace_period_seconds),
        "--tracking-relink-iou-threshold",
        str(config.tracking_relink_iou_threshold),
        "--tracking-relink-center-ratio",
        str(config.tracking_relink_center_ratio),
        "--tracking-relink-max-time-gap-seconds",
        str(config.tracking_relink_max_time_gap_seconds),
        "--pose-debug-summary-every-n",
        str(config.pose_debug_summary_every_n),
        "--pose-min-keypoint-confidence",
        str(config.pose_min_keypoint_confidence),
        "--pose-debug-image-dir",
        config.pose_debug_image_dir,
        "--pose-debug-image-every-n",
        str(config.pose_debug_image_every_n),
        "--pose-tracking-diag-jsonl-path",
        config.pose_tracking_diag_jsonl_path,
    ]
    if config.pose_debug:
        command.append("--pose-debug")
    if config.pose_debug_save_images:
        command.append("--pose-debug-save-images")
    if config.pose_tracking_diag_jsonl:
        command.append("--pose-tracking-diag-jsonl")
    if tracking_stability_fallback_enabled(camera, config):
        command.append("--tracking-stability-fallback")
    if config.mjpeg_debug:
        command.append("--mjpeg-debug")
    if config.webrtc_sync_enabled:
        sync_port = int(config.webrtc_sync_base_port) + max(0, int(port) - int(config.overlay_base_port))
        command.extend(
            [
                "--webrtc-sync-enabled",
                "--webrtc-sync-host",
                config.webrtc_sync_host,
                "--webrtc-sync-port",
                str(sync_port),
                "--webrtc-sync-stream-id",
                f"{camera.camera_login_id}_ai",
            ]
        )
    if config.mjpeg_enabled:
        command.append("--mjpeg-enabled")
    command.extend(
        [
            "--mjpeg-fps",
            str(config.mjpeg_fps),
            "--mjpeg-width",
            str(config.mjpeg_width),
            "--mjpeg-height",
            str(config.mjpeg_height),
            "--mjpeg-jpeg-quality",
            str(config.mjpeg_jpeg_quality),
            "--mjpeg-base-path",
            config.mjpeg_base_path,
        ]
    )
    if not config.mjpeg_enable_overlay:
        command.append("--no-mjpeg-enable-overlay")
    optional_pairs = [
        ("--mqtt-host", config.mqtt_host),
        ("--mqtt-port", str(config.mqtt_port) if config.mqtt_port is not None else None),
        ("--mqtt-topic", config.mqtt_topic),
        ("--mqtt-camera-topic", config.mqtt_camera_topic),
        ("--mqtt-event-topic", config.mqtt_event_topic),
        ("--mqtt-status-topic", config.mqtt_status_topic),
        ("--mqtt-client-id", f"{config.mqtt_client_id_prefix}-{camera.camera_login_id}"),
        ("--mqtt-username", config.mqtt_username),
        ("--action-model", config.action_model),
        ("--action-threshold", str(config.action_threshold) if config.action_threshold is not None else None),
        ("--selected-track-id", str(config.selected_track_id) if config.selected_track_id is not None else None),
        ("--selected-track-mode", config.selected_track_mode),
        ("--selected-track-missing-frames", str(config.selected_track_missing_frames) if config.selected_track_missing_frames is not None else None),
    ]
    for key, value in optional_pairs:
        if value:
            command.extend([key, value])
    if config.print_events:
        command.append("--print-events")
    if camera.roi_configs:
        command.extend(["--roi-configs", json.dumps(list(camera.roi_configs))])
    if camera.exit_roi_configs:
        command.extend(["--exit-roi-configs", json.dumps(list(camera.exit_roi_configs))])
    return command


def tracking_stability_fallback_enabled(camera: RegisteredCamera, config: RunnerConfig) -> bool:
    fallback_camera_ids = {
        normalize_camera_login_id(camera_id)
        for camera_id in config.tracking_stability_fallback_camera_ids
    }
    return bool(config.tracking_stability_fallback or camera.camera_login_id in fallback_camera_ids)


def input_rtsp_for_camera(camera: RegisteredCamera, config: RunnerConfig) -> tuple[str, list[str] | None]:
    """카메라 sourceType에 따라 worker 입력 RTSP URL과 ffmpeg 송출 명령을 정한다.

    REAL_RTSP는 backend의 `rtspUrl`을 그대로 분석한다. SIMULATED_RTSP는
    MediaMTX의 `rtsp://.../{cameraLoginId}` 경로를 만들고, 필요하면 mp4를 그
    경로로 publish하는 ffmpeg 명령을 함께 반환한다.
    """

    match camera.source_type:
        case "REAL_RTSP":
            if not camera.rtsp_url:
                raise RuntimeError(f"REAL_RTSP camera has no rtspUrl: {camera.camera_login_id}")
            return camera.rtsp_url, None
        case "SIMULATED_RTSP":
            rtsp_url = camera_rtsp_url(config.rtsp_base_url, camera.camera_login_id)
            if getattr(config, "skip_ffmpeg_spawn", False):
                return rtsp_url, None
            video_path = resolve_simulated_video(camera, config.video_pool, config)
            return rtsp_url, build_ffmpeg_command(video_path, rtsp_url)
        case unreachable:
            assert_never(unreachable)
