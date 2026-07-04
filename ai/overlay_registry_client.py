from __future__ import annotations

import json
import sys
from http.client import HTTPException
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from ai.registered_cameras import RegisteredCamera, RunnerConfig


def overlay_stream_url(config: RunnerConfig, port: int) -> str:
    base_url = config.overlay_public_base_url or f"http://localhost:{port}"
    parsed = urlparse(base_url)
    hostname = parsed.hostname or "localhost"
    netloc = f"{hostname}:{port}"
    return urlunparse((parsed.scheme or "http", netloc, "/stream", "", "", ""))


def report_overlay_status(
    camera: RegisteredCamera,
    rtsp_url: str,
    port: int,
    config: RunnerConfig,
    status: str,
    pid: int | None,
) -> None:
    if not config.overlay_report_enabled:
        return
    payload = {
        "cameraLoginId": camera.camera_login_id,
        "rtspUrl": rtsp_url,
        "overlayPort": port,
        "overlayUrl": overlay_stream_url(config, port),
        "pid": pid,
        "status": status,
    }
    request = Request(
        f"{config.backend_base_url.rstrip('/')}/api/internal/ai-overlays/report",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    if config.backend_token:
        request.add_header("Authorization", f"Bearer {config.backend_token}")
    try:
        with urlopen(request, timeout=config.backend_timeout_seconds):
            return
    except TimeoutError:
        print(
            f"[registered-cameras][warning] overlay registry report timed out camera={camera.camera_login_id}",
            file=sys.stderr,
            flush=True,
        )
    except (HTTPException, OSError, URLError) as exc:
        print(
            f"[registered-cameras][warning] overlay registry report failed camera={camera.camera_login_id}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def report_overlay_stopped(
    camera_login_id: str,
    rtsp_url: str | None,
    port: int,
    config: RunnerConfig,
) -> None:
    camera = RegisteredCamera(
        camera_id=camera_login_id,
        camera_login_id=camera_login_id,
        rtsp_url=rtsp_url,
        source_type="REAL_RTSP",
        assigned_video_path=None,
    )
    report_overlay_status(camera, rtsp_url or "", port, config, "STOPPED", None)
