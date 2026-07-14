from __future__ import annotations

import socket
import subprocess
import sys
import re
import time
from typing import Protocol

from ai.registered_cameras import RunnerConfig


class OverlayPortOwner(Protocol):
    overlay_port: int


def _is_port_bound(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


def _kill_orphan_on_port(port: int) -> None:
    """등록되지 않은 채 포트를 점유 중인 프로세스를 정리해서 되찾는다.

    이 포트는 특정 카메라 전용으로 고정 배정된 포트라, 여기 떠있는 건 거의 항상
    그 카메라의 이전 세션이 안 죽고 남은 좀비 프로세스다. 회피하지 않고 정리해서
    되찾아야 카메라별 포트가 재시작마다 흔들리지 않고 항상 같게 유지된다.
    """
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5, check=False)
    except Exception as exc:
        print(
            f"[registered-cameras][warning] failed to reclaim orphan overlay port {port}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _is_overlay_port_available(port: int, used_ports: set[int]) -> bool:
    if port in used_ports:
        return False
    if _is_port_bound(port):
        print(
            f"[registered-cameras][warning] overlay port {port} is open without a registry owner; treating as orphan",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def overlay_port_for_camera_login_id(camera_login_id: str, config: RunnerConfig) -> int | None:
    """Return the stable MJPEG port for numeric cameraLoginId suffixes.

    cam_01 maps to overlay_base_port, cam_02 maps to overlay_base_port + 1,
    and the same suffix rule keeps working beyond four cameras.
    """

    match = re.search(r"(\d+)$", camera_login_id.strip())
    if match is None:
        return None
    camera_number = int(match.group(1))
    if camera_number <= 0:
        return None
    return int(config.overlay_base_port) + camera_number - 1


def next_overlay_port(
    workers: dict[str, OverlayPortOwner],
    config: RunnerConfig,
    preferred_port: int | None = None,
    camera_login_id: str | None = None,
) -> int:
    used_ports = {worker.overlay_port for worker in workers.values()}
    if camera_login_id is not None:
        camera_port = overlay_port_for_camera_login_id(camera_login_id, config)
        if camera_port is not None and camera_port not in used_ports:
            if _is_port_bound(camera_port):
                print(
                    f"[registered-cameras][warning] overlay port {camera_port} (dedicated to {camera_login_id}) "
                    "is held by an orphan process; reclaiming it",
                    file=sys.stderr,
                    flush=True,
                )
                _kill_orphan_on_port(camera_port)
                time.sleep(0.5)  # 소켓이 반환될 시간 확보
            if not _is_port_bound(camera_port):
                return camera_port

    if preferred_port is not None and _is_overlay_port_available(preferred_port, used_ports):
        return preferred_port

    port = config.overlay_base_port
    while not _is_overlay_port_available(port, used_ports):
        port += 1
    return port
