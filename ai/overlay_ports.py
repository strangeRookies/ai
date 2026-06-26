from __future__ import annotations

import socket
import sys
from typing import Protocol

from ai.registered_cameras import RunnerConfig


class OverlayPortOwner(Protocol):
    overlay_port: int


def _is_port_bound(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


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


def next_overlay_port(
    workers: dict[str, OverlayPortOwner],
    config: RunnerConfig,
    preferred_port: int | None = None,
) -> int:
    used_ports = {worker.overlay_port for worker in workers.values()}
    if preferred_port is not None and _is_overlay_port_available(preferred_port, used_ports):
        return preferred_port

    port = config.overlay_base_port
    while not _is_overlay_port_available(port, used_ports):
        port += 1
    return port
