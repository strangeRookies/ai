"""MQTT client identity construction."""

from __future__ import annotations

import os
import re


def build_mqtt_client_id(base: str, camera_login_id: str, role: str, *, pid: int | None = None) -> str:
    """Return a broker-safe identifier unique to camera, connection role, and process."""
    safe_base = _safe_segment(base, "strange-ai")
    safe_camera = _safe_segment(camera_login_id, "unknown-camera")
    safe_role = _safe_segment(role, "inference")
    process_id = os.getpid() if pid is None else int(pid)
    return f"{safe_base}-{safe_camera}-{safe_role}-{process_id}"[:128]


def _safe_segment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip()).strip("-")
    return normalized or fallback
