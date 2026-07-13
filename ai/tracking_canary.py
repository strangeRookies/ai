"""Tracking suppression defaults + optional per-camera canary override.

Production defaults (2026-07 canary-validated, known limitations remain):
  NEAR_DUP_SUPPRESS_MODE=hybrid_kp
  SIMPLE_TRACK_NEW_TRACK_THRESH=0.30

Rollback defaults:
  NEAR_DUP_SUPPRESS_MODE=none
  SIMPLE_TRACK_NEW_TRACK_THRESH=0.25

Optional per-camera canary still supported for future experiments:

  TRACKING_CANARY_CONFIG=runs/tracking_canary/canary_config.json
  TRACKING_CANARY_CAMERA_IDS=cam_03
  TRACKING_CANARY_NEAR_DUP_MODE=...
  TRACKING_CANARY_NEW_TRACK_THRESH=...

When canary is inactive, all cameras use production defaults (not forced none).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CANARY_CONFIG_PATH = "runs/tracking_canary/canary_config.json"

# Adopted production defaults after offline A/B + cam_03 live canary.
PRODUCTION_NEAR_DUP_SUPPRESS_MODE = "hybrid_kp"
PRODUCTION_NEW_TRACK_THRESH = 0.30

# Pre-adoption values for rollback scripts / docs.
ROLLBACK_NEAR_DUP_SUPPRESS_MODE = "none"
ROLLBACK_NEW_TRACK_THRESH = 0.25


def canary_config_path() -> Path:
    raw = (os.getenv("TRACKING_CANARY_CONFIG") or DEFAULT_CANARY_CONFIG_PATH).strip()
    path = Path(raw)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        path = root / path
    return path


def load_canary_file() -> dict[str, Any]:
    path = canary_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    cameras = data.get("cameras")
    if isinstance(cameras, dict):
        return cameras
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def env_canary_camera_ids() -> set[str]:
    raw = os.getenv("TRACKING_CANARY_CAMERA_IDS") or ""
    return {part.strip() for part in raw.split(",") if part.strip()}


def production_tracking_defaults() -> dict[str, Any]:
    """Effective production defaults (env override allowed)."""
    mode = (os.getenv("NEAR_DUP_SUPPRESS_MODE") or PRODUCTION_NEAR_DUP_SUPPRESS_MODE).strip().lower()
    thresh_raw = os.getenv("SIMPLE_TRACK_NEW_TRACK_THRESH")
    if thresh_raw is None or str(thresh_raw).strip() == "":
        thresh = float(PRODUCTION_NEW_TRACK_THRESH)
    else:
        try:
            thresh = float(thresh_raw)
        except (TypeError, ValueError):
            thresh = float(PRODUCTION_NEW_TRACK_THRESH)
    return {
        "canary": False,
        "near_dup_suppress_mode": mode or PRODUCTION_NEAR_DUP_SUPPRESS_MODE,
        "new_track_thresh": thresh,
        "source": "production-default",
        "configSource": "production-default",
    }


def resolve_canary_settings(camera_login_id: str) -> dict[str, Any]:
    """Return effective tracking settings for one camera.

    - canary camera: optional experimental override
    - otherwise: production defaults (hybrid_kp / 0.30 unless env overrides)
    """
    cam_id = str(camera_login_id or "").strip()
    file_cfg = load_canary_file().get(cam_id) or {}
    env_ids = env_canary_camera_ids()
    in_env_list = cam_id in env_ids
    if file_cfg:
        enabled = bool(file_cfg.get("enabled", True))
    else:
        enabled = in_env_list

    if not enabled:
        return production_tracking_defaults()

    mode = (
        file_cfg.get("near_dup_suppress_mode")
        or os.getenv("TRACKING_CANARY_NEAR_DUP_MODE")
        or PRODUCTION_NEAR_DUP_SUPPRESS_MODE
    )
    thresh_raw = file_cfg.get("new_track_thresh")
    if thresh_raw is None:
        thresh_raw = os.getenv("TRACKING_CANARY_NEW_TRACK_THRESH") or str(PRODUCTION_NEW_TRACK_THRESH)
    try:
        thresh = float(thresh_raw)
    except (TypeError, ValueError):
        thresh = float(PRODUCTION_NEW_TRACK_THRESH)
    return {
        "canary": True,
        "near_dup_suppress_mode": str(mode).strip().lower() or PRODUCTION_NEAR_DUP_SUPPRESS_MODE,
        "new_track_thresh": thresh,
        "source": "file" if file_cfg else "env",
        "configSource": "canary-override",
    }


def apply_canary_env(camera_login_id: str, env: dict[str, str]) -> dict[str, Any]:
    """Mutate worker env with production defaults or canary override.

    Non-canary workers receive production defaults (hybrid_kp/0.30), not forced none.
    """
    settings = resolve_canary_settings(camera_login_id)
    env["NEAR_DUP_SUPPRESS_MODE"] = str(settings["near_dup_suppress_mode"])
    env["SIMPLE_TRACK_NEW_TRACK_THRESH"] = str(settings["new_track_thresh"])
    env["NEAR_DUP_SORT_BY_CONF"] = env.get("NEAR_DUP_SORT_BY_CONF") or "1"
    if settings["canary"]:
        env["TRACKING_CANARY"] = "true"
        env["TRACK_ID_LOG"] = env.get("TRACK_ID_LOG") or "true"
    else:
        env["TRACKING_CANARY"] = "false"
    return settings


def canary_signature_fragment(camera_login_id: str) -> dict[str, Any]:
    s = resolve_canary_settings(camera_login_id)
    return {
        "canary": bool(s["canary"]),
        "near_dup_suppress_mode": s["near_dup_suppress_mode"],
        "new_track_thresh": s["new_track_thresh"],
        "configSource": s.get("configSource") or s.get("source"),
    }


def write_canary_config(
    cameras: dict[str, dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    out = path or canary_config_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cameras": cameras}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def clear_canary_config(*, path: Path | None = None) -> Path:
    """Disable all per-camera canary overrides (cameras fall back to production defaults)."""
    return write_canary_config({}, path=path)
