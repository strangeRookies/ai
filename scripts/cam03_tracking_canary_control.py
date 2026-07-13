#!/usr/bin/env python3
"""Enable/disable cam_03 tracking canary via JSON config (per-camera only).

Usage:
  python scripts/cam03_tracking_canary_control.py enable
  python scripts/cam03_tracking_canary_control.py disable
  python scripts/cam03_tracking_canary_control.py status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.tracking_canary import (  # noqa: E402
    canary_config_path,
    clear_canary_config,
    resolve_canary_settings,
    write_canary_config,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["enable", "disable", "status"])
    ap.add_argument("--camera", default="cam_03")
    ap.add_argument("--mode", default="hybrid_kp")
    ap.add_argument("--new-track-thresh", type=float, default=0.30)
    args = ap.parse_args()
    cam = args.camera
    if args.action == "enable":
        path = write_canary_config(
            {
                cam: {
                    "enabled": True,
                    "near_dup_suppress_mode": args.mode,
                    "new_track_thresh": args.new_track_thresh,
                }
            }
        )
        print(f"[canary] enabled {cam} mode={args.mode} thresh={args.new_track_thresh} path={path}", flush=True)
    elif args.action == "disable":
        path = clear_canary_config()
        print(f"[canary] cleared all canary cameras path={path}", flush=True)
    s = resolve_canary_settings(cam)
    print(json.dumps({"camera": cam, "settings": s, "config_path": str(canary_config_path())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
