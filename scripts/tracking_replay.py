#!/usr/bin/env python3
"""CLI: replay synthetic detections through tracker (no real video)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.tracking_replay import DISCLAIMER, load_detections_jsonl, run_tracking_replay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tracking replay over detections JSONL (synthetic only).")
    parser.add_argument(
        "--detections",
        type=Path,
        default=ROOT / "fixtures" / "tracking" / "synthetic_scenarios.jsonl",
        help="JSONL path with per-frame detections",
    )
    parser.add_argument("--camera-login-id", default="cam_synthetic")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    frames = load_detections_jsonl(args.detections)
    result = run_tracking_replay(
        frames,
        camera_login_id=args.camera_login_id,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(DISCLAIMER, file=sys.stderr)
    print(json.dumps({"runId": result["runId"], "outputDir": result["outputDir"], "disclaimer": DISCLAIMER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
