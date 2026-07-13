# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ------------------------------------------------------------
# python scripts/process_vlm_mock.py --job fixtures/vlm/demo_job.json
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.vlm_mock import JSONValue, job_to_json, parse_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DB-less mock VLM worker.")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s:%(message)s")
    logging.info("reading mock VLM job from %s", args.job)
    with args.job.open("r", encoding="utf-8") as handle:
        payload: dict[str, JSONValue] = json.load(handle)
    sys.stdout.write(job_to_json(parse_job(payload)))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
