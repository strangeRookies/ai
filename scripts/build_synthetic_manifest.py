#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# --- How to run ---
# 1. Install uv (if not installed): https://docs.astral.sh/uv/
# 2. Run: uv run scripts/build_synthetic_manifest.py --metadata-csv data/metadata.csv --output-csv runs/synthetic/synthetic_candidates.csv
# ------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.synthetic_manifest import build_synthetic_candidate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an augmentation candidate manifest without generating video.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    summary = build_synthetic_candidate_manifest(Path(args.metadata_csv), Path(args.output_csv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
