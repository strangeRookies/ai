#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# --- How to run ---
# 1. Install uv (if not installed): https://docs.astral.sh/uv/
# 2. Run: uv run scripts/export_feedback_candidates.py --feedback-jsonl runs/self_improvement/feedback.jsonl --output-dir runs/self_improvement/candidates
# ------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.learning.retraining_candidates import export_retraining_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Export self-improvement feedback into retraining candidate CSV files.")
    parser.add_argument("--feedback-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = export_retraining_candidates(Path(args.feedback_jsonl), Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
