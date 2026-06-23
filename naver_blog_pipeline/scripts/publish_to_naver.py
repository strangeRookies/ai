from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NODE_SCRIPT = ROOT / "scripts" / "md-to-naver-blog-preview.mjs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Naver Blog preview HTML from a blog Markdown draft.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Blog draft Markdown path.")
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="Preview HTML path. Defaults to <input>.preview.html.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Conversion metadata JSON path. Defaults to <input>.naver.json.",
    )
    parser.add_argument("--node-script", default=DEFAULT_NODE_SCRIPT, type=Path)
    parser.add_argument("--draft", action="store_true", default=True, help="Preview-only mode.")
    parser.add_argument("--publish", action="store_true", help="Reserved for explicit publishing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publish:
        print(
            "publish mode is not implemented: @jjlabsio/md-to-naver-blog converts Markdown to "
            "paste-ready HTML, but it does not provide authenticated Naver Blog publishing.",
            file=sys.stderr,
        )
        return 2

    preview_output = args.preview_output or args.input.with_suffix(".preview.html")
    metadata_output = args.metadata_output or args.input.with_suffix(".naver.json")
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "node",
        str(args.node_script),
        "--input",
        str(args.input),
        "--html",
        str(preview_output),
        "--json",
        str(metadata_output),
    ]
    completed = subprocess.run(command, check=False, cwd=ROOT)
    if completed.returncode != 0:
        return completed.returncode

    print(f"preview: {preview_output}")
    print(f"metadata: {metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
