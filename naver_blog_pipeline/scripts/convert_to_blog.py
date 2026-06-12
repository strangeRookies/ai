from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv(ROOT / ".env")

from blog_pipeline.llm_client import LLMClient
from blog_pipeline.markdown_blog import write_blog_draft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a source Markdown note into a Korean blog-style Markdown draft.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Source Markdown file.")
    parser.add_argument("--output", required=True, type=Path, help="Blog draft Markdown path.")
    
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-llm",
        action="store_true",
        default=None,
        help="Force use of LLM for humanizing text.",
    )
    llm_group.add_argument(
        "--no-llm",
        action="store_true",
        default=None,
        help="Disable LLM and use fallback text-softening.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    use_llm = True
    if args.no_llm:
        use_llm = False
    elif args.use_llm:
        use_llm = True
    else:
        # Auto-detect keys in environment
        import os
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not gemini_key and not openai_key:
            use_llm = False

    llm_client = None
    if use_llm:
        llm_client = LLMClient()
        if not llm_client.is_enabled():
            print("[convert_to_blog] LLM requested but client initialization failed. Falling back to regex mode.", flush=True)
            llm_client = None
        else:
            print(f"[convert_to_blog] Using LLM ({llm_client.provider}) for humanizing the blog post...", flush=True)
    else:
        print("[convert_to_blog] Running in regex-based fallback mode (no LLM).", flush=True)

    draft = write_blog_draft(args.input, args.output, llm_client=llm_client)
    print(f"wrote: {args.output}")
    print(f"title: {draft.title}")
    print(f"sections: {len(draft.sections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
