#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.embedding_sdk import embed_text, resolve_provider  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s:%(message)s")
    parser = argparse.ArgumentParser(description="Embedding worker (direct SDK, no LangChain).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to embed")
    group.add_argument("--file", type=Path, help="UTF-8 text file to embed")
    parser.add_argument("--provider", default=None, help="mock|gemini (default EMBEDDING_PROVIDER or mock)")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    provider = resolve_provider(provider_name=args.provider)
    logging.info("embedding with provider=%s dim=%s", provider.model_name(), provider.dimension())
    result = embed_text(text, provider=provider)
    sys.stdout.write(result.to_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
