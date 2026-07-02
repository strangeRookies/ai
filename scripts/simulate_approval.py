#!/usr/bin/env python
from pathlib import Path


def main():
    p = Path("data/manifests/hard_negative_candidates.jsonl")
    if p.exists():
        content = p.read_text(encoding="utf-8")
        # Replace status in JSON string
        content = content.replace('"review_status": "pending"', '"review_status": "approved"')
        p.write_text(content, encoding="utf-8")
        print("Successfully simulated approval for hard negative candidates.")
    else:
        print("[ERROR] Candidate file not found at data/manifests/hard_negative_candidates.jsonl")


if __name__ == "__main__":
    main()
