from __future__ import annotations

import json
from pathlib import Path

from ai.learning.feedback import FeedbackRecord


def append_feedback_jsonl(path: Path, record: FeedbackRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        fp.write("\n")


def load_feedback_jsonl(path: Path) -> list[FeedbackRecord]:
    if not path.exists():
        return []
    records: list[FeedbackRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if text:
                records.append(FeedbackRecord(**json.loads(text)))
    return records
