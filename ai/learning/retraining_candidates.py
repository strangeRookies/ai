from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final, TypedDict

from ai.learning.feedback import FeedbackRecord
from ai.learning.feedback_store import load_feedback_jsonl


FIELDNAMES: Final = [
    "event_id",
    "camera_login_id",
    "frame_id",
    "timestamp_ms",
    "feedback_outcome",
    "candidate_type",
    "confidence",
    "bbox",
    "snapshot_path",
    "clip_path",
    "feedback_text",
]


class CandidateSummary(TypedDict):
    total: int
    hard_negative: int
    faint_fall_reinforcement: int
    verified_positive: int
    verified_negative: int


def export_retraining_candidates(feedback_path: Path, output_dir: Path) -> CandidateSummary:
    records = load_feedback_jsonl(feedback_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {
        "hard_negative": _filter_records(records, "hard_negative"),
        "faint_fall_reinforcement": _filter_records(records, "faint_fall_reinforcement"),
        "verified_positive": _filter_records(records, "verified_positive"),
        "verified_negative": _filter_records(records, "verified_negative"),
    }
    for candidate_type, rows in grouped.items():
        _write_candidates(output_dir / f"{candidate_type}_candidates.csv", rows)
    summary = CandidateSummary(
        total=len(records),
        hard_negative=len(grouped["hard_negative"]),
        faint_fall_reinforcement=len(grouped["faint_fall_reinforcement"]),
        verified_positive=len(grouped["verified_positive"]),
        verified_negative=len(grouped["verified_negative"]),
    )
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _filter_records(records: list[FeedbackRecord], candidate_type: str) -> list[FeedbackRecord]:
    return [record for record in records if record["candidate_type"] == candidate_type]


def _write_candidates(path: Path, rows: list[FeedbackRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "event_id": row["event_id"],
                    "camera_login_id": row["camera_login_id"],
                    "frame_id": row["frame_id"],
                    "timestamp_ms": row["timestamp_ms"],
                    "feedback_outcome": row["feedback_outcome"],
                    "candidate_type": row["candidate_type"],
                    "confidence": row["confidence"],
                    "bbox": json.dumps(row["bbox"], ensure_ascii=False),
                    "snapshot_path": row["snapshot_path"] or "",
                    "clip_path": row["clip_path"] or "",
                    "feedback_text": row["feedback_text"],
                }
            )
