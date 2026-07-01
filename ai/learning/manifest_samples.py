from __future__ import annotations

from pathlib import Path


def sample_metadata_rows() -> list[dict[str, str]]:
    return [
        {
            "clip_id": "sample_real_normal",
            "clip_path": "samples/normal.mp4",
            "label": "0",
            "label_name": "Normal",
            "source_video": "sample_source_normal.mp4",
            "split": "train",
            "split_group_id": "sample_source_normal",
        },
        {
            "clip_id": "sample_real_faint",
            "clip_path": "samples/faint.mp4",
            "label": "1",
            "label_name": "Faint",
            "source_video": "sample_source_faint.mp4",
            "split": "train",
            "split_group_id": "sample_source_faint",
        },
    ]


def sample_false_positive_rows() -> list[dict[str, str]]:
    return [
        {
            "error_type": "FP",
            "clip_id": "sample_fp_bending",
            "clip_path": "samples/bending_normal.mp4",
            "label_name": "Normal",
            "prediction": "Faint",
            "faint_prob": "0.82",
            "reason": "bending_false_positive",
            "evidence_id": "sample-cam-1-100",
        }
    ]


def sample_false_negative_rows() -> list[dict[str, str]]:
    return [
        {
            "error_type": "FN",
            "clip_id": "sample_fn_night",
            "clip_path": "samples/night_faint.mp4",
            "label_name": "Faint",
            "prediction": "Normal",
            "faint_prob": "0.21",
            "reason": "night_false_negative",
            "evidence_id": "sample-cam-2-200",
        }
    ]


def sample_faint_candidate_rows() -> list[dict[str, str]]:
    return [
        {
            "clip_id": "sample_fn_night",
            "clip_path": "samples/night_faint.mp4",
            "label": "1",
            "label_name": "Faint",
            "source_type": "faint_reinforcement",
            "review_status": "approved",
            "failure_type": "false_negative",
            "scenario_tag": "night_false_negative",
            "split": "train",
            "split_group_id": "sample_source_faint",
        }
    ]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
