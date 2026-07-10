from __future__ import annotations

import json
from pathlib import Path

from ai.learning.manifest_schema import CandidateKind, LEGACY_SYNTHETIC_ALIASES, REASON_ENUM, REQUIRED_TRAINING_MANIFEST_V2_FIELDS, ReviewStatus


def dedupe(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = next((row.get(name, "").strip() for name in keys if row.get(name, "").strip()), "")
        if key and key not in seen:
            seen.add(key)
            output.append(row)
    return output


def fieldnames(rows: list[dict[str, str]]) -> list[str]:
    ordered = [*REQUIRED_TRAINING_MANIFEST_V2_FIELDS]
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def clip_id(row: dict[str, str]) -> str:
    return row.get("clip_id", "").strip() or Path(row.get("clip_path", row.get("video_path", "clip"))).stem


def label(row: dict[str, str]) -> str:
    raw = row.get("label", "").strip()
    if raw and raw not in {"Faint", "Normal", "faint", "normal"}:
        return raw
    return "1" if label_name(row) == "Faint" else "0"


def label_name(row: dict[str, str]) -> str:
    raw = row.get("label_name", row.get("true_label", row.get("label", ""))).strip()
    return "Faint" if raw in {"1", "Faint", "faint", "Fall", "fall"} else "Normal"


def predicted_label(row: dict[str, str]) -> str:
    return row.get("predicted_label", row.get("prediction", row.get("pred_label", ""))).strip()


def first(row: dict[str, str], meta: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "").strip() or meta.get(key, "").strip()
        if value:
            return value
    return ""


def scenario_tag(row: dict[str, str], kind: CandidateKind) -> str:
    fallback = "normal_false_positive" if kind == "hard_negative" else "faint_false_negative"
    return enum_reason(row.get("scenario_tag", row.get("reason", row.get("weak_condition", fallback))).strip() or fallback)


def enum_reason(reason: str) -> str:
    return reason if reason in REASON_ENUM else "weak_condition_augmentation_candidate"


def augmentation_type(raw_type: str) -> str:
    if not raw_type:
        return ""
    return LEGACY_SYNTHETIC_ALIASES.get(raw_type, raw_type)


def split_group_id(row: dict[str, str], fallback: str) -> str:
    return row.get("split_group_id", "").strip() or row.get("parent_clip_id", "").strip() or row.get("source_video", "").strip() or fallback


def estimated_visibility(value: str) -> float:
    if value == "partial_occlusion":
        return 0.55
    if value in {"scale_down", "blur"}:
        return 0.7
    return 0.9


def synthetic_review_status(_visibility: float, _min_visibility_for_approved: float) -> ReviewStatus:
    return "pending"


def augmentation_config(value: str, seed: int) -> str:
    configs = {
        "brightness": {"alpha": 0.75, "beta": -12},
        "noise": {"sigma": 0.03},
        "blur": {"kernel": 5},
        "compression": {"jpeg_quality": 35},
        "scale_down": {"scale": 0.6},
        "partial_occlusion": {"occlusion_ratio": 0.35},
        "horizontal_flip": {"flip": True},
    }
    return json.dumps({"seed": seed, "type": value, "params": configs.get(value, {})}, sort_keys=True)


def sample_rows(rows: list[dict[str, str]], sample: int | None) -> list[dict[str, str]]:
    if sample is None or sample < 0:
        return rows
    return rows[:sample]
