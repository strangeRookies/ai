from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np

from ai.action.classifier import sequence_to_lstm_features
from ai.evaluation.prediction_log import NEGATIVE_LABEL, POSITIVE_LABEL, normalize_ground_truth


FEATURE_SCHEMA: Final = "keypoint_bbox54"
FEATURE_DIM: Final = 54
CandidateType = Literal["hard_negative", "faint_fall_reinforcement"]


class PredictionRow(TypedDict, total=False):
    source_video: str
    clip_id: str
    start_frame: int
    end_frame: int
    frameId: int
    frame_id: int
    prediction_label: str
    prediction_score: float
    ground_truth_label: str
    threshold: float
    model_name: str
    checkpoint_path: str
    sequence_length: int
    sequence_stride: int
    feature_schema: str
    sequence: dict[str, list[dict[str, object]] | list[tuple[int, int, int]]]
    label_interval_verified: bool


class MiningSummary(TypedDict):
    processed: int
    accepted_counts: dict[str, int]
    quarantine_count: int
    quarantine_reasons: dict[str, int]
    hard_negative_path: str
    reinforcement_path: str
    quarantine_path: str


def mine_prediction_rows(rows: list[PredictionRow], output_dir: Path) -> MiningSummary:
    hard_path = output_dir / "hard_negative_candidates.jsonl"
    reinforcement_path = output_dir / "faint_fall_reinforcement_candidates.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    accepted: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    for row in rows:
        candidate_type = classify_error(row)
        if candidate_type is None:
            continue
        candidate, reason = build_candidate(row, candidate_type)
        if reason is not None:
            quarantine_reasons[reason] += 1
            append_jsonl(quarantine_path, quarantine_row(row, reason, candidate_type))
            continue
        target_path = hard_path if candidate_type == "hard_negative" else reinforcement_path
        append_jsonl(target_path, candidate)
        accepted[candidate_type] += 1
    return {
        "processed": len(rows),
        "accepted_counts": dict(sorted(accepted.items())),
        "quarantine_count": sum(quarantine_reasons.values()),
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "hard_negative_path": str(hard_path),
        "reinforcement_path": str(reinforcement_path),
        "quarantine_path": str(quarantine_path),
    }


def classify_error(row: PredictionRow) -> CandidateType | None:
    truth = normalize_ground_truth(row.get("ground_truth_label"))
    prediction = normalize_ground_truth(row.get("prediction_label")) or NEGATIVE_LABEL
    if prediction == POSITIVE_LABEL and truth == NEGATIVE_LABEL:
        return "hard_negative"
    if prediction == NEGATIVE_LABEL and truth == POSITIVE_LABEL:
        return "faint_fall_reinforcement"
    return None


def build_candidate(row: PredictionRow, candidate_type: CandidateType) -> tuple[dict[str, object], str | None]:
    reason = validation_reason(row)
    if reason is not None:
        return {}, reason
    feature_result, feature_reason = bbox54_features(row)
    if feature_reason is not None:
        return {}, feature_reason
    feature_sequence = feature_result.tolist()
    bbox_features = [round(float(value), 6) for value in feature_result[-1, 51:54]]
    return {
        "candidate_id": f"{row['clip_id']}:{row_frame_id(row)}:{candidate_type}",
        "candidate_type": candidate_type,
        "source_video": row["source_video"],
        "clip_id": row["clip_id"],
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        "frameId": row_frame_id(row),
        "prediction_label": row["prediction_label"],
        "prediction_score": float(row["prediction_score"]),
        "ground_truth_label": row["ground_truth_label"],
        "threshold": float(row["threshold"]),
        "model_name": row["model_name"],
        "checkpoint_path": row["checkpoint_path"],
        "sequence_length": int(row["sequence_length"]),
        "sequence_stride": int(row["sequence_stride"]),
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "bbox_features": bbox_features,
        "keypoint_confidence_summary": keypoint_confidence_summary(row),
        "feature_sequence": feature_sequence,
        "auto_merge_to_train": False,
        "review_status": "pending",
    }, None


def validation_reason(row: PredictionRow) -> str | None:
    required = ("source_video", "clip_id", "prediction_label", "ground_truth_label", "model_name", "checkpoint_path")
    for key in required:
        if not str(row.get(key, "")).strip():
            return f"missing_{key}"
    numeric_requirements = (
        ("start_frame", "missing_start_frame"),
        ("end_frame", "missing_end_frame"),
        ("prediction_score", "missing_prediction_score"),
        ("threshold", "missing_threshold"),
        ("sequence_length", "missing_sequence_length"),
        ("sequence_stride", "missing_sequence_stride"),
    )
    frame_value = row.get("frameId", row.get("frame_id"))
    if not numeric_value_present(frame_value):
        return "missing_frame_id"
    for key, reason in numeric_requirements:
        if not numeric_value_present(row.get(key)):
            return reason
    if not bool(row.get("label_interval_verified")):
        return "missing_label_interval"
    if int(row.get("end_frame", -1)) < int(row.get("start_frame", 0)):
        return "invalid_frame_range"
    if str(row.get("feature_schema", FEATURE_SCHEMA)) != FEATURE_SCHEMA:
        return "unsupported_feature_schema"
    return None


def bbox54_features(row: PredictionRow) -> tuple[np.ndarray, str | None]:
    sequence = row.get("sequence")
    if not isinstance(sequence, dict):
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "missing_sequence"
    detections = sequence.get("detections") or []
    if not detections:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "missing_sequence"
    if any(not isinstance(item, dict) or not item.get("bbox") for item in detections):
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "missing_bbox_feature"
    if any(not bbox_is_valid(item["bbox"]) for item in detections if isinstance(item, dict)):
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "malformed_bbox"
    features = sequence_to_lstm_features(sequence, input_size=FEATURE_DIM, feature_schema=FEATURE_SCHEMA)
    if int(features.shape[-1]) != FEATURE_DIM:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "feature_dim_mismatch"
    if np.allclose(features[:, 51:54], 0.0):
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), "missing_bbox_feature"
    return features.astype(np.float32), None


def keypoint_confidence_summary(row: PredictionRow) -> dict[str, float | int]:
    detections = ((row.get("sequence") or {}).get("detections") or []) if isinstance(row.get("sequence"), dict) else []
    values: list[float] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        for keypoint in detection.get("keypoints") or []:
            if isinstance(keypoint, dict) and keypoint.get("confidence") is not None:
                values.append(float(keypoint["confidence"]))
    if not values:
        return {"count": 0, "avg": 0.0, "min": 0.0}
    return {"count": len(values), "avg": round(sum(values) / len(values), 6), "min": round(min(values), 6)}


def quarantine_row(row: PredictionRow, reason: str, candidate_type: CandidateType) -> dict[str, object]:
    return {
        "reason": reason,
        "candidate_type": candidate_type,
        "source_video": row.get("source_video", ""),
        "clip_id": row.get("clip_id", ""),
        "frameId": row_frame_id(row),
        "prediction_label": row.get("prediction_label", ""),
        "ground_truth_label": row.get("ground_truth_label", ""),
        "auto_merge_to_train": False,
    }


def row_frame_id(row: PredictionRow) -> int:
    value = row.get("frameId", row.get("frame_id", 0))
    if not numeric_value_present(value):
        return 0
    return int(float(value))


def numeric_value_present(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str) and value.strip():
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def bbox_is_valid(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) < 4:
        return False
    try:
        [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return False
    return True


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        fp.write("\n")


def quarantine_reason_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("reason", "unknown")) for row in rows).items()))
