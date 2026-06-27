from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from ai.action.classifier import KEYPOINT_FEATURE_DIM, MOTION_KEYPOINT_FEATURE_DIM, keypoint_sequence_to_features
from benchmark.keypoint_cache_loader import keypoint_array_to_frames, load_keypoint_cache


LABEL_TO_ID: Final[dict[str, int]] = {"Normal": 0, "Fight": 1}
ID_TO_LABEL: Final[dict[int, str]] = {0: "Normal", 1: "Fight"}
FEATURE_DIMS: Final[set[int]] = {KEYPOINT_FEATURE_DIM, MOTION_KEYPOINT_FEATURE_DIM}


class FightDatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FightRow:
    npz_path: Path
    label: int
    label_name: str
    clip_id: str
    domain: str
    npz_path_raw: str


@dataclass(frozen=True, slots=True)
class SplitRows:
    train: list[FightRow]
    val: list[FightRow]
    test: list[FightRow]


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    x: np.ndarray
    y: np.ndarray
    rows: list[dict[str, int | str]]


def load_fight_rows(csv_path: Path) -> list[FightRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = [parse_row(row, csv_path.parent) for row in reader]
    if not rows:
        raise FightDatasetError(f"No rows found in {csv_path}")
    return rows


def parse_row(row: Mapping[str, str], csv_dir: Path) -> FightRow:
    label_name = label_name_from_row(row)
    raw_path = required_text(row, "npz_path")
    return FightRow(
        npz_path=resolve_npz_path(raw_path, csv_dir),
        label=LABEL_TO_ID[label_name],
        label_name=label_name,
        clip_id=row.get("clip_id", "").strip() or Path(raw_path).stem,
        domain=row.get("domain", "").strip(),
        npz_path_raw=raw_path,
    )


def label_name_from_row(row: Mapping[str, str]) -> str:
    raw_name = row.get("label_name", "").strip()
    if raw_name in LABEL_TO_ID:
        return raw_name
    raw_label = row.get("label", "").strip()
    if raw_label in {"0", "1"}:
        return ID_TO_LABEL[int(raw_label)]
    raise FightDatasetError(f"Unsupported label row: label={raw_label!r}, label_name={raw_name!r}")


def required_text(row: Mapping[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise FightDatasetError(f"Missing required column value: {key}")
    return value


def resolve_npz_path(raw_path: str, csv_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    csv_relative = csv_dir / path
    if csv_relative.exists():
        return csv_relative
    return path


def validate_rows(rows: list[FightRow]) -> dict[str, object]:
    counts = Counter(row.label_name for row in rows)
    missing = [str(row.npz_path) for row in rows if not row.npz_path.exists()]
    return {"rows": len(rows), "labels": dict(counts), "missing_files": len(missing), "missing_examples": missing[:10]}


def stratified_split(rows: list[FightRow], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> SplitRows:
    by_label: dict[int, list[FightRow]] = {0: [], 1: []}
    for row in rows:
        by_label[row.label].append(row)
    train: list[FightRow] = []
    val: list[FightRow] = []
    test: list[FightRow] = []
    for label, label_rows in by_label.items():
        if not label_rows:
            raise FightDatasetError(f"No rows for label {ID_TO_LABEL[label]}")
        shuffled = list(label_rows)
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        train_count = int(round(len(shuffled) * train_ratio))
        val_count = int(round(len(shuffled) * val_ratio))
        train.extend(shuffled[:train_count])
        val.extend(shuffled[train_count : train_count + val_count])
        test.extend(shuffled[train_count + val_count :])
    return SplitRows(train=stable_shuffle(train, seed, "train"), val=stable_shuffle(val, seed, "val"), test=stable_shuffle(test, seed, "test"))


def stable_shuffle(rows: list[FightRow], seed: int, split: str) -> list[FightRow]:
    shuffled = list(rows)
    random.Random(f"{seed}:{split}").shuffle(shuffled)
    return shuffled


def collect_sequences(rows: list[FightRow], sequence_length: int, sequence_stride: int) -> SequenceBatch:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    sequence_rows: list[dict[str, int | str]] = []
    for row in rows:
        sequences = sequences_from_npz(row.npz_path, sequence_length, sequence_stride)
        for index, features in enumerate(sequences):
            x_rows.append(features)
            y_rows.append(row.label)
            sequence_rows.append(
                {
                    "clip_id": row.clip_id,
                    "domain": row.domain,
                    "label": row.label,
                    "label_name": row.label_name,
                    "npz_path": str(row.npz_path),
                    "sequence_index": index,
                }
            )
    if not x_rows:
        raise FightDatasetError("No LSTM sequences were generated from the selected rows.")
    return SequenceBatch(x=np.stack(x_rows).astype(np.float32), y=np.asarray(y_rows, dtype=np.int64), rows=sequence_rows)


def sequences_from_npz(path: Path, sequence_length: int, sequence_stride: int) -> list[np.ndarray]:
    raw = load_keypoint_cache(path)
    feature_sequences = feature_array_sequences(raw, sequence_length, sequence_stride)
    if feature_sequences:
        return feature_sequences
    frames = keypoint_array_to_frames(raw)
    results: list[np.ndarray] = []
    for start in range(0, max(0, len(frames) - sequence_length + 1), sequence_stride):
        window = frames[start : start + sequence_length]
        sequence = {
            "detections": [first_detection(frame) for frame in window],
            "frame_shapes": [frame.get("frame_shape") for frame in window],
        }
        results.append(keypoint_sequence_to_features(sequence))
    return results


def feature_array_sequences(raw: np.ndarray, sequence_length: int, sequence_stride: int) -> list[np.ndarray]:
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3 and int(array.shape[-1]) in FEATURE_DIMS:
        return [normalize_feature_dim(array[index]) for index in range(array.shape[0])]
    if array.ndim == 2 and int(array.shape[-1]) in FEATURE_DIMS:
        return [normalize_feature_dim(array[start : start + sequence_length]) for start in range(0, max(0, array.shape[0] - sequence_length + 1), sequence_stride)]
    return []


def normalize_feature_dim(features: np.ndarray) -> np.ndarray:
    if features.shape[-1] == KEYPOINT_FEATURE_DIM:
        return features.astype(np.float32)
    if features.shape[-1] == MOTION_KEYPOINT_FEATURE_DIM:
        return features[..., :KEYPOINT_FEATURE_DIM].astype(np.float32)
    raise FightDatasetError(f"Unsupported LSTM feature dimension: {features.shape[-1]}")


def first_detection(frame: Mapping[str, object]) -> dict[str, object]:
    detections = frame.get("detections")
    if isinstance(detections, list) and detections:
        candidate = detections[0]
        if isinstance(candidate, dict):
            return candidate
    return {"bbox": None, "keypoints": []}


def inspect_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        return {"path": str(path), "files": list(data.files), "arrays": {key: list(data[key].shape) for key in data.files}}
