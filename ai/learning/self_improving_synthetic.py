from __future__ import annotations

import random
from typing import Final

import numpy as np


AUGMENTATION_TYPES: Final = (
    "keypoint_noise",
    "bbox_scale_jitter",
    "bbox_aspect_jitter",
    "confidence_drop",
    "frame_drop",
    "temporal_jitter",
    "horizontal_flip",
)


def build_synthetic_preview(candidates: list[dict[str, object]], seed: int = 42) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for candidate in candidates:
        sequence = candidate.get("feature_sequence")
        if int(candidate.get("feature_dim", 0) or 0) != 54 or not isinstance(sequence, list):
            continue
        features = np.asarray(sequence, dtype=np.float32)
        if features.ndim != 2 or int(features.shape[-1]) != 54:
            continue
        for augmentation_type in AUGMENTATION_TYPES:
            augmented = augment_feature_sequence(features, augmentation_type, seed)
            previews.append(
                {
                    "parent_candidate_id": str(candidate.get("candidate_id", "")),
                    "clip_id": str(candidate.get("clip_id", "")),
                    "source_type": "synthetic_preview",
                    "augmentation_type": augmentation_type,
                    "feature_schema": "keypoint_bbox54",
                    "feature_dim": int(augmented.shape[-1]),
                    "sequence_length": int(augmented.shape[0]),
                    "auto_merge_to_train": False,
                    "review_status": "preview_only",
                }
            )
    return previews


def augment_feature_sequence(features: np.ndarray, augmentation_type: str, seed: int) -> np.ndarray:
    rng = random.Random(f"{seed}:{augmentation_type}")
    augmented = np.asarray(features, dtype=np.float32).copy()
    if augmentation_type == "keypoint_noise":
        noise = np.asarray([[rng.uniform(-0.01, 0.01) for _ in range(51)] for _ in range(augmented.shape[0])])
        augmented[:, :51] = np.clip(augmented[:, :51] + noise.astype(np.float32), 0.0, 1.0)
    elif augmentation_type == "bbox_scale_jitter":
        factor = rng.uniform(0.9, 1.1)
        augmented[:, 51] = np.clip(augmented[:, 51] * factor, 0.0, 1.0)
        augmented[:, 52] = np.clip(augmented[:, 52] * factor, 0.0, 1.0)
        augmented[:, 53] = augmented[:, 51] * augmented[:, 52]
    elif augmentation_type == "bbox_aspect_jitter":
        w_factor = rng.uniform(0.9, 1.1)
        h_factor = rng.uniform(0.9, 1.1)
        augmented[:, 51] = np.clip(augmented[:, 51] * w_factor, 0.0, 1.0)
        augmented[:, 52] = np.clip(augmented[:, 52] * h_factor, 0.0, 1.0)
        augmented[:, 53] = augmented[:, 51] * augmented[:, 52]
    elif augmentation_type == "confidence_drop":
        augmented[:, 2:51:3] = np.clip(augmented[:, 2:51:3] * rng.uniform(0.65, 0.9), 0.0, 1.0)
    elif augmentation_type == "frame_drop" and augmented.shape[0] > 1:
        drop_index = rng.randrange(augmented.shape[0])
        replacement = augmented[drop_index - 1 if drop_index > 0 else 1]
        augmented[drop_index] = replacement
    elif augmentation_type == "temporal_jitter" and augmented.shape[0] > 1:
        augmented = np.roll(augmented, shift=1, axis=0).astype(np.float32)
    elif augmentation_type == "horizontal_flip":
        augmented[:, 0:51:3] = 1.0 - augmented[:, 0:51:3]
    return augmented.astype(np.float32)
