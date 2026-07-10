from __future__ import annotations

from typing import Final, Literal, TypedDict


CandidateKind = Literal["hard_negative", "faint_reinforcement"]
ReviewStatus = Literal["pending", "approved", "rejected", "needs_review"]

REQUIRED_TRAINING_MANIFEST_V2_FIELDS: Final = [
    "clip_id",
    "clip_path",
    "label",
    "label_name",
    "source_type",
    "parent_clip_id",
    "review_status",
    "failure_type",
    "scenario_tag",
    "augmentation_type",
    "augmentation_config",
    "random_seed",
    "split_group_id",
    "created_at",
]

RECOMMENDED_TRAINING_MANIFEST_V2_FIELDS: Final = [
    "reviewer",
    "reviewed_at",
    "original_clip_id",
    "source_video",
    "start_frame",
    "end_frame",
    "fps",
    "width",
    "height",
    "notes",
]

REASON_ENUM: Final = {
    "bending_false_positive",
    "sitting_false_positive",
    "pickup_false_positive",
    "sofa_lying_false_positive",
    "occluded_normal_false_positive",
    "leaning_normal_false_positive",
    "lying_normal_false_positive",
    "normal_false_positive",
    "night_false_negative",
    "far_distance_false_negative",
    "occlusion_false_negative",
    "small_person_false_negative",
    "slow_fall_false_negative",
    "faint_false_negative",
    "weak_condition_augmentation_candidate",
}

SYNTHETIC_TYPES: Final = (
    "brightness",
    "noise",
    "blur",
    "compression",
    "scale_down",
    "partial_occlusion",
    "horizontal_flip",
)

LEGACY_SYNTHETIC_ALIASES: Final = {
    "brightness_down": "brightness",
    "brightness_up": "brightness",
    "gaussian_noise": "noise",
    "motion_blur": "blur",
    "compression_artifact": "compression",
    "crop": "scale_down",
    "occlusion": "partial_occlusion",
    "distance": "scale_down",
}

CANDIDATE_FIELDS: Final = [
    *REQUIRED_TRAINING_MANIFEST_V2_FIELDS,
    "reason",
    "evidence_id",
    "camera_login_id",
    "frame_id",
    "captured_at_ms",
    "predicted_label",
    "faint_prob",
    "source_video",
    "split",
]

SYNTHETIC_FIELDS: Final = [
    *REQUIRED_TRAINING_MANIFEST_V2_FIELDS,
    "synthetic_type",
    "parent_clip_path",
    "parent_split",
    "split",
    "reason",
    "evidence_id",
    "augmentation_seed",
    "estimated_visibility",
]


class ManifestSummary(TypedDict):
    rows: int
    output_csv: str
    written: bool
    counts: dict[str, int]
    class_counts: dict[str, int]
    source_type_counts: dict[str, int]


class LeakageSummary(TypedDict):
    rows: int
    synthetic_rows: int
    synthetic_ratio: float
    split_group_leaks: int
    parent_clip_leaks: int
    pending_or_rejected_rows: int
    synthetic_missing_parent_rows: int
