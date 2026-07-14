from __future__ import annotations

from typing import Final


KEYPOINT_COUNT: Final = 17
KEYPOINT_FEATURES_PER_POINT: Final = 3
KEYPOINT51_INPUT_SIZE: Final = KEYPOINT_COUNT * KEYPOINT_FEATURES_PER_POINT
KEYPOINT_MOTION54_INPUT_SIZE: Final = KEYPOINT51_INPUT_SIZE + 3
KEYPOINT_BBOX54_INPUT_SIZE: Final = KEYPOINT51_INPUT_SIZE + 3
KEYPOINT51_SCHEMA_VERSION: Final = "keypoint51"
KEYPOINT_MOTION54_SCHEMA_VERSION: Final = "keypoint_motion54"
KEYPOINT_BBOX54_SCHEMA_VERSION: Final = "keypoint_bbox54"
KEYPOINT_MOTION54_EXTRA_FEATURE_NAMES: Final = (
    "center_drop",
    "velocity",
    "torso_angle_norm",
)
KEYPOINT_BBOX54_EXTRA_FEATURE_NAMES: Final = (
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
)
KNOWN_KEYPOINT_SCHEMAS: Final = frozenset(
    {
        KEYPOINT51_SCHEMA_VERSION,
        KEYPOINT_MOTION54_SCHEMA_VERSION,
        KEYPOINT_BBOX54_SCHEMA_VERSION,
    }
)


def keypoint51_feature_names() -> list[str]:
    return [f"kp{index}_{coord}" for index in range(KEYPOINT_COUNT) for coord in ("x", "y", "conf")]


def keypoint_motion54_feature_names() -> list[str]:
    return [*keypoint51_feature_names(), *KEYPOINT_MOTION54_EXTRA_FEATURE_NAMES]


def keypoint_bbox54_feature_names() -> list[str]:
    return [*keypoint51_feature_names(), *KEYPOINT_BBOX54_EXTRA_FEATURE_NAMES]


def feature_dim_for_schema(schema_version: str) -> int:
    match schema_version:
        case "keypoint51":
            return KEYPOINT51_INPUT_SIZE
        case "keypoint_motion54":
            return KEYPOINT_MOTION54_INPUT_SIZE
        case "keypoint_bbox54":
            return KEYPOINT_BBOX54_INPUT_SIZE
        case _:
            raise ValueError(f"Unknown feature_schema_version={schema_version!r}")


def feature_names_for_schema(schema_version: str) -> list[str]:
    match schema_version:
        case "keypoint51":
            return keypoint51_feature_names()
        case "keypoint_motion54":
            return keypoint_motion54_feature_names()
        case "keypoint_bbox54":
            return keypoint_bbox54_feature_names()
        case _:
            raise ValueError(f"Unknown feature_schema_version={schema_version!r}")
