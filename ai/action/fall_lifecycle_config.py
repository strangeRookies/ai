"""Build FaintEventPostProcessor from argparse/Namespace with OBJECTIVE tuning knobs."""

from __future__ import annotations

from ai.action.faint_post_processing import (
    DEFAULT_BLOCK_UPRIGHT_FAINT,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    DEFAULT_PERSISTENT_DELAY_SEC,
    DEFAULT_PERSISTENT_REPEAT_SEC,
    DEFAULT_RECOVER_CONSECUTIVE,
    DEFAULT_REQUIRE_UPRIGHT_TO_LYING,
    FaintEventPostProcessor,
)


def resolved_faint_threshold(args) -> float:
    for name in ("faint_threshold", "action_threshold"):
        val = getattr(args, name, None)
        if val is not None:
            return float(val)
    return float(DEFAULT_FAINT_THRESHOLD)


def resolved_fall_threshold(args) -> float:
    val = getattr(args, "fall_threshold", None)
    if val is not None:
        return float(val)
    return resolved_faint_threshold(args)


def resolved_consecutive(args) -> int:
    for name in ("consecutive_required", "min_consecutive_faint"):
        val = getattr(args, name, None)
        if val is not None:
            return int(val)
    return int(DEFAULT_MIN_CONSECUTIVE_FAINT)


def resolved_cooldown(args) -> float:
    for name in ("cooldown_sec", "camera_cooldown_seconds"):
        val = getattr(args, name, None)
        if val is not None:
            return float(val)
    return float(DEFAULT_CAMERA_COOLDOWN_SECONDS)


def build_faint_post_processor_from_args(args) -> FaintEventPostProcessor:
    return FaintEventPostProcessor(
        min_consecutive_faint=resolved_consecutive(args),
        cooldown_seconds=resolved_cooldown(args),
        use_fall_state_machine=bool(getattr(args, "use_fall_state_machine", True)),
        recover_consecutive=int(getattr(args, "normal_recover_required", DEFAULT_RECOVER_CONSECUTIVE)),
        require_upright_to_lying=bool(
            getattr(args, "require_upright_to_lying", DEFAULT_REQUIRE_UPRIGHT_TO_LYING)
        ),
        block_upright_faint=bool(
            getattr(args, "block_upright_faint", DEFAULT_BLOCK_UPRIGHT_FAINT)
        ),
        unrecovered_after_seconds=float(
            getattr(args, "persistent_delay_sec", DEFAULT_PERSISTENT_DELAY_SEC)
        ),
        unrecovered_repeat_seconds=float(
            getattr(args, "persistent_repeat_sec", DEFAULT_PERSISTENT_REPEAT_SEC)
        ),
        lying_aspect_ratio=float(getattr(args, "lying_aspect_ratio", 1.2)),
        upright_aspect_ratio=float(getattr(args, "upright_aspect_ratio", 1.3)),
        min_keypoint_conf=float(getattr(args, "min_keypoint_conf", 0.3)),
        lying_frames_required=int(getattr(args, "lying_frames_required", 2)),
        upright_frames_required=int(getattr(args, "upright_frames_required", 2)),
        movement_low_threshold=float(getattr(args, "movement_low_threshold", 12.0)),
        faint_threshold=resolved_faint_threshold(args),
        fall_threshold=resolved_fall_threshold(args),
        track_lost_grace_sec=float(getattr(args, "track_lost_grace_sec", 3.0)),
    )
