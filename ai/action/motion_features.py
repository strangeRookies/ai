"""keypoint_motion54 motion append (raw per-frame displacement; no Δt normalization)."""
from __future__ import annotations

import numpy as np

# Allow a few latest-frame queue drops without treating them as recovery-scale gaps.
# Measured recovery miss was ~67 frames; ordinary drops are typically 1–2.
DEFAULT_MAX_CONTINUOUS_FRAME_STEP = 3
# Used only when frame_ids/frame_idxs are unavailable.
DEFAULT_MAX_CONTINUOUS_TIME_GAP_MS = 150


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_motion_discontinuity_mask(
    sequence: dict | None,
    seq_len: int,
    *,
    max_frame_step: int = DEFAULT_MAX_CONTINUOUS_FRAME_STEP,
    max_time_gap_ms: int = DEFAULT_MAX_CONTINUOUS_TIME_GAP_MS,
) -> np.ndarray:
    """Return per-sample mask where center_drop/velocity must be forced to 0.

    Priority:
      1. Explicit recovery / motion discontinuity markers on the sample
      2. Valid consecutive frame_ids or frame_idxs → frame step only
         (step <= max_frame_step keeps continuity even if timestamp is large;
          step > max_frame_step or non-monotonic step → discontinuity)
      3. Else fallback to captured_at_ms gap > max_time_gap_ms
         (non-monotonic / non-positive delta → discontinuity)
    """
    mask = np.zeros(int(seq_len), dtype=bool)
    if seq_len <= 0:
        return mask
    mask[0] = True
    if sequence is None or seq_len < 2:
        return mask

    detections = list(sequence.get("detections") or [])
    frame_ids = list(sequence.get("frame_ids") or [])
    frame_idxs = list(sequence.get("frame_idxs") or [])
    times = list(sequence.get("sample_captured_at_ms") or [])

    for index in range(1, seq_len):
        det = detections[index] if index < len(detections) else None
        if isinstance(det, dict) and (
            det.get("recovery_relink")
            or det.get("incident_recovery_relink")
            or det.get("motion_discontinuity")
        ):
            mask[index] = True
            continue

        prev_id = _int_or_none(frame_ids[index - 1] if index - 1 < len(frame_ids) else None)
        cur_id = _int_or_none(frame_ids[index] if index < len(frame_ids) else None)
        prev_idx = _int_or_none(frame_idxs[index - 1] if index - 1 < len(frame_idxs) else None)
        cur_idx = _int_or_none(frame_idxs[index] if index < len(frame_idxs) else None)

        # Prefer frame_ids; else frame_idxs. If either pair is valid, do not use time.
        frame_pair = None
        if prev_id is not None and cur_id is not None:
            frame_pair = (prev_id, cur_id)
        elif prev_idx is not None and cur_idx is not None:
            frame_pair = (prev_idx, cur_idx)

        if frame_pair is not None:
            step = frame_pair[1] - frame_pair[0]
            if step <= 0 or step > int(max_frame_step):
                mask[index] = True
            # step in 1..max_frame_step → continuous (ignore timestamp)
            continue

        prev_t = _int_or_none(times[index - 1] if index - 1 < len(times) else None)
        cur_t = _int_or_none(times[index] if index < len(times) else None)
        if prev_t is not None and cur_t is not None:
            delta = cur_t - prev_t
            if delta <= 0 or delta > int(max_time_gap_ms):
                mask[index] = True
    return mask


def append_motion_features(base_features, discontinuity_mask=None):
    """
    Append motion features (center_drop, velocity, torso_angle) to keypoint features.

    Continuous-frame semantics are unchanged raw hip-midpoint displacements:
      center_drop[t] = hip_mid_y[t] - hip_mid_y[t-1]
      velocity[t]    = hypot(dx, dy)
      torso_angle_norm from current-frame shoulder→hip vector
    First sample and any discontinuity sample force center_drop=velocity=0.
    torso_angle_norm is always recomputed for the current frame.

    Args:
        base_features: (seq_len, 51) normalized keypoints
        discontinuity_mask: optional bool array (seq_len,); True → zero motion deltas
    Returns:
        np.ndarray shape (seq_len, 54)
    """
    seq_len = int(base_features.shape[0])
    if seq_len == 0:
        return base_features

    motion_features = np.zeros((seq_len, 3), dtype=np.float32)

    # YOLO Pose (COCO 17): 5 LShoulder, 6 RShoulder, 11 LHip, 12 RHip
    def get_xy(kp_idx):
        x = base_features[:, kp_idx * 3]
        y = base_features[:, kp_idx * 3 + 1]
        conf = base_features[:, kp_idx * 3 + 2]
        return x, y, conf

    ls_x, ls_y, ls_conf = get_xy(5)
    rs_x, rs_y, rs_conf = get_xy(6)
    lh_x, lh_y, lh_conf = get_xy(11)
    rh_x, rh_y, rh_conf = get_xy(12)

    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_mid_y = (ls_y + rs_y) / 2.0
    hip_mid_x = (lh_x + rh_x) / 2.0
    hip_mid_y = (lh_y + rh_y) / 2.0

    dx = hip_mid_x - shoulder_mid_x
    dy = hip_mid_y - shoulder_mid_y
    angles = np.arctan2(dy, dx)
    torso_angle_norm = (angles + np.pi) / (2 * np.pi)
    motion_features[:, 2] = torso_angle_norm

    center_drop = np.zeros(seq_len, dtype=np.float32)
    velocity = np.zeros(seq_len, dtype=np.float32)

    if seq_len > 1:
        diff_y = hip_mid_y[1:] - hip_mid_y[:-1]
        diff_x = hip_mid_x[1:] - hip_mid_x[:-1]
        center_drop[1:] = diff_y
        velocity[1:] = np.sqrt(diff_x**2 + diff_y**2)

    if discontinuity_mask is not None:
        mask = np.asarray(discontinuity_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != seq_len:
            raise ValueError(
                f"discontinuity_mask length {mask.shape[0]} != seq_len {seq_len}"
            )
        center_drop = np.where(mask, 0.0, center_drop).astype(np.float32)
        velocity = np.where(mask, 0.0, velocity).astype(np.float32)

    motion_features[:, 0] = center_drop
    motion_features[:, 1] = velocity

    return np.concatenate([base_features, motion_features], axis=1).astype(np.float32)
