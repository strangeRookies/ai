"""Bbox / keypoint posture estimator for Fall lifecycle (Phase B).

Labels:
- upright_like
- lying_like
- unknown
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class PostureEstimate:
    label: str
    aspect_wh: float | None = None  # width / height
    aspect_hw: float | None = None  # height / width
    shoulder_hip_angle_deg: float | None = None
    vertical_spread_norm: float | None = None
    center_y: float | None = None
    center_y_delta: float | None = None
    upright_to_lying_transition: bool = False
    confidence: float = 0.0
    reason: str | None = None


def _bbox_xyxy(detection: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if all(k in detection for k in ("x1", "y1", "x2", "y2")):
        return (
            float(detection["x1"]),
            float(detection["y1"]),
            float(detection["x2"]),
            float(detection["y2"]),
        )
    return None


def _keypoints(detection: Mapping[str, Any]) -> list[dict[str, float]] | None:
    kps = detection.get("keypoints")
    if not kps:
        return None
    return list(kps)


def _kp_point(keypoints: Sequence[Mapping[str, Any]], index: int, conf_min: float) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    item = keypoints[index]
    conf = float(item.get("confidence", item.get("conf", 0.0)) or 0.0)
    if conf < conf_min:
        return None
    return float(item["x"]), float(item["y"])


def shoulder_hip_angle_degrees(keypoints: Sequence[Mapping[str, Any]], conf_min: float = 0.25) -> float | None:
    """Angle of shoulder-hip line vs horizontal. ~0° horizontal (lying-like), ~90° vertical (upright)."""
    # COCO: L/R shoulder 5,6  L/R hip 11,12
    ls = _kp_point(keypoints, 5, conf_min)
    rs = _kp_point(keypoints, 6, conf_min)
    lh = _kp_point(keypoints, 11, conf_min)
    rh = _kp_point(keypoints, 12, conf_min)
    if None in (ls, rs, lh, rh):
        return None
    assert ls and rs and lh and rh
    shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # angle from horizontal: 0 = horizontal, 90 = vertical
    angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
    return angle


def vertical_spread_normalized(keypoints: Sequence[Mapping[str, Any]], bbox_h: float, conf_min: float = 0.2) -> float | None:
    ys = []
    for item in keypoints:
        conf = float(item.get("confidence", item.get("conf", 0.0)) or 0.0)
        if conf < conf_min:
            continue
        ys.append(float(item["y"]))
    if len(ys) < 2 or bbox_h <= 1e-6:
        return None
    return (max(ys) - min(ys)) / max(bbox_h, 1.0)


class PostureEstimator:
    """Per-track posture history for upright/lying and transition detection."""

    def __init__(
        self,
        *,
        aspect_lying_wh: float = 1.2,
        aspect_upright_hw: float = 1.3,
        angle_lying_max_deg: float = 40.0,
        angle_upright_min_deg: float = 55.0,
        vertical_spread_upright_min: float = 0.35,
        history_len: int = 45,
        upright_window: int = 30,
        conf_min: float = 0.3,
        lying_frames_required: int = 2,
        upright_frames_required: int = 2,
        movement_low_threshold: float = 12.0,
    ):
        self.aspect_lying_wh = float(aspect_lying_wh)
        self.aspect_upright_hw = float(aspect_upright_hw)
        self.angle_lying_max_deg = float(angle_lying_max_deg)
        self.angle_upright_min_deg = float(angle_upright_min_deg)
        self.vertical_spread_upright_min = float(vertical_spread_upright_min)
        self.history_len = max(2, int(history_len))
        self.upright_window = max(1, int(upright_window))
        self.conf_min = float(conf_min)
        self.lying_frames_required = max(1, int(lying_frames_required))
        self.upright_frames_required = max(1, int(upright_frames_required))
        self.movement_low_threshold = float(movement_low_threshold)
        self._history: dict[str, deque[str]] = {}
        self._center_y: dict[str, float] = {}

    def reset_all(self) -> None:
        self._history.clear()
        self._center_y.clear()

    def reset_track(self, track_key: str) -> None:
        self._history.pop(track_key, None)
        self._center_y.pop(track_key, None)

    def estimate(self, track_key: str, detection: Mapping[str, Any] | None, timestamp: float | None = None) -> PostureEstimate:
        if not detection:
            return PostureEstimate(label="unknown", reason="no_detection", confidence=0.0)

        # Prefer detector pose_horizontal if present
        pose_horizontal = detection.get("pose_horizontal")
        xyxy = _bbox_xyxy(detection)
        aspect_wh = aspect_hw = center_y = None
        center_y_delta = None
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            w = max(1e-6, x2 - x1)
            h = max(1e-6, y2 - y1)
            aspect_wh = w / h
            aspect_hw = h / w
            center_y = (y1 + y2) / 2.0
            prev_y = self._center_y.get(track_key)
            if prev_y is not None:
                center_y_delta = center_y - prev_y
            self._center_y[track_key] = center_y

        kps = _keypoints(detection)
        angle = shoulder_hip_angle_degrees(kps, self.conf_min) if kps else None
        v_spread = None
        if kps is not None and xyxy is not None:
            v_spread = vertical_spread_normalized(kps, max(1e-6, xyxy[3] - xyxy[1]), self.conf_min)

        lying_votes = 0
        upright_votes = 0
        reasons: list[str] = []

        if pose_horizontal is True:
            lying_votes += 2
            reasons.append("pose_horizontal")
        elif pose_horizontal is False:
            upright_votes += 1
            reasons.append("pose_not_horizontal")

        if aspect_wh is not None and aspect_wh >= self.aspect_lying_wh:
            lying_votes += 1
            reasons.append("aspect_wh_lying")
        if aspect_hw is not None and aspect_hw >= self.aspect_upright_hw:
            upright_votes += 1
            reasons.append("aspect_hw_upright")

        if angle is not None:
            if angle <= self.angle_lying_max_deg:
                lying_votes += 2
                reasons.append("torso_angle_lying")
            elif angle >= self.angle_upright_min_deg:
                upright_votes += 2
                reasons.append("torso_angle_upright")

        if v_spread is not None:
            if v_spread >= self.vertical_spread_upright_min:
                upright_votes += 1
                reasons.append("vertical_spread_upright")
            elif v_spread < self.vertical_spread_upright_min * 0.55:
                lying_votes += 1
                reasons.append("vertical_spread_lying")

        if lying_votes > upright_votes and lying_votes >= 1:
            label = "lying_like"
            conf = min(1.0, 0.35 + 0.15 * lying_votes)
        elif upright_votes > lying_votes and upright_votes >= 1:
            label = "upright_like"
            conf = min(1.0, 0.35 + 0.15 * upright_votes)
        else:
            label = "unknown"
            conf = 0.2

        hist = self._history.setdefault(track_key, deque(maxlen=self.history_len))
        # Append raw instantaneous label first, then stabilize outward label.
        raw_label = label
        hist.append(raw_label)
        recent = list(hist)
        # Consecutive trailing frames of the raw class (including current).
        trailing_lying = 0
        for item in reversed(recent):
            if item == "lying_like":
                trailing_lying += 1
            else:
                break
        trailing_upright = 0
        for item in reversed(recent):
            if item == "upright_like":
                trailing_upright += 1
            else:
                break

        # Stability gate: do not emit lying_like until N consecutive lying frames.
        if raw_label == "lying_like" and trailing_lying < self.lying_frames_required:
            label = "unknown"
            conf = min(conf, 0.25)
            reasons.append(f"lying_unstable_{trailing_lying}_of_{self.lying_frames_required}")
        elif raw_label == "upright_like" and trailing_upright < self.upright_frames_required:
            label = "unknown"
            conf = min(conf, 0.25)
            reasons.append(f"upright_unstable_{trailing_upright}_of_{self.upright_frames_required}")

        recent_upright = sum(1 for item in recent[-self.upright_window :] if item == "upright_like")
        upright_to_lying = bool(
            label == "lying_like" and recent_upright >= self.upright_frames_required
        )
        if upright_to_lying:
            reasons.append("upright_to_lying_transition")

        return PostureEstimate(
            label=label,
            aspect_wh=aspect_wh,
            aspect_hw=aspect_hw,
            shoulder_hip_angle_deg=angle,
            vertical_spread_norm=v_spread,
            center_y=center_y,
            center_y_delta=center_y_delta,
            upright_to_lying_transition=upright_to_lying,
            confidence=conf,
            reason=",".join(reasons) if reasons else None,
        )


def detection_for_track(
    detections: Sequence[Mapping[str, Any]] | None,
    track_id: Any,
) -> Mapping[str, Any] | None:
    if not detections or track_id is None:
        return None
    target = int(float(str(track_id)))
    for det in detections:
        tid = det.get("track_id")
        if tid is None:
            continue
        try:
            if int(float(str(tid))) == target:
                return det
        except (TypeError, ValueError):
            continue
    return None


def track_posture_key(camera_id: str, track_id: Any) -> str:
    return f"{camera_id}:track:{track_id}"
