"""Event-frame face blur before Snapshot Assist upload (side-channel)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.events.clip_worker import _blur_region, _face_box_from_keypoints, _upper_body_fallback_box


def deidentify_event_frame(
    frame_bgr: Any,
    *,
    bbox: Any = None,
    keypoints: Any = None,
) -> Any:
    """Return a copy of frame_bgr with a single-subject face region blurred when bbox/keypoints allow."""
    if frame_bgr is None:
        return frame_bgr
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr

    image = frame_bgr.copy()
    height, width = image.shape[:2]
    box: dict[str, Any] = {}
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        box["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    if keypoints is not None:
        box["keypoints"] = keypoints

    if not box.get("bbox") and not box.get("keypoints"):
        return image

    region = _face_box_from_keypoints(box, width, height)
    if region is None:
        region = _upper_body_fallback_box(box, width, height)
    if region is not None:
        _blur_region(cv2, image, region)
    return image