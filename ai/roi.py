from __future__ import annotations

import json

import cv2
import numpy as np


def parse_polygon_points(polygon_points_str: str) -> list[list[float]]:
    """JSON 문자열([[x,y], ...], 0~1 정규화)을 파싱. 실패 시 빈 리스트 반환."""
    if not polygon_points_str:
        return []
    try:
        data = json.loads(polygon_points_str)
        if isinstance(data, list) and len(data) >= 3:
            return data
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def build_roi_mask(
    polygon_normalized: list[list[float]],
    frame_height: int,
    frame_width: int,
) -> np.ndarray | None:
    """0~1 정규화 좌표 폴리곤을 프레임 크기의 OpenCV 마스크로 변환.

    반환값: uint8 마스크 (ROI 내부=255, 외부=0), 유효한 폴리곤이 없으면 None.
    """
    if not polygon_normalized or len(polygon_normalized) < 3:
        return None
    points = np.array(
        [[round(x * frame_width), round(y * frame_height)] for x, y in polygon_normalized],
        dtype=np.int32,
    )
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def combine_roi_masks(
    roi_configs: list[dict],
    frame_height: int,
    frame_width: int,
) -> np.ndarray | None:
    """여러 ROI 설정의 마스크를 합산(union)해서 하나의 마스크로 반환.

    모든 ROI 영역의 합집합으로 마스크를 만들어 어느 ROI 안에 있으면 탐지 대상이 된다.
    유효한 ROI가 없으면 None 반환 (전체 프레임 탐지).
    """
    if not roi_configs:
        return None

    combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
    valid = 0
    for roi in roi_configs:
        polygon = parse_polygon_points(roi.get("polygonPoints", ""))
        if not polygon:
            continue
        points = np.array(
            [[round(x * frame_width), round(y * frame_height)] for x, y in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(combined, [points], 255)
        valid += 1

    if valid == 0:
        return None
    return combined


def apply_roi_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """마스크를 프레임에 적용. mask가 None이면 원본 반환 (ROI 없음 = 전체 분석)."""
    if mask is None:
        return frame
    return cv2.bitwise_and(frame, frame, mask=mask)


def find_boxes_in_exit_zone(boxes: list, mask: np.ndarray | None) -> set:
    """트래킹된 박스 중 center가 EXIT ROI 마스크 안에 있는 track_id 집합 반환."""
    if mask is None:
        return set()
    h, w = mask.shape[:2]
    result = set()
    for box in boxes:
        track_id = box.get("track_id")
        if track_id is None:
            continue
        cx = int((float(box.get("x1", 0)) + float(box.get("x2", 0))) / 2)
        cy = int((float(box.get("y1", 0)) + float(box.get("y2", 0))) / 2)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        if mask[cy, cx] > 0:
            result.add(int(float(str(track_id))))
    return result
