"""프레임별 얼굴 keypoint 기반 블러 (clip_worker._apply_per_frame_face_blur 등) 유닛 테스트.

시나리오는 2026-07-15 세션에서 요청받은 표 그대로:
1) 서 있는 자세, keypoint 신뢰도 높음 -> 1순위(keypoint) 정상 산출
2) 쓰러진 자세(옆으로 누움), keypoint 신뢰도 낮음 -> 2순위(상위 %) 폴백, 오차 실측
3) 완전히 엎드린 자세(얼굴이 바닥) -> 2순위 결과가 실제로 머리 근처인지 확인
4) 1~2프레임만 탐지 실패 -> 3순위(직전 프레임 재사용) 정상 동작
5) 5프레임 초과 연속 탐지 실패 -> 3순위 제한, 최종 포기(스킵)로 전환
6) 좌->우 빠른 이동 -> 프레임별 블러 위치가 실제 이동을 따라가는지
"""

import numpy as np
import pytest

from ai.events.clip_worker import (
    _MAX_STALE_FACE_BOX_FRAMES,
    _apply_per_frame_face_blur,
    _face_box_from_keypoints,
    _find_track_box,
    _upper_body_fallback_box,
)

WIDTH, HEIGHT = 400, 400
TRACK_ID = 7


def _kp(x, y, conf):
    return {"x": float(x), "y": float(y), "confidence": float(conf)}


def _empty_kp():
    return _kp(0.0, 0.0, 0.0)


def _box(track_id, x1, y1, x2, y2, keypoints=None):
    return {
        "track_id": track_id,
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "keypoints": keypoints,
    }


def _standing_keypoints(head_x, head_y):
    """서 있는 자세: 코/눈/귀가 bbox 상단에 고신뢰도로 잡힘."""
    kps = [_empty_kp()] * 17
    kps[0] = _kp(head_x, head_y, 0.9)  # nose
    kps[1] = _kp(head_x - 5, head_y - 3, 0.85)  # L eye
    kps[2] = _kp(head_x + 5, head_y - 3, 0.85)  # R eye
    kps[3] = _kp(head_x - 10, head_y, 0.8)  # L ear
    kps[4] = _kp(head_x + 10, head_y, 0.8)  # R ear
    kps[5] = _kp(head_x - 20, head_y + 40, 0.9)  # L shoulder
    kps[6] = _kp(head_x + 20, head_y + 40, 0.9)  # R shoulder
    kps[11] = _kp(head_x - 15, head_y + 150, 0.7)  # L hip
    kps[12] = _kp(head_x + 15, head_y + 150, 0.7)  # R hip
    return kps


def _low_confidence_head_keypoints(head_x, head_y, conf=0.15):
    """머리 keypoint는 있지만 신뢰도가 임계값(0.3) 미만 -> 1순위에서 필터링됨."""
    kps = [_empty_kp()] * 17
    kps[0] = _kp(head_x, head_y, conf)
    kps[1] = _kp(head_x - 4, head_y - 2, conf)
    kps[2] = _kp(head_x + 4, head_y - 2, conf)
    return kps


class TestScenario1Standing:
    def test_high_confidence_keypoints_produce_face_box(self):
        keypoints = _standing_keypoints(head_x=200, head_y=80)
        box = _box(TRACK_ID, 150, 60, 250, 350, keypoints)

        region = _face_box_from_keypoints(box, WIDTH, HEIGHT)

        assert region is not None
        x1, y1, x2, y2 = region
        # 얼굴 박스가 실제 머리(200,80) 주변을 감싸야 하고, 어깨/골반까지 내려가면 안 됨
        assert x1 < 200 < x2
        assert y1 < 80 < y2
        assert y2 < 150  # 어깨(120) 근처까지 과도하게 안 내려감


class TestScenario2LyingOnSide:
    def test_low_confidence_head_falls_back_to_upper_band_with_measurable_error(self):
        # 옆으로 누운 자세: bbox는 가로로 넓고, 실제 머리는 왼쪽 끝 쪽에 있음
        actual_head_x, actual_head_y = 110, 224
        keypoints = _low_confidence_head_keypoints(actual_head_x, actual_head_y)
        box = _box(TRACK_ID, 100, 200, 300, 260, keypoints)

        assert _face_box_from_keypoints(box, WIDTH, HEIGHT) is None  # 1순위 실패해야 정상

        region = _upper_body_fallback_box(box, WIDTH, HEIGHT)
        assert region is not None
        x1, y1, x2, y2 = region

        # 실제 머리가 이 폴백 박스에 얼마나 벗어나는지 측정
        y_error = max(0, actual_head_y - y2) if actual_head_y > y2 else max(0, y1 - actual_head_y)
        x_covered = x1 <= actual_head_x <= x2
        print(f"[scenario2] fallback_box={region} actual_head=({actual_head_x},{actual_head_y}) "
              f"y_error_px={y_error} x_covered={x_covered}")

        # 가로 폭은 bbox 전체라 실제 머리 x좌표는 대개 커버되지만, 세로 밴드는 살짝 벗어날 수 있음
        assert x_covered
        assert y_error <= 10  # 참고용 허용치 -- 실측 오차를 리포트에 남기기 위한 기록성 단언


class TestScenario3Prone:
    def test_upper_band_may_miss_head_when_body_lies_horizontally(self):
        # 완전히 엎드린 자세: 몸이 수평이라 "상위 %" 개념(세로축 상단)이 머리-발 축과 안 맞음
        # 실제 머리는 bbox 오른쪽 끝 근처(예: 발이 왼쪽)
        actual_head_x, actual_head_y = 290, 230
        box = _box(TRACK_ID, 100, 200, 300, 260, keypoints=None)  # keypoint 자체가 아예 없음

        assert _face_box_from_keypoints(box, WIDTH, HEIGHT) is None

        region = _upper_body_fallback_box(box, WIDTH, HEIGHT)
        assert region is not None
        x1, y1, x2, y2 = region

        head_in_region = (x1 <= actual_head_x <= x2) and (y1 <= actual_head_y <= y2)
        print(f"[scenario3] fallback_box={region} actual_head=({actual_head_x},{actual_head_y}) "
              f"head_in_region={head_in_region}")

        # 이 케이스는 실패가 "기대되는" 시나리오임: 수평으로 누운 자세에서 상위 %는
        # 머리가 아니라 몸통(등/허리) 근처를 가리킬 수 있음을 문서화하기 위한 테스트.
        # (아래 assert는 일부러 실패를 허용하지 않고 현재 동작을 고정해서 회귀 감지용으로 둠)
        assert head_in_region is False, (
            "예상대로 상위 % 폴백이 가로로 누운 자세에서 머리를 못 잡음 -- "
            "본문 리포트의 '개선 방향' 참고 (자세 인식 기반 밴드 방향 전환 필요)"
        )


class TestScenario4ShortDropout:
    def test_one_to_two_missed_frames_reuse_last_known_position(self):
        keypoints_a = _standing_keypoints(head_x=200, head_y=80)
        box_a = _box(TRACK_ID, 150, 60, 250, 350, keypoints_a)

        frames = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(4)]
        frame_boxes = [
            [box_a],  # frame 0: 정상 탐지
            None,     # frame 1: 탐지 누락
            None,     # frame 2: 탐지 누락 (2프레임 연속, 한도 5 이내)
            [box_a],  # frame 3: 다시 정상 탐지
        ]

        tiers = _apply_per_frame_face_blur(_cv2(), frames, frame_boxes, TRACK_ID, WIDTH, HEIGHT)

        assert tiers["keypoint"] == 2
        assert tiers["stale_reuse"] == 2
        assert tiers["skipped"] == 0


class TestScenario5LongDropout:
    def test_more_than_five_consecutive_misses_gives_up_after_limit(self):
        keypoints_a = _standing_keypoints(head_x=200, head_y=80)
        box_a = _box(TRACK_ID, 150, 60, 250, 350, keypoints_a)

        num_missing = 6  # 5프레임 한도를 넘기기 위해 6프레임 연속 누락
        frames = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(1 + num_missing)]
        frame_boxes = [[box_a]] + [None] * num_missing

        tiers = _apply_per_frame_face_blur(_cv2(), frames, frame_boxes, TRACK_ID, WIDTH, HEIGHT)

        assert tiers["keypoint"] == 1
        assert tiers["stale_reuse"] == _MAX_STALE_FACE_BOX_FRAMES  # 최대 5프레임까지만 재사용
        assert tiers["skipped"] == num_missing - _MAX_STALE_FACE_BOX_FRAMES  # 나머지는 포기


class TestScenario6FastMovement:
    def test_face_box_follows_left_to_right_movement(self):
        # 5프레임에 걸쳐 머리가 왼쪽에서 오른쪽으로 빠르게 이동
        head_positions = [(60, 100), (120, 100), (180, 100), (240, 100), (300, 100)]
        frames = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in head_positions]
        frame_boxes = []
        for hx, hy in head_positions:
            keypoints = _standing_keypoints(head_x=hx, head_y=hy)
            box = _box(TRACK_ID, hx - 50, hy - 20, hx + 50, hy + 270, keypoints)
            frame_boxes.append([box])

        computed_centers = []
        for idx in range(len(frames)):
            matched = _find_track_box(frame_boxes[idx], TRACK_ID)
            region = _face_box_from_keypoints(matched, WIDTH, HEIGHT)
            assert region is not None
            x1, y1, x2, y2 = region
            computed_centers.append((x1 + x2) / 2.0)

        # 고정된 옛 위치에 머무르지 않고 프레임마다 오른쪽으로 이동해야 함
        for earlier, later in zip(computed_centers, computed_centers[1:]):
            assert later > earlier

        # 첫 프레임과 마지막 프레임의 위치 차이가 실제 이동 거리와 비슷한 크기여야 함
        assert (computed_centers[-1] - computed_centers[0]) > 150


def _cv2():
    import cv2
    return cv2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
