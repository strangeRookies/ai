import numpy as np
import cv2
import tempfile
from pathlib import Path
from ai.roi import combine_roi_masks, apply_roi_mask, parse_polygon_points
from ai.events.clip_worker import save_clip_to_mp4, EventClipTask


def test_parse_polygon_points():
    assert parse_polygon_points("") == []
    assert parse_polygon_points("[[0.1, 0.2]]") == [] # 점 개수 부족
    parsed = parse_polygon_points("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]")
    assert len(parsed) == 3
    assert parsed[0] == [0.1, 0.2]


def test_roi_mask_combination():
    # 전체 화면 크기 다각형
    roi_configs = [
        {"polygonPoints": "[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]"}
    ]
    mask = combine_roi_masks(roi_configs, 100, 100)
    assert mask is not None
    assert mask.shape == (100, 100)
    assert np.all(mask == 255)


def test_apply_roi_mask():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    
    masked = apply_roi_mask(frame, mask)
    assert np.all(masked[10, 10] == 0)      # 마스크 0 영역은 검은색
    assert np.all(masked[50, 50] == 100)    # 마스크 255 영역은 원본 보존


def test_save_clip_to_mp4_with_blur():
    # 3채널 흰색 이미지 10장 생성
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # 중앙 부분에 검은색 사각형 생성 (이 영역을 블러할 예정)
    frame[40:60, 40:60] = 0

    # 1. 매칭되는 track/box 정보가 없으면 블러 없이 인플레이스 연산 검증
    # FFMPEG VideoWriter 에러가 발생하더라도 프레임 픽셀 변환은 그전에 인플레이스로 처리되므로,
    # 이를 try-except로 감싸 코덱 오류를 우회하고 알고리즘만 검증합니다.
    frames_no_blur = [frame.copy() for _ in range(5)]
    task_no_blur = EventClipTask(
        event_type="test_event",
        camera_id="cam_01",
        frames=frames_no_blur,
        fps=10.0,
        output_dir="tmp_clips_test",
        metadata={"track_id": 1},  # frame_boxes가 없어 매칭 실패 -> 블러 스킵
    )

    try:
        save_clip_to_mp4(task_no_blur)
    except Exception:
        pass

    # 블러 처리가 들어가지 않았으므로 중앙 영역은 여전히 완전한 검은색(0)
    assert np.mean(frames_no_blur[0][40:60, 40:60]) == 0.0

    # 2. track_id가 매칭되는 keypoint 기반 얼굴 박스로 인플레이스 연산 검증
    frames_with_blur = [frame.copy() for _ in range(5)]
    keypoints = [
        {"x": 50.0, "y": 48.0, "confidence": 0.9},  # 0 코
        {"x": 45.0, "y": 45.0, "confidence": 0.9},  # 1 왼눈
        {"x": 55.0, "y": 45.0, "confidence": 0.9},  # 2 오른눈
        {"x": 42.0, "y": 47.0, "confidence": 0.9},  # 3 왼귀
        {"x": 58.0, "y": 47.0, "confidence": 0.9},  # 4 오른귀
    ]
    box = {"track_id": 1, "x1": 30.0, "y1": 30.0, "x2": 70.0, "y2": 90.0, "keypoints": keypoints}
    task_with_blur = EventClipTask(
        event_type="test_event",
        camera_id="cam_01",
        frames=frames_with_blur,
        fps=10.0,
        output_dir="tmp_clips_test",
        metadata={"track_id": 1},
        frame_boxes=[[box] for _ in range(5)],
    )

    try:
        save_clip_to_mp4(task_with_blur)
    except Exception:
        pass

    # 인플레이스 블러 필터를 통해 완벽한 검은색(0) 영역 내부가 흰색(255) 픽셀과 Gaussian Blur로 합성되어
    # 평균 값이 0.0보다 확실히 커졌는지를 검증합니다.
    assert np.mean(frames_with_blur[0][40:60, 40:60]) > 0.0

