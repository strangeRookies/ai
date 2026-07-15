import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventClipTask:
    event_type: str
    camera_id: str
    frames: list[Any]
    fps: float
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # frames와 1:1 대응되는 그 프레임 시점의 원본 탐지 결과(boxes, track_id+keypoints 포함).
    # 프레임별 얼굴 블러 위치 계산에 쓰임 (frames[i]가 없으면 해당 인덱스는 None).
    frame_boxes: list[Any] = field(default_factory=list)


class CircularFrameBuffer:
    def __init__(self, maxlen=150):
        self._frames = deque(maxlen=max(1, int(maxlen)))

    def append(self, frame):
        self._frames.append(frame)

    def snapshot(self):
        return list(self._frames)

    def __len__(self):
        return len(self._frames)


class EventClipBuffer:
    def __init__(
        self,
        pre_event_frame_count=150,
        post_event_frame_count=150,
        cooldown_seconds=10.0,
        fps=30.0,
        output_dir="clips",
        max_concurrent_events=4,
    ):
        self.pre_event_frame_count = max(1, int(pre_event_frame_count))
        self.post_event_frame_count = max(1, int(post_event_frame_count))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.fps = float(fps or 30.0)
        self.output_dir = output_dir
        self.max_concurrent_events = max(1, int(max_concurrent_events))
        #원형 큐 초기화(생성)
        self.pre_event_buffer = CircularFrameBuffer(self.pre_event_frame_count)
        self._last_event_at: dict[tuple[str, str], float] = {}
        # camera_id -> 그 카메라에서 현재 동시에 pre/post 프레임을 모으고 있는 이벤트 목록
        # (카메라당 최대 max_concurrent_events개까지 독립적으로 진행 가능)
        self._active_events: dict[str, list[dict]] = {}

    def add_frame(self, frame, boxes=None):
        self.pre_event_buffer.append((frame, boxes)) # <-- 평상시 실시간 프레임+탐지결과를 원형 큐에 추가하는 부분
        completed_tasks: list[EventClipTask] = []
        if not self._active_events:
            return completed_tasks

        for camera_id, events in list(self._active_events.items()):
            still_active = []
            for event in events:
                event["post_frames"].append(self._copy_frame(frame))
                event["post_boxes"].append(boxes)
                # 감지 이후 프레임 수(예: 150프레임 = 5초)가 충족되었는지 검사
                if len(event["post_frames"]) < self.post_event_frame_count:
                    still_active.append(event)
                    continue
                # 이전 5초(pre)와 이후 5초(post) 프레임을 합쳐서 인코딩 태스크 생성
                completed_tasks.append(
                    EventClipTask(
                        event_type=event["event_type"],
                        camera_id=event["camera_id"],
                        frames=event["pre_frames"] + event["post_frames"], # <-- 합쳐지는 부분
                        fps=self.fps,
                        output_dir=self.output_dir,
                        metadata=event["metadata"],
                        created_at=event["created_at"],
                        frame_boxes=event["pre_boxes"] + event["post_boxes"],
                    )
                )
            if still_active:
                self._active_events[camera_id] = still_active
            else:
                del self._active_events[camera_id]

        return completed_tasks

    def trigger_event(self, event_type, camera_id, metadata=None, now=None):
        now = time.time() if now is None else float(now)
        camera_key = str(camera_id)
        key = (camera_key, str(event_type))
        active_for_camera = self._active_events.get(camera_key, [])
        if len(active_for_camera) >= self.max_concurrent_events:
            return False
        if now - self._last_event_at.get(key, 0.0) < self.cooldown_seconds:
            return False

        self._last_event_at[key] = now
        snapshot = self.pre_event_buffer.snapshot()
        pre_frames = [self._copy_frame(frame) for frame, _boxes in snapshot]
        pre_boxes = [boxes for _frame, boxes in snapshot]
        new_event = {
            "event_type": str(event_type),
            "camera_id": camera_key,
            "metadata": dict(metadata or {}),
            "pre_frames": pre_frames,
            "post_frames": [],
            "pre_boxes": pre_boxes,
            "post_boxes": [],
            "created_at": now,
        }
        self._active_events.setdefault(camera_key, []).append(new_event)
        return True

    @staticmethod
    def _copy_frame(frame):
        copy = getattr(frame, "copy", None)
        if callable(copy):
            return copy()
        return frame
