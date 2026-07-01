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
    ):
        self.pre_event_frame_count = max(1, int(pre_event_frame_count))
        self.post_event_frame_count = max(1, int(post_event_frame_count))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.fps = float(fps or 30.0)
        self.output_dir = output_dir
        #원형 큐 초기화(생성)
        self.pre_event_buffer = CircularFrameBuffer(self.pre_event_frame_count)
        self._last_event_at: dict[tuple[str, str], float] = {}
        self._active_event: dict[str, Any] | None = None

    def add_frame(self, frame):
        self.pre_event_buffer.append(frame) # <-- 평상시 실시간 프레임을 원형 큐에 추가하는 부분
        if self._active_event is None:
            return None

        self._active_event["post_frames"].append(self._copy_frame(frame))
        # 감지 이후 프레임 수(예: 150프레임 = 5초)가 충족되었는지 검사
        if len(self._active_event["post_frames"]) < self.post_event_frame_count:
            return None

        event = self._active_event
        self._active_event = None
        # 이전 5초(pre)와 이후 5초(post) 프레임을 합쳐서 인코딩 태스크 생성
        return EventClipTask(
            event_type=event["event_type"],
            camera_id=event["camera_id"],
            frames=event["pre_frames"] + event["post_frames"], # <-- 합쳐지는 부분
            fps=self.fps,
            output_dir=self.output_dir,
            metadata=event["metadata"],
            created_at=event["created_at"],
        )

    def trigger_event(self, event_type, camera_id, metadata=None, now=None):
        now = time.time() if now is None else float(now)
        key = (str(camera_id), str(event_type))
        if self._active_event is not None:
            return False
        if now - self._last_event_at.get(key, 0.0) < self.cooldown_seconds:
            return False

        self._last_event_at[key] = now
        pre_frames = [self._copy_frame(frame) for frame in self.pre_event_buffer.snapshot()]
        self._active_event = {
            "event_type": str(event_type),
            "camera_id": str(camera_id),
            "metadata": dict(metadata or {}),
            "pre_frames": pre_frames,
            "post_frames": [],
            "created_at": now,
        }
        return True

    @staticmethod
    def _copy_frame(frame):
        copy = getattr(frame, "copy", None)
        if callable(copy):
            return copy()
        return frame
