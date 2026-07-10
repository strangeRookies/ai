from __future__ import annotations

import time
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import numpy as np

from ai.evidence import evidence_id, latency_order_valid


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_login_id: str
    frame_id: int
    captured_at_ms: int
    frame: np.ndarray
    width: int
    height: int
    frame_idx: int
    timestamp: float
    fps: float = 0.0
    # Session boundary tags: packets older than current generation must be dropped.
    stream_run_id: str | None = None
    session_generation: int = 0


class CameraFrameQueue:
    """Bounded per-camera queue. Overflow prefers latest frames (drops oldest).

    Age-based drop uses captured_at_ms (UTC epoch ms) vs optional now_ms callback;
    latency intervals elsewhere should use monotonic clocks.
    """

    def __init__(
        self,
        camera_login_id: str,
        maxsize: int = 5,
        *,
        max_packet_age_ms: float | None = None,
        now_ms: Callable[[], int] | None = None,
    ):
        self.camera_login_id = str(camera_login_id)
        self.maxlen = max(1, int(maxsize))
        self.queue: deque[FramePacket] = deque(maxlen=self.maxlen)
        self.dropped_frame_count = 0
        self.queue_overflow_drop_total = 0
        self.aged_packet_drop_total = 0
        self.max_packet_age_ms = None if max_packet_age_ms is None else max(0.0, float(max_packet_age_ms))
        self._now_ms = now_ms or current_epoch_ms
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self.maxlen

    def put_latest(self, packet: FramePacket) -> None:
        with self._lock:
            if len(self.queue) >= self.maxlen:
                self.dropped_frame_count += 1
                self.queue_overflow_drop_total += 1
            self.queue.append(packet)

    def get_latest(self, drop_stale: bool = True) -> FramePacket | None:
        with self._lock:
            self._drop_aged_locked()
            if not self.queue:
                return None
            if drop_stale:
                dropped = len(self.queue) - 1
                if dropped > 0:
                    self.dropped_frame_count += dropped
                    for _ in range(dropped):
                        self.queue.popleft()
            return self.queue.popleft()

    def clear(self) -> int:
        """Drop all queued packets (session boundary). Returns cleared count."""
        with self._lock:
            cleared = len(self.queue)
            self.queue.clear()
            if cleared:
                self.dropped_frame_count += cleared
            return cleared

    def size(self) -> int:
        with self._lock:
            return len(self.queue)

    def oldest_packet_age_ms(self, *, now_ms: int | None = None) -> float | None:
        with self._lock:
            if not self.queue:
                return None
            now = int(self._now_ms() if now_ms is None else now_ms)
            oldest = self.queue[0]
            return max(0.0, float(now - int(oldest.captured_at_ms)))

    def stats(self) -> dict[str, float | int | None]:
        with self._lock:
            age = None
            if self.queue:
                now = int(self._now_ms())
                age = max(0.0, float(now - int(self.queue[0].captured_at_ms)))
            return {
                "camera_login_id": self.camera_login_id,
                "queue_size": len(self.queue),
                "queue_capacity": self.maxlen,
                "queue_overflow_drop_total": self.queue_overflow_drop_total,
                "oldest_packet_age_ms": age,
                "dropped_frame_count": self.dropped_frame_count,
                "aged_packet_drop_total": self.aged_packet_drop_total,
            }

    def _drop_aged_locked(self) -> None:
        if self.max_packet_age_ms is None:
            return
        now = int(self._now_ms())
        while self.queue:
            age = now - int(self.queue[0].captured_at_ms)
            if age <= self.max_packet_age_ms:
                break
            self.queue.popleft()
            self.dropped_frame_count += 1
            self.aged_packet_drop_total += 1


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    camera_login_id: str
    frame_id: int
    captured_at_ms: int
    width: int
    height: int
    read_index: int
    source_frame_index: int
    processed_at_ms: int | None = None
    published_at_ms: int | None = None

    @property
    def ai_latency_ms(self) -> int | None:
        if self.processed_at_ms is None:
            return None
        return max(0, self.processed_at_ms - self.captured_at_ms)

    @property
    def publish_latency_ms(self) -> int | None:
        if self.published_at_ms is None:
            return None
        return max(0, self.published_at_ms - self.captured_at_ms)

    @property
    def latency_order_valid(self) -> bool:
        return latency_order_valid(self.captured_at_ms, self.processed_at_ms, self.published_at_ms)


class FrameMetadataBuffer:
    def __init__(self, maxlen: int = 60, now_ms: Callable[[], int] | None = None):
        self.maxlen = max(1, int(maxlen))
        self._now_ms = now_ms or current_epoch_ms
        self._frames_by_camera: defaultdict[str, deque[FrameMetadata]] = defaultdict(lambda: deque(maxlen=self.maxlen))
        self._next_frame_id_by_camera: defaultdict[str, int] = defaultdict(int)

    def record_capture(self, camera_login_id: str, packet, frame_shape: Sequence[int]) -> FrameMetadata:
        frame_id = self._next_frame_id(camera_login_id)
        width, height = frame_size_from_shape(frame_shape)
        metadata = FrameMetadata(
            camera_login_id=str(camera_login_id),
            frame_id=frame_id,
            captured_at_ms=self._now_ms(),
            width=width,
            height=height,
            read_index=int(getattr(packet, "frame_idx", frame_id - 1)),
            source_frame_index=int(getattr(packet, "frame_idx", frame_id - 1)),
        )
        self._frames_by_camera[str(camera_login_id)].append(metadata)
        return metadata

    def mark_processed(self, camera_login_id: str, frame_id: int) -> FrameMetadata:
        return self._replace_metadata(camera_login_id, frame_id, processed_at_ms=self._now_ms())

    def mark_published(self, camera_login_id: str, frame_id: int) -> FrameMetadata:
        return self._replace_metadata(camera_login_id, frame_id, published_at_ms=self._now_ms())

    def get_latest(self, camera_login_id: str) -> FrameMetadata | None:
        frames = self._frames_by_camera.get(str(camera_login_id))
        if not frames:
            return None
        return frames[-1]

    def get_by_frame_id(self, camera_login_id: str, frame_id: int) -> FrameMetadata | None:
        for metadata in self._frames_by_camera.get(str(camera_login_id), ()):
            if metadata.frame_id == int(frame_id):
                return metadata
        return None

    def get_nearest_by_timestamp(self, camera_login_id: str, timestamp_ms: int) -> FrameMetadata | None:
        frames = self._frames_by_camera.get(str(camera_login_id))
        if not frames:
            return None
        return min(frames, key=lambda item: abs(item.captured_at_ms - int(timestamp_ms)))

    def evidence_context(
        self,
        camera_login_id: str,
        frame_id: int,
        dropped_frame_count: int = 0,
        snapshot_path: str | None = None,
        clip_path: str | None = None,
    ) -> dict[str, int | str | bool | None]:
        metadata = self.get_by_frame_id(camera_login_id, frame_id)
        if metadata is None:
            raise LookupError(f"frame metadata not found: camera_login_id={camera_login_id} frame_id={frame_id}")
        context: dict[str, int | str | bool | None] = {
            "cameraLoginId": metadata.camera_login_id,
            "frameId": int(metadata.frame_id),
            "timestampMs": int(metadata.captured_at_ms),
            "capturedAtMs": int(metadata.captured_at_ms),
            "processedAtMs": metadata.processed_at_ms,
            "publishedAtMs": metadata.published_at_ms,
            "aiLatencyMs": metadata.ai_latency_ms,
            "publishLatencyMs": metadata.publish_latency_ms,
            "droppedFrameCount": int(dropped_frame_count),
            "evidenceId": evidence_id(metadata.camera_login_id, metadata.frame_id, metadata.captured_at_ms),
            "traceId": evidence_id(metadata.camera_login_id, metadata.frame_id, metadata.captured_at_ms),
            "latencyOrderValid": metadata.latency_order_valid,
        }
        if snapshot_path is not None:
            context["snapshotPath"] = snapshot_path
        if clip_path is not None:
            context["clipPath"] = clip_path
        return context

    def size(self, camera_login_id: str) -> int:
        return len(self._frames_by_camera.get(str(camera_login_id), ()))

    def _next_frame_id(self, camera_login_id: str) -> int:
        key = str(camera_login_id)
        self._next_frame_id_by_camera[key] += 1
        return self._next_frame_id_by_camera[key]

    def _replace_metadata(self, camera_login_id: str, frame_id: int, **changes: int) -> FrameMetadata:
        key = str(camera_login_id)
        frames = self._frames_by_camera.get(key)
        if not frames:
            raise LookupError(f"frame metadata not found: camera_login_id={key} frame_id={frame_id}")
        for index, metadata in enumerate(frames):
            if metadata.frame_id == int(frame_id):
                updated = replace(metadata, **changes)
                frames[index] = updated
                return updated
        raise LookupError(f"frame metadata not found: camera_login_id={key} frame_id={frame_id}")


def current_epoch_ms() -> int:
    return time.time_ns() // 1_000_000


def evidence_context_from_packet(
    packet: FramePacket,
    processed_at_ms: int | None = None,
    published_at_ms: int | None = None,
    dropped_frame_count: int = 0,
    snapshot_path: str | None = None,
    clip_path: str | None = None,
) -> dict[str, int | str | bool | None]:
    context: dict[str, int | str | bool | None] = {
        "cameraLoginId": packet.camera_login_id,
        "frameId": int(packet.frame_id),
        "timestampMs": int(packet.captured_at_ms),
        "capturedAtMs": int(packet.captured_at_ms),
        "processedAtMs": processed_at_ms,
        "publishedAtMs": published_at_ms,
        "aiLatencyMs": max(0, int(processed_at_ms) - int(packet.captured_at_ms))
        if processed_at_ms is not None
        else None,
        "publishLatencyMs": max(0, int(published_at_ms) - int(packet.captured_at_ms))
        if published_at_ms is not None
        else None,
        "droppedFrameCount": int(dropped_frame_count),
        "evidenceId": evidence_id(packet.camera_login_id, packet.frame_id, packet.captured_at_ms),
        "traceId": evidence_id(packet.camera_login_id, packet.frame_id, packet.captured_at_ms),
        "latencyOrderValid": latency_order_valid(packet.captured_at_ms, processed_at_ms, published_at_ms),
    }
    if snapshot_path is not None:
        context["snapshotPath"] = snapshot_path
    if clip_path is not None:
        context["clipPath"] = clip_path
    return context


def frame_size_from_shape(frame_shape: Sequence[int]) -> tuple[int, int]:
    if len(frame_shape) < 2:
        return 0, 0
    return int(frame_shape[1]), int(frame_shape[0])
