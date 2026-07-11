"""Real keyframe extraction from clip bytes (OpenCV). Offline-testable helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MAX_FRAMES = 6
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True, slots=True)
class ExtractedKeyframe:
    index: int
    timestamp_sec: float
    frame_index: int
    jpeg_bytes: bytes


def select_frame_indices(
    total_frames: int,
    *,
    max_frames: int = MAX_FRAMES,
    event_frame_index: int | None = None,
) -> list[int]:
    """Deterministic sample indices in [0, total_frames).

    If event_frame_index is known, bias samples around it; else uniform.
    """
    if total_frames <= 0:
        return []
    n = min(max(1, int(max_frames)), total_frames)
    if n == 1:
        if event_frame_index is None:
            return [0]
        return [max(0, min(total_frames - 1, int(event_frame_index)))]

    if event_frame_index is None:
        # Uniform inclusive endpoints: 0 .. last
        if n == total_frames:
            return list(range(total_frames))
        # Deterministic linspace without numpy
        return sorted({int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)})

    center = max(0, min(total_frames - 1, int(event_frame_index)))
    # Spread half before/after center
    half = n // 2
    start = center - half
    end = start + n - 1
    if start < 0:
        end -= start
        start = 0
    if end >= total_frames:
        shift = end - (total_frames - 1)
        start = max(0, start - shift)
        end = total_frames - 1
    # Ensure exactly n indices by uniform fill in window
    window = end - start
    if window <= 0:
        return [center]
    indices = sorted({int(round(start + i * window / (n - 1))) for i in range(n)})
    # Pad if de-dupe shrank set
    cursor = start
    while len(indices) < n and cursor <= end:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return sorted(indices)[:n]


def parse_event_offset_sec(metadata: dict[str, str]) -> float | None:
    for key in (
        "event_offset_sec",
        "eventOffsetSec",
        "event_timestamp_sec",
        "eventTimestampSec",
        "offset_sec",
    ):
        raw = metadata.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def extract_keyframes_from_path(
    video_path: str | Path,
    *,
    max_frames: int = MAX_FRAMES,
    event_offset_sec: float | None = None,
    jpeg_quality: int = 85,
) -> list[ExtractedKeyframe]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"OpenCV required for keyframe extraction: {exc}") from exc

    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            # Some containers report 0; fall back to sequential read count
            frames_buf: list = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_buf.append(frame)
            total = len(frames_buf)
            event_idx = None
            if event_offset_sec is not None and fps > 0:
                event_idx = int(round(event_offset_sec * fps))
            indices = select_frame_indices(total, max_frames=max_frames, event_frame_index=event_idx)
            out: list[ExtractedKeyframe] = []
            for out_i, fi in enumerate(indices):
                frame = frames_buf[fi]
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if not ok:
                    raise RuntimeError(f"JPEG encode failed at frame {fi}")
                jpeg = buf.tobytes()
                if not jpeg.startswith(JPEG_MAGIC):
                    raise RuntimeError("encoded bytes are not JPEG")
                ts = (fi / fps) if fps > 0 else float(fi)
                out.append(
                    ExtractedKeyframe(
                        index=out_i,
                        timestamp_sec=ts,
                        frame_index=fi,
                        jpeg_bytes=jpeg,
                    )
                )
            return out

        event_idx = None
        if event_offset_sec is not None and fps > 0:
            event_idx = int(round(event_offset_sec * fps))
        indices = select_frame_indices(total, max_frames=max_frames, event_frame_index=event_idx)
        out = []
        for out_i, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame index {fi}")
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not ok:
                raise RuntimeError(f"JPEG encode failed at frame {fi}")
            jpeg = buf.tobytes()
            if not jpeg.startswith(JPEG_MAGIC):
                raise RuntimeError("encoded bytes are not JPEG")
            ts = (fi / fps) if fps > 0 else float(fi)
            out.append(
                ExtractedKeyframe(
                    index=out_i,
                    timestamp_sec=ts,
                    frame_index=fi,
                    jpeg_bytes=jpeg,
                )
            )
        return out
    finally:
        cap.release()


def extract_keyframes_from_bytes(
    clip_bytes: bytes,
    *,
    max_frames: int = MAX_FRAMES,
    event_offset_sec: float | None = None,
    jpeg_quality: int = 85,
    suffix: str = ".mp4",
) -> list[ExtractedKeyframe]:
    if not clip_bytes:
        raise RuntimeError("empty clip bytes")
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="vlm_clip_", suffix=suffix)
        os.close(fd)
        Path(tmp_path).write_bytes(clip_bytes)
        return extract_keyframes_from_path(
            tmp_path,
            max_frames=max_frames,
            event_offset_sec=event_offset_sec,
            jpeg_quality=jpeg_quality,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def resolve_fixture_mp4(metadata: dict[str, str] | None = None) -> Path | None:
    """Mock offline path: env VLM_FIXTURE_MP4 or fixtures/vlm/sample.mp4."""
    env_path = os.getenv("VLM_FIXTURE_MP4", "").strip()
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    if metadata:
        meta_path = metadata.get("fixture_mp4") or metadata.get("fixtureMp4")
        if meta_path and Path(meta_path).is_file():
            return Path(meta_path)
    # default relative to repo ai/
    root = Path(__file__).resolve().parents[2]
    candidate = root / "fixtures" / "vlm" / "sample.mp4"
    if candidate.is_file():
        return candidate
    return None
