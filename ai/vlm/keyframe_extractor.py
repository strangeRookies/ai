"""Deterministic in-memory keyframe extraction for finalized incident clips."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


KEYFRAME_COUNT = 8


class KeyframeExtractionError(ValueError):
    """A safe-to-report clip acquisition or extraction failure."""


@dataclass(frozen=True, slots=True)
class ExtractedKeyframe:
    index: int
    timestamp_sec: float
    frame_index: int
    width: int
    height: int
    sha256: str
    jpeg_bytes: bytes

    def metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "timestamp_sec": self.timestamp_sec,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }


@contextmanager
def local_video_source(input_url: str, *, timeout_sec: float = 30.0) -> Iterator[Path]:
    """Resolve a local/file/HTTP input and always remove downloaded temporary data."""

    parsed = urlparse(input_url)
    temporary_path: Path | None = None
    try:
        if parsed.scheme in {"http", "https"}:
            suffix = Path(unquote(parsed.path)).suffix or ".video"
            try:
                request = Request(input_url, headers={"User-Agent": "strange-ai-vlm/1"})
                with urlopen(request, timeout=timeout_sec) as response:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as target:
                        temporary_path = Path(target.name)
                        while chunk := response.read(1024 * 1024):
                            target.write(chunk)
            except Exception as exc:
                raise KeyframeExtractionError("failed to download input video") from exc
            yield temporary_path
            return

        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise KeyframeExtractionError("unsupported file URL host")
            decoded_path = unquote(parsed.path)
            if os.name == "nt" and decoded_path.startswith("/") and len(decoded_path) >= 3 and decoded_path[2] == ":":
                decoded_path = decoded_path[1:]
            path = Path(decoded_path)
        elif parsed.scheme == "" or (
            os.name == "nt"
            and len(parsed.scheme) == 1
            and len(input_url) >= 2
            and input_url[1] == ":"
        ):
            path = Path(input_url)
        else:
            raise KeyframeExtractionError("unsupported input URL scheme")

        if not path.is_file():
            raise KeyframeExtractionError("input video is not a readable file")
        yield path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def extract_eight_keyframes(
    video_path: str | os.PathLike[str],
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[ExtractedKeyframe, ...]:
    """Extract frames nearest the centers of eight equal clip segments.

    The selected source frame indices are deterministic and ordered. Only the eight
    selected decoded frames and their JPEG payloads are retained in memory.
    """

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared by the project
        raise KeyframeExtractionError("OpenCV is required for keyframe extraction") from exc

    if not math.isfinite(start_sec) or start_sec < 0:
        raise KeyframeExtractionError("invalid clip range")
    if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
        raise KeyframeExtractionError("invalid clip range")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise KeyframeExtractionError("failed to decode input video")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise KeyframeExtractionError("input video has invalid FPS")
        if not math.isfinite(frame_count_value) or frame_count_value <= 0:
            raise KeyframeExtractionError("input video has invalid duration")
        frame_count = int(frame_count_value)
        duration_sec = frame_count / fps
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise KeyframeExtractionError("input video has invalid duration")
        if start_sec >= duration_sec or (end_sec is not None and end_sec > duration_sec):
            raise KeyframeExtractionError("clip range is outside the input video")

        first_frame = int(math.ceil(start_sec * fps))
        end_exclusive = frame_count if end_sec is None else min(frame_count, int(math.ceil(end_sec * fps)))
        available_frames = end_exclusive - first_frame
        if available_frames < KEYFRAME_COUNT:
            raise KeyframeExtractionError("input video is too short for eight unique keyframes")

        target_indices = tuple(
            first_frame + ((2 * index + 1) * available_frames) // (2 * KEYFRAME_COUNT)
            for index in range(KEYFRAME_COUNT)
        )
        if len(set(target_indices)) != KEYFRAME_COUNT:
            raise KeyframeExtractionError("input video is too short for eight unique keyframes")

        frames: list[ExtractedKeyframe] = []
        target_position = 0
        source_index = 0
        while target_position < KEYFRAME_COUNT:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise KeyframeExtractionError("failed to decode requested keyframe")
            target_index = target_indices[target_position]
            if source_index == target_index:
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0:
                    raise KeyframeExtractionError("decoded keyframe has invalid dimensions")
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                if not encoded or jpeg is None or jpeg.size == 0:
                    raise KeyframeExtractionError("failed to encode keyframe as JPEG")
                payload = jpeg.tobytes()
                frames.append(
                    ExtractedKeyframe(
                        index=target_position,
                        timestamp_sec=target_index / fps,
                        frame_index=target_index,
                        width=int(width),
                        height=int(height),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        jpeg_bytes=payload,
                    )
                )
                target_position += 1
            source_index += 1

        if len({frame.sha256 for frame in frames}) != KEYFRAME_COUNT:
            raise KeyframeExtractionError("input video produced duplicate keyframes")
        return tuple(frames)
    finally:
        capture.release()
