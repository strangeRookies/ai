import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FramePacket:
    frame_idx: int
    fps: float
    timestamp: float
    frame: object


class VideoReader:
    def __init__(self, input_uri):
        self.input_uri = input_uri
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0.0
        self.duration_sec = 0.0
        self.frame_idx = 0

    def __enter__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(f"OpenCV is required to read video input: {exc}") from exc

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(self.input_uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video input: {self.input_uri}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0
        print(
            "[video-reader] opened: "
            f"input_uri={self.input_uri} "
            f"source_fps={self.fps:.6f} "
            f"total_frames={int(self.total_frames)} "
            f"duration_sec={self.duration_sec:.6f} "
            "frame_sampling=disabled",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.cap is not None:
            self.cap.release()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            print(f"[video-reader] frame read failed or stream ended: frame_idx={self.frame_idx}", file=sys.stderr)
            return None
        packet = FramePacket(
            frame_idx=self.frame_idx,
            fps=self.fps,
            timestamp=self.frame_idx / self.fps,
            frame=frame,
        )
        self.frame_idx += 1
        return packet
