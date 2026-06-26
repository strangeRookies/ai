import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from stream.frame_queue import LatestFrameQueue


def redact_url(url):
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    if not parsed.password:
        return url
    username = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{username}:***@{host}{port}" if username else f"***@{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class RtspFrameReader:
    def __init__(self, rtsp_url, queue_size=2, reconnect_delay_seconds=3):
        self.rtsp_url = rtsp_url
        self.frames = LatestFrameQueue(queue_size)
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._stop_event = threading.Event()
        self._thread = None
        self.frames_read = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="rtsp-frame-reader", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def read_latest(self, timeout=1):
        return self.frames.get(timeout=timeout)

    @property
    def drop_count(self):
        return self.frames.drop_count

    def _run(self):
        try:
            import cv2
            import os
        except ImportError as exc:
            print(f"[rtsp-reader] OpenCV import failed: {exc}", file=sys.stderr)
            return

        # Enable TCP transport and 5-second socket timeout to prevent indefinite blocking
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(
                    f"[rtsp-reader] RTSP connection failed: url={redact_url(self.rtsp_url)}. "
                    f"Retrying in {self.reconnect_delay_seconds}s",
                    file=sys.stderr,
                )
                cap.release()
                time.sleep(self.reconnect_delay_seconds)
                continue

            print(f"[rtsp-reader] Connected to RTSP stream: {redact_url(self.rtsp_url)}")
            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        print(
                            f"[rtsp-reader] Failed to read RTSP frame: url={redact_url(self.rtsp_url)}. Reconnecting.",
                            file=sys.stderr,
                        )
                        break
                    self.frames_read += 1
                    self.frames.put_latest(frame)
            finally:
                cap.release()

            time.sleep(self.reconnect_delay_seconds)
