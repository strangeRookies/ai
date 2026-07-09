import json
import threading
import time
import queue
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ai.visualization.action_overlay import initial_overlay_summary, make_placeholder


class OverlayState:
    def __init__(self):
        self.frame = None
        self.summary = initial_overlay_summary()
        self.connected = False
        self.last_error = ""
        self.lock = threading.Lock()
        self.event_queues = []
        self.mjpeg_client_count = 0
        self.mjpeg_frame_count = 0
        self.mjpeg_encode_latency_ms = 0.0
        self.mjpeg_encode_fail_count = 0
        self.last_mjpeg_sent_at = 0.0

    def register_event_queue(self, q):
        with self.lock:
            self.event_queues.append(q)

    def unregister_event_queue(self, q):
        with self.lock:
            if q in self.event_queues:
                self.event_queues.remove(q)

    def push_event(self, event_payload):
        with self.lock:
            for q in self.event_queues:
                try:
                    q.put_nowait(event_payload)
                except queue.Full:
                    pass

    def update_frame(self, frame, summary):
        with self.lock:
            self.frame = frame
            self.summary = dict(summary)
            self.connected = True
            self.last_error = ""

    def update_error(self, message):
        with self.lock:
            self.connected = False
            self.last_error = message

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def client_connected(self):
        with self.lock:
            self.mjpeg_client_count += 1

    def client_disconnected(self):
        with self.lock:
            self.mjpeg_client_count = max(0, self.mjpeg_client_count - 1)

    def record_mjpeg_frame(self, encode_latency_ms):
        with self.lock:
            self.mjpeg_frame_count += 1
            self.mjpeg_encode_latency_ms = float(encode_latency_ms)
            self.last_mjpeg_sent_at = time.time()

    def record_mjpeg_encode_failure(self):
        with self.lock:
            self.mjpeg_encode_fail_count += 1

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "last_error": self.last_error,
                "summary": {
                    **dict(self.summary),
                    "mjpeg_client_count": self.mjpeg_client_count,
                    "mjpeg_frame_count": self.mjpeg_frame_count,
                    "mjpeg_encode_latency_ms": self.mjpeg_encode_latency_ms,
                    "mjpeg_encode_fail_count": self.mjpeg_encode_fail_count,
                    "last_mjpeg_sent_at": self.last_mjpeg_sent_at,
                },
            }


class OverlayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = None

    def do_OPTIONS(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_cors_headers()
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            status_data = self.state.status()
            summary_data = status_data["summary"]
            
            frame = self.state.snapshot()
            src_res = "unknown"
            if frame is not None:
                src_res = f"{frame.shape[1]}x{frame.shape[0]}"
            
            width = int(getattr(self.server, "mjpeg_width", 0) or 0)
            height = int(getattr(self.server, "mjpeg_height", 0) or 0)
            out_res = f"{width}x{height}" if (width > 0 and height > 0) else src_res
            
            latest_cap = summary_data.get("latest_captured_at_ms", 0)
            frame_age_ms = int(time.time() * 1000 - latest_cap) if latest_cap else -1
            
            health_payload = {
                "status": "ok",
                "source_resolution": src_res,
                "output_resolution": out_res,
                "jpeg_quality": int(getattr(self.server, "jpeg_quality", 80)),
                "fps": float(getattr(self.server, "target_fps", 8.0)),
                "frame_age_ms": frame_age_ms,
                "active_tracks": summary_data.get("active_tracks", 0),
                "mjpeg_frame_count": summary_data.get("mjpeg_frame_count", 0),
                "stream_clients": summary_data.get("mjpeg_client_count", 0),
                "processed_frame_count": summary_data.get("frames_processed", 0),
                **status_data
            }
            self.send_json(health_payload)
            return
        if path == "/cameras":
            camera_id = getattr(self.server, "camera_id", "camera-1")
            self.send_json({"cameras": [{"id": camera_id, "connected": self.state.status()["connected"]}]})
            return
        if path == "/summary":
            self.send_json(self.state.status()["summary"])
            return
        if path == "/events":
            self.stream_events()
            return
        camera_id = getattr(self.server, "camera_id", "camera-1")
        base_path = getattr(self.server, "base_path", "/mjpeg")
        if path == "/" or path.startswith("/stream") or path == f"{base_path}/{camera_id}":
            self.stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt, *args):
        print(f"[ai-overlay-http] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_cors_headers()
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def stream_events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue(maxsize=100)
        self.state.register_event_queue(q)
        try:
            while True:
                try:
                    event = q.get(timeout=15.0)
                    data = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.state.unregister_event_queue(q)

    def stream(self):
        import cv2

        self.state.client_connected()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Connection", "keep-alive")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        delay = 1.0 / max(1.0, float(self.server.target_fps))
        try:
            while True:
                frame = self.state.snapshot()
                if frame is None:
                    status = self.state.status()
                    frame = make_placeholder(status["last_error"] or "waiting for RTSP frames")
                width = int(getattr(self.server, "mjpeg_width", 0) or 0)
                height = int(getattr(self.server, "mjpeg_height", 0) or 0)
                if width > 0 and height > 0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                started_at = time.perf_counter()
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.server.jpeg_quality)],
                )
                if not ok:
                    self.state.record_mjpeg_encode_failure()
                    time.sleep(delay)
                    continue
                self.state.record_mjpeg_frame((time.perf_counter() - started_at) * 1000.0)
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, GeneratorExit):
                    return
                time.sleep(delay)
        finally:
            self.state.client_disconnected()


def create_overlay_server(
    host,
    port,
    state,
    camera_id,
    target_fps,
    base_path="/mjpeg",
    jpeg_quality=70,
    width=640,
    height=360,
):
    OverlayHandler.state = state
    server = ThreadingHTTPServer((host, port), OverlayHandler)
    server.target_fps = target_fps
    server.camera_id = camera_id
    server.base_path = base_path.rstrip("/") or "/mjpeg"
    server.jpeg_quality = max(1, min(100, int(jpeg_quality)))
    server.mjpeg_width = max(0, int(width))
    server.mjpeg_height = max(0, int(height))
    return server
