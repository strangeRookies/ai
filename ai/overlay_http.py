import json
import threading
import time
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

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "last_error": self.last_error,
                "summary": dict(self.summary),
            }


class OverlayHandler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json({"status": "ok", **self.state.status()})
            return
        if path == "/cameras":
            camera_id = getattr(self.server, "camera_id", "camera-1")
            self.send_json({"cameras": [{"id": camera_id, "connected": self.state.status()["connected"]}]})
            return
        if path == "/summary":
            self.send_json(self.state.status()["summary"])
            return
        if path == "/" or path.startswith("/stream"):
            self.stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt, *args):
        print(f"[ai-overlay-http] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        import cv2

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        delay = 1.0 / max(1.0, float(self.server.target_fps))
        while True:
            frame = self.state.snapshot()
            if frame is None:
                status = self.state.status()
                frame = make_placeholder(status["last_error"] or "waiting for RTSP frames")
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                time.sleep(delay)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(delay)


def create_overlay_server(host, port, state, camera_id, target_fps):
    OverlayHandler.state = state
    server = ThreadingHTTPServer((host, port), OverlayHandler)
    server.target_fps = target_fps
    server.camera_id = camera_id
    return server
