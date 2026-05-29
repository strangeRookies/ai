import json
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from stream.rtsp_reader import redact_url


DEFAULT_CAMERAS = {
    "camera-1": {
        "name": "CCTV-01",
        "location": "1F Room 1",
        "rtsp_url": "rtsp://localhost:8554/cam1",
    },
    "camera-2": {
        "name": "CCTV-02",
        "location": "1F Corridor A",
        "rtsp_url": "rtsp://localhost:8554/cam2",
    },
    "camera-3": {
        "name": "CCTV-03",
        "location": "Remote Webcam 1",
        "rtsp_url": "rtsp://192.168.0.10:8554/cam3",
    },
    "camera-4": {
        "name": "CCTV-04",
        "location": "Remote Webcam 2",
        "rtsp_url": "rtsp://192.168.0.11:8554/cam4",
    },
}


def load_cameras():
    cameras = {}
    for camera_id, defaults in DEFAULT_CAMERAS.items():
        env_prefix = camera_id.replace("-", "_").upper()
        rtsp_url = os.getenv(f"{env_prefix}_RTSP_URL", defaults["rtsp_url"])
        cameras[camera_id] = {
            "id": camera_id,
            "name": os.getenv(f"{env_prefix}_NAME", defaults["name"]),
            "location": os.getenv(f"{env_prefix}_LOCATION", defaults["location"]),
            "rtsp_url": rtsp_url,
        }
    return cameras


class CameraWorker:
    def __init__(self, camera_id, rtsp_url, reconnect_delay=3.0):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.frame = None
        self.connected = False
        self.last_error = ""
        self.updated_at = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"mjpeg-{camera_id}", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def snapshot(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "last_error": self.last_error,
                "updated_at": self.updated_at,
            }

    def _set_state(self, frame=None, connected=None, last_error=None):
        with self.lock:
            if frame is not None:
                self.frame = frame
                self.updated_at = time.time()
            if connected is not None:
                self.connected = connected
            if last_error is not None:
                self.last_error = last_error

    def _run(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                message = f"RTSP connection failed: {redact_url(self.rtsp_url)}"
                print(f"[mjpeg] {self.camera_id} {message}", flush=True)
                self._set_state(connected=False, last_error=message)
                cap.release()
                time.sleep(self.reconnect_delay)
                continue

            print(f"[mjpeg] {self.camera_id} connected: {redact_url(self.rtsp_url)}", flush=True)
            self._set_state(connected=True, last_error="")
            try:
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._set_state(connected=False, last_error="failed to read frame")
                        break
                    self._set_state(frame=frame, connected=True, last_error="")
            finally:
                cap.release()

            time.sleep(self.reconnect_delay)


def make_placeholder(camera_id, message):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (12, 18, 32)
    cv2.putText(image, camera_id, (32, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (226, 232, 240), 2, cv2.LINE_AA)
    cv2.putText(image, message, (32, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2, cv2.LINE_AA)
    return image


class StreamHandler(BaseHTTPRequestHandler):
    workers = {}
    cameras = {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json({"status": "ok", "cameras": self.public_camera_status()})
            return
        if path == "/cameras":
            self.send_json({"cameras": self.public_camera_status()})
            return
        if path.startswith("/stream/"):
            camera_id = path.removeprefix("/stream/").strip("/")
            self.stream_camera(camera_id)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt, *args):
        print(f"[mjpeg-http] {self.address_string()} {fmt % args}", flush=True)

    def public_camera_status(self):
        result = []
        for camera_id, camera in self.cameras.items():
            state = self.workers[camera_id].status()
            result.append(
                {
                    "id": camera_id,
                    "name": camera["name"],
                    "location": camera["location"],
                    "streamUrl": f"/stream/{camera_id}",
                    "connected": state["connected"],
                    "updated_at": state["updated_at"],
                    "last_error": state["last_error"],
                }
            )
        return result

    def send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", os.getenv("STREAM_CORS_ORIGIN", "*"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_camera(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown camera")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", os.getenv("STREAM_CORS_ORIGIN", "*"))
        self.end_headers()

        target_fps = max(1.0, float(os.getenv("MJPEG_FPS", "8")))
        delay = 1.0 / target_fps
        while True:
            frame = worker.snapshot()
            if frame is None:
                state = worker.status()
                frame = make_placeholder(camera_id, "OFFLINE" if not state["connected"] else "WAITING FOR FRAME")
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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


def main():
    host = os.getenv("MJPEG_HOST", "0.0.0.0")
    port = int(os.getenv("MJPEG_PORT", "8000"))
    reconnect_delay = float(os.getenv("RTSP_RECONNECT_DELAY_SECONDS", "3"))
    cameras = load_cameras()
    workers = {
        camera_id: CameraWorker(camera_id, camera["rtsp_url"], reconnect_delay=reconnect_delay)
        for camera_id, camera in cameras.items()
    }
    StreamHandler.cameras = cameras
    StreamHandler.workers = workers

    for worker in workers.values():
        worker.start()

    server = ThreadingHTTPServer((host, port), StreamHandler)
    print(f"[mjpeg] serving http://{host}:{port}", flush=True)
    for camera_id, camera in cameras.items():
        print(f"[mjpeg] {camera_id} {camera['name']} url={redact_url(camera['rtsp_url'])}", flush=True)

    def shutdown(_signum, _frame):
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    finally:
        for worker in workers.values():
            worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
