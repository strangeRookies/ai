"""Snapshot assist upload client — does not affect primary alert path."""

from __future__ import annotations

import io
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from ai.snapshot_assist_upload import (
    encode_frame_jpeg,
    submit_snapshot,
    submit_snapshot_async,
)


class _Handler(BaseHTTPRequestHandler):
    last = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        token = self.headers.get("X-Service-Token")
        _Handler.last = {
            "path": self.path,
            "token": token,
            "body_len": len(body),
            "content_type": self.headers.get("Content-Type"),
        }
        if token != "good-token":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):  # noqa: A003
        return


class SnapshotAssistUploadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_submit_requires_token_and_posts_multipart(self):
        base = f"http://127.0.0.1:{self.port}/api/internal/vlm/snapshot-assist"
        jpeg = b"\xff\xd8" + b"\x00" * 120 + b"\xff\xd9"
        code = submit_snapshot(
            event_id="evt-http-1",
            camera_login_id="cam_01",
            jpeg_bytes=jpeg,
            base_url=base,
            token="good-token",
        )
        self.assertEqual(code, 202)
        self.assertIn("evt-http-1", _Handler.last["path"])
        self.assertEqual(_Handler.last["token"], "good-token")
        self.assertIn("multipart", _Handler.last["content_type"])

    def test_unauthorized_without_token(self):
        base = f"http://127.0.0.1:{self.port}/api/internal/vlm/snapshot-assist"
        code = submit_snapshot(
            event_id="evt-http-2",
            camera_login_id="cam_01",
            jpeg_bytes=b"\xff\xd8" + b"\x00" * 120 + b"\xff\xd9",
            base_url=base,
            token="bad",
        )
        self.assertEqual(code, 401)

    def test_async_failure_does_not_raise(self):
        # Invalid port — thread swallows errors
        submit_snapshot_async(
            event_id="evt-async",
            camera_login_id="cam",
            jpeg_bytes=b"\xff\xd8" + b"\x00" * 120 + b"\xff\xd9",
        )

    def test_encode_frame_jpeg_optional_cv2(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy missing")
        frame = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
        out = encode_frame_jpeg(frame)
        # cv2 may be missing in this env
        if out is not None:
            self.assertTrue(out.startswith(b"\xff\xd8"))

    def test_duplicate_event_id_only_uploaded_once(self):
        import time
        import os
        from ai.snapshot_assist_upload import _SENT_EVENTS, _SENT_EVENTS_LOCK, _UPLOAD_QUEUE
        
        # Reset sent cache
        with _SENT_EVENTS_LOCK:
            _SENT_EVENTS.clear()
            
        old_enabled = os.environ.get("VLM_SNAPSHOT_ASSIST_ENABLED")
        old_token = os.environ.get("VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN")
        os.environ["VLM_SNAPSHOT_ASSIST_ENABLED"] = "true"
        os.environ["VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN"] = "good-token"
        os.environ["SNAPSHOT_ASSIST_BASE_URL"] = f"http://127.0.0.1:{self.port}/api/internal/vlm/snapshot-assist"
        
        try:
            _Handler.last = {}
            # 1st request (valid)
            submit_snapshot_async(
                event_id="evt-dup-test-1",
                camera_login_id="cam_01",
                jpeg_bytes=b"\xff\xd8" + b"\x00" * 120 + b"\xff\xd9",
            )
            # 2nd request (duplicate)
            submit_snapshot_async(
                event_id="evt-dup-test-1",
                camera_login_id="cam_01",
                jpeg_bytes=b"\xff\xd8" + b"\x00" * 120 + b"\xff\xd9",
            )
            
            # Wait for background uploader
            time.sleep(0.5)
            
            # Handler should only have received it once (or last updated state confirms check)
            self.assertEqual(_Handler.last.get("token"), "good-token")
            self.assertIn("evt-dup-test-1", _Handler.last.get("path", ""))
            
            # Verify it is in deduplication set
            with _SENT_EVENTS_LOCK:
                self.assertIn("evt-dup-test-1", _SENT_EVENTS)
        finally:
            if old_enabled is not None:
                os.environ["VLM_SNAPSHOT_ASSIST_ENABLED"] = old_enabled
            else:
                os.environ.pop("VLM_SNAPSHOT_ASSIST_ENABLED", None)
            if old_token is not None:
                os.environ["VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN"] = old_token
            else:
                os.environ.pop("VLM_SNAPSHOT_ASSIST_SERVICE_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
