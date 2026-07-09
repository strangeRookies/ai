import http.client
import threading
import unittest
from argparse import Namespace
from contextlib import closing

import numpy as np

from ai.overlay_http import OverlayState, create_overlay_server
from scripts.serve_ai_overlay import mjpeg_debug_enabled


class MjpegOverlayHttpTest(unittest.TestCase):
    def test_mjpeg_enabled_accepts_demo_env_flag(self):
        self.assertTrue(mjpeg_debug_enabled(Namespace(mjpeg_debug=False, mjpeg_enabled=True)))

    def test_mjpeg_camera_path_streams_multipart_jpeg(self):
        state = OverlayState()
        state.update_frame(np.zeros((8, 8, 3), dtype=np.uint8), {"frames_processed": 1})
        server = create_overlay_server(
            "127.0.0.1",
            0,
            state,
            "cam_01",
            30.0,
            base_path="/mjpeg",
            jpeg_quality=65,
            width=8,
            height=8,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            with closing(http.client.HTTPConnection(host, port, timeout=2.0)) as connection:
                connection.request("GET", "/mjpeg/cam_01")
                response = connection.getresponse()
                chunk = response.read(4096)

            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=frame")
            self.assertEqual(response.getheader("Connection"), "keep-alive")
            self.assertIn(b"--frame", chunk)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            # Count the occurrences of b"--frame" in the chunk to verify multiple frames are streamed
            self.assertGreaterEqual(chunk.count(b"--frame"), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_health_response_allows_browser_fetch_with_cors_headers(self):
        state = OverlayState()
        state.update_frame(np.zeros((8, 8, 3), dtype=np.uint8), {"frames_processed": 7, "active_tracks": 2})
        server = create_overlay_server(
            "127.0.0.1",
            0,
            state,
            "cam_05",
            8.0,
            base_path="/mjpeg",
            jpeg_quality=80,
            width=1280,
            height=720,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            with closing(http.client.HTTPConnection(host, port, timeout=2.0)) as connection:
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read()

            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "*")
            self.assertEqual(response.getheader("Access-Control-Allow-Methods"), "GET, OPTIONS")
            self.assertEqual(response.getheader("Access-Control-Allow-Headers"), "Content-Type")
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertIn(b'"frame_age_ms"', body)
            self.assertIn(b'"processed_frame_count": 7', body)
            self.assertIn(b'"mjpeg_frame_count"', body)
            self.assertIn(b'"active_tracks": 2', body)
            self.assertIn(b'"output_resolution": "1280x720"', body)
            self.assertIn(b'"fps": 8.0', body)
            self.assertIn(b'"jpeg_quality": 80', body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_health_options_preflight_returns_cors_headers_without_body(self):
        state = OverlayState()
        server = create_overlay_server(
            "127.0.0.1",
            0,
            state,
            "cam_05",
            8.0,
            base_path="/mjpeg",
            jpeg_quality=80,
            width=1280,
            height=720,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            with closing(http.client.HTTPConnection(host, port, timeout=2.0)) as connection:
                connection.request("OPTIONS", "/health")
                response = connection.getresponse()
                body = response.read()

            self.assertIn(response.status, {200, 204})
            self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "*")
            self.assertEqual(response.getheader("Access-Control-Allow-Methods"), "GET, OPTIONS")
            self.assertEqual(response.getheader("Access-Control-Allow-Headers"), "Content-Type")
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(body, b"")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
