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
                chunk = response.read(256)

            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=frame")
            self.assertIn(b"--frame", chunk)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
