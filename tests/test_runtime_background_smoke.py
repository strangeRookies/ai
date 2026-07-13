import threading
import unittest

from ai.publishers.async_delivery import AsyncMqttDelivery
import ai.snapshot_assist_upload as snapshot


class _Publisher:
    def __init__(self):
        self.connected = True
        self.messages = []
        self.done = threading.Event()

    def connect(self):
        self.connected = True
        return True

    def publish(self, payload, topic=None, qos=0):
        self.messages.append((topic, qos, payload))
        self.done.set()
        return True

    def close(self):
        pass


class _Frame:
    def copy(self):
        return self


class RuntimeBackgroundSmokeTest(unittest.TestCase):
    def test_mqtt_delivery_uses_background_thread_and_qos(self):
        publisher = _Publisher()
        delivery = AsyncMqttDelivery(publisher)
        self.assertTrue(delivery.enqueue_overlay({"kind": "overlay"}, "camera"))
        self.assertTrue(delivery.enqueue_event({"kind": "event"}, "event"))
        self.assertTrue(publisher.done.wait(1.0))
        delivery.close()
        self.assertIn(("event", 1, {"kind": "event"}), publisher.messages)

    def test_snapshot_encode_runs_in_worker_and_stops(self):
        encoded = threading.Event()
        main_thread = threading.get_ident()
        old_enabled = snapshot.snapshot_assist_enabled
        old_encode = snapshot.encode_frame_jpeg
        old_submit = snapshot.submit_snapshot_async
        try:
            snapshot.snapshot_assist_enabled = lambda: True
            snapshot.encode_frame_jpeg = lambda frame, quality=85: (encoded.set() or b"jpg")
            snapshot.submit_snapshot_async = lambda **kwargs: True
            self.assertTrue(snapshot.submit_frame_snapshot_async("event", "cam", _Frame()))
            self.assertTrue(encoded.wait(1.0))
            snapshot.stop_snapshot_assist_worker()
            self.assertFalse(snapshot._FRAME_WORKER_THREAD.is_alive())
        finally:
            snapshot.snapshot_assist_enabled = old_enabled
            snapshot.encode_frame_jpeg = old_encode
            snapshot.submit_snapshot_async = old_submit
            snapshot.stop_snapshot_assist_worker()
