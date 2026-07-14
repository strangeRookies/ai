import tempfile
import threading
import unittest
from pathlib import Path

from ai.publishers.async_delivery import AsyncMqttDelivery
from ai.publishers.event_outbox import SqliteEventOutbox


class _FailingPublisher:
    connected = True

    def publish(self, payload, topic=None, qos=0):
        return False

    def close(self):
        pass


class _SuccessfulPublisher:
    connected = True

    def __init__(self):
        self.messages = []
        self.done = threading.Event()

    def publish(self, payload, topic=None, qos=0):
        self.messages.append((topic, qos, payload))
        self.done.set()
        return True

    def close(self):
        pass


class DurableOutboxTest(unittest.TestCase):
    def test_event_survives_failed_delivery_and_replays_after_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outbox_path = Path(temp_dir) / "events.sqlite3"
            first = AsyncMqttDelivery(_FailingPublisher(), outbox_path=outbox_path, retry_count=0)
            self.assertTrue(first.enqueue_event({"eventId": "evt-1"}, "event"))
            first.close()

            publisher = _SuccessfulPublisher()
            second = AsyncMqttDelivery(publisher, outbox_path=outbox_path, retry_count=0)
            self.assertTrue(publisher.done.wait(1.0))
            second.close()

            self.assertEqual(publisher.messages, [("event", 1, {"eventId": "evt-1"})])
            outbox = SqliteEventOutbox(outbox_path)
            self.assertIsNone(outbox.next_pending())
            outbox.close()


if __name__ == "__main__":
    unittest.main()
