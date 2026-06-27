import unittest

from scripts.publish_test_mqtt_event import build_test_event, mqtt_publisher_kwargs


class PublishTestMqttEventTest(unittest.TestCase):
    def test_publisher_kwargs_excludes_topic_split_settings(self):
        settings = {
            "host": "127.0.0.1",
            "port": 1883,
            "topic": "safety/events",
            "camera_topic": "camera",
            "event_topic": "event",
            "client_id": "strange-ai-local",
            "username": None,
            "password": None,
        }

        self.assertEqual(
            mqtt_publisher_kwargs(settings),
            {
                "host": "127.0.0.1",
                "port": 1883,
                "topic": "safety/events",
                "client_id": "strange-ai-local",
                "username": None,
                "password": None,
            },
        )

    def test_test_event_contains_backend_camera_alias(self):
        event = build_test_event()

        self.assertEqual(event["camera_id"], "cam_01")
        self.assertEqual(event["camera_login_id"], "cam_01")
        self.assertEqual(event["event_type"], "Faint")
        self.assertEqual(event["source"], "edge-ai-test")


if __name__ == "__main__":
    unittest.main()
