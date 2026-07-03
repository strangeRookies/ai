import os
import unittest

from ai.publishers.event_publisher import MqttEventPublisher, _payload_context, mqtt_settings_from_env


class MqttEventPublisherTest(unittest.TestCase):
    def test_mqtt_settings_from_env_uses_local_defaults(self):
        original_values = {name: os.environ.get(name) for name in mqtt_env_names()}
        try:
            for name in mqtt_env_names():
                os.environ.pop(name, None)

            settings = mqtt_settings_from_env()
        finally:
            restore_env(original_values)

        self.assertEqual(settings["host"], "localhost")
        self.assertEqual(settings["port"], 1883)
        self.assertEqual(settings["topic"], "event")
        self.assertEqual(settings["camera_topic"], "camera")
        self.assertEqual(settings["event_topic"], "event")
        self.assertEqual(settings["client_id"], "strange-ai-local")
        self.assertIsNone(settings["username"])
        self.assertIsNone(settings["password"])

    def test_mqtt_settings_from_env_reads_aws_ready_overrides(self):
        original_values = {name: os.environ.get(name) for name in mqtt_env_names()}
        try:
            os.environ["MQTT_HOST"] = "mqtt.aws.internal"
            os.environ["MQTT_PORT"] = "8883"
            os.environ["MQTT_TOPIC"] = "prod/safety/events"
            os.environ["MQTT_CAMERA_TOPIC"] = "camera"
            os.environ["MQTT_EVENT_TOPIC"] = "event"
            os.environ["MQTT_CLIENT_ID"] = "strange-ai-prod"
            os.environ["MQTT_USERNAME"] = "edge-user"
            os.environ["MQTT_PASSWORD"] = "secret-value"

            settings = mqtt_settings_from_env()
        finally:
            restore_env(original_values)

        self.assertEqual(settings["host"], "mqtt.aws.internal")
        self.assertEqual(settings["port"], 8883)
        self.assertEqual(settings["topic"], "prod/safety/events")
        self.assertEqual(settings["camera_topic"], "camera")
        self.assertEqual(settings["event_topic"], "event")
        self.assertEqual(settings["client_id"], "strange-ai-prod")
        self.assertEqual(settings["username"], "edge-user")
        self.assertEqual(settings["password"], "secret-value")

    def test_mqtt_connection_failure_does_not_raise(self):
        publisher = MqttEventPublisher(
            host="127.0.0.1",
            port=1,
            topic="safety/events",
            client_id="strange-ai-test",
        )

        connected = publisher.connect()
        published = publisher.publish({"event_type": "Faint"})

        self.assertFalse(connected)
        self.assertFalse(published)

    def test_payload_context_includes_publish_diagnostics(self):
        context = _payload_context(
            {
                "messageType": "frame_sync",
                "streamId": "cam_05",
                "cameraLoginId": "cam_05",
                "frameId": 42,
            },
            "camera",
            connected=True,
            rc=0,
        )

        self.assertEqual(
            context,
            "topic=camera, messageType=frame_sync, streamId=cam_05, cameraLoginId=cam_05, frameId=42, rc=0, connected=true",
        )


def mqtt_env_names():
    return (
        "MQTT_HOST",
        "MQTT_PORT",
        "MQTT_TOPIC",
        "MQTT_CAMERA_TOPIC",
        "MQTT_EVENT_TOPIC",
        "MQTT_CLIENT_ID",
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
    )


def restore_env(values):
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
