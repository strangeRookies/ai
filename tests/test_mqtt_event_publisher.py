import os
import unittest

from ai.publishers.event_publisher import (
    MqttEventPublisher,
    _mqtt_connect_succeeded,
    _payload_context,
    _stamp_event_publish_attempt,
    mqtt_settings_from_env,
)


class MqttEventPublisherTest(unittest.TestCase):
    def test_connect_callback_accepts_paho_v2_reason_code(self):
        class ReasonCode:
            value = 0

            def __int__(self):
                raise TypeError("ReasonCode cannot be converted to int")

        self.assertTrue(_mqtt_connect_succeeded(ReasonCode()))
        self.assertFalse(_mqtt_connect_succeeded(type("FailureReasonCode", (), {"value": 5})()))

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
            "topic=camera, messageType=frame_sync, streamId=cam_05, cameraLoginId=cam_05, frameId=42, eventId=none, rc=0, connected=true, payloadBytes=93",
        )

    def test_event_payload_is_stamped_with_mqtt_publish_timing(self):
        payload = {
            "messageType": "event",
            "streamId": "cam_05",
            "cameraLoginId": "cam_05",
            "frameId": 42,
            "eventId": "evt-1",
            "capturedAtMs": 1000,
            "processedAtMs": 1200,
        }

        _stamp_event_publish_attempt(payload, 1500)
        context = _payload_context(payload, "event", connected=True, rc=0, publish_returned_at_ms=1503)

        self.assertEqual(payload["mqttPublishStartedAtMs"], 1500)
        self.assertEqual(payload["mqttPublishedAtMs"], 1500)
        self.assertIn("processedToMqttMs=300", context)
        self.assertIn("capturedToMqttMs=500", context)
        self.assertIn("mqttPublishCallMs=3", context)


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
