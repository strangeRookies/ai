import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.publishers.event_publisher import MqttEventPublisher, mqtt_settings_from_env


def mqtt_publisher_kwargs(settings):
    return {
        "host": settings["host"],
        "port": settings["port"],
        "topic": settings["topic"],
        "client_id": settings["client_id"],
        "username": settings["username"],
        "password": settings["password"],
    }


def build_test_event():
    return {
        "schemaVersion": "1.0",
        "camera_id": "cam_01",
        "camera_login_id": "cam_01",
        "timestamp": 1710000000.0,
        "event_type": "Faint",
        "severity": "HIGH",
        "message": "MQTT test event",
        "source": "edge-ai-test",
        "confidence": 0.91,
        "bbox": [100.0, 80.0, 220.0, 300.0],
        "track_id": 1,
    }


def main():
    try:
        settings = mqtt_settings_from_env()
    except ValueError as exc:
        print(f"[mqtt-test] configuration error: {exc}", file=sys.stderr)
        return 1

    publisher = MqttEventPublisher(**mqtt_publisher_kwargs(settings))
    connected = publisher.connect()
    if not connected:
        return 1
    try:
        event = build_test_event()
        published = publisher.publish(event)
        if not published:
            return 1
        print(f"[mqtt-test] published to {settings['topic']}: {json.dumps(event, ensure_ascii=False)}")
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
