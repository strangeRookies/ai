import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from messaging.event_schema import build_safety_event


EVENT_TYPES = [
    "fall_detected",
    "unconscious_detected",
    "violence_detected",
    "unauthorized_exit",
    "bed_fall_detected",
]

SEVERITIES = ["LOW", "MEDIUM", "HIGH"]
CAMERA_IDS = ["cam_01", "cam_02", "cam_03"]
EVENT_MESSAGES = {
    "fall_detected": "Fall detected from mock edge AI",
    "unconscious_detected": "Unconscious person detected from mock edge AI",
    "violence_detected": "Violence detected from mock edge AI",
    "unauthorized_exit": "Unauthorized exit detected from mock edge AI",
    "bed_fall_detected": "Bed fall detected from mock edge AI",
}


def get_env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {value}")


def build_event():
    event_type = random.choice(EVENT_TYPES)
    return build_safety_event(
        event_type=event_type,
        camera_id=random.choice(CAMERA_IDS),
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        severity=random.choice(SEVERITIES),
        message=EVENT_MESSAGES[event_type],
        source="edge-ai-mock",
    )


def connect_mqtt(host, port, client_id):
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
    )
    client.connect(host, port, keepalive=60)
    client.loop_start()
    return client


def main():
    host = os.getenv("MQTT_HOST", "localhost")
    topic = os.getenv("MQTT_TOPIC", "safety/events")
    client_id = os.getenv("MQTT_CLIENT_ID", "edge-ai-mock-001")

    try:
        port = get_env_int("MQTT_PORT", 1883)
        interval_seconds = get_env_int("PUBLISH_INTERVAL_SECONDS", 3)
    except ValueError as exc:
        print(f"[mock-edge-ai] Configuration error: {exc}", file=sys.stderr)
        return 1

    if interval_seconds <= 0:
        print(
            "[mock-edge-ai] Configuration error: PUBLISH_INTERVAL_SECONDS must be greater than 0",
            file=sys.stderr,
        )
        return 1

    try:
        mqtt_client = connect_mqtt(host, port, client_id)
    except Exception as exc:
        print(
            f"[mock-edge-ai] MQTT connection failed: host={host}, port={port}, client_id={client_id}, error={exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[mock-edge-ai] Publishing mock safety events to MQTT topic '{topic}' "
        f"at mqtt://{host}:{port} every {interval_seconds}s"
    )

    try:
        while True:
            event = build_event()
            payload = json.dumps(event, ensure_ascii=False)
            result = mqtt_client.publish(topic, payload, qos=0)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(
                    f"[mock-edge-ai] MQTT publish failed: topic={topic}, rc={result.rc}",
                    file=sys.stderr,
                )
                return 1
            print(f"[mock-edge-ai] published: {payload}", flush=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n[mock-edge-ai] Stopped by user")
        return 0
    except Exception as exc:
        print(f"[mock-edge-ai] MQTT publish failed: {exc}", file=sys.stderr)
        return 1
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
