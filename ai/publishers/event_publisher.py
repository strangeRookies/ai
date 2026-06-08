import json
import os
import sys


def _env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


class EventPublisher:
    def publish(self, payload):
        raise NotImplementedError


class ConsoleEventPublisher(EventPublisher):
    def publish(self, payload):
        print(f"[event] {json.dumps(payload, ensure_ascii=False)}", flush=True)


class MqttEventPublisher(EventPublisher):
    def __init__(self, host, port, topic, client_id, username=None, password=None):
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.client_id = client_id
        self.connected = False
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.mqtt = None
            self.client = None
            print("[mqtt] paho-mqtt is not installed; MQTT publish disabled", file=sys.stderr)
            return
        self.mqtt = mqtt
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
        if username:
            self.client.username_pw_set(username=username, password=password or None)

    def connect(self):
        if self.client is None:
            return False
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self.connected = True
            print(f"[mqtt] connected: mqtt://{self.host}:{self.port}, topic={self.topic}, client_id={self.client_id}")
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.connected = False
            print(
                f"[mqtt] connection failed: host={self.host}, port={self.port}, client_id={self.client_id}, error={exc}",
                file=sys.stderr,
            )
            return False

    def publish(self, payload):
        if not self.connected or self.client is None:
            print("[mqtt] publish skipped because MQTT client is not connected", file=sys.stderr)
            return False
        try:
            result = self.client.publish(self.topic, json.dumps(payload, ensure_ascii=False), qos=0)
            if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
                print(f"[mqtt] publish failed: topic={self.topic}, rc={result.rc}", file=sys.stderr)
                return False
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.connected = False
            print(f"[mqtt] publish failed: topic={self.topic}, error={exc}", file=sys.stderr)
            return False

    def close(self):
        if not self.connected or self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False


def mqtt_settings_from_env():
    return {
        "host": os.getenv("MQTT_HOST", "localhost"),
        "port": _env_int("MQTT_PORT", 1883),
        "topic": os.getenv("MQTT_TOPIC", "safety/events"),
        "client_id": os.getenv("MQTT_CLIENT_ID", "strange-ai-local"),
        "username": os.getenv("MQTT_USERNAME") or None,
        "password": os.getenv("MQTT_PASSWORD") or None,
    }


def create_event_publisher(args):
    publisher_mode = getattr(args, "publisher", None) or ("console" if args.dry_run else "mqtt")
    if publisher_mode == "console":
        return ConsoleEventPublisher(), "console"
    settings = mqtt_settings_from_env()
    publisher = MqttEventPublisher(
        host=getattr(args, "mqtt_host", None) or settings["host"],
        port=getattr(args, "mqtt_port", None) or settings["port"],
        topic=getattr(args, "mqtt_topic", None) or settings["topic"],
        client_id=getattr(args, "mqtt_client_id", None) or settings["client_id"],
        username=getattr(args, "mqtt_username", None) or settings["username"],
        password=getattr(args, "mqtt_password", None) or settings["password"],
    )
    publisher.connect()
    return publisher, "mqtt"


def build_event_payload(camera_id, frame_idx, timestamp, event_type, score, boxes, snapshot_path=None):
    return {
        "camera_id": camera_id,
        "frame_idx": int(frame_idx),
        "timestamp": float(timestamp),
        "event_type": event_type,
        "score": float(score),
        "confidence": float(score),
        "boxes": boxes,
        "bbox": boxes[0] if boxes else None,
        "snapshot_path": snapshot_path,
    }
