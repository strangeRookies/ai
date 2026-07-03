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
    def publish(self, payload, topic=None):
        raise NotImplementedError


class ConsoleEventPublisher(EventPublisher):
    def publish(self, payload, topic=None):
        topic_text = topic or "console"
        print(f"[event][topic={topic_text}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


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

    def publish(self, payload, topic=None):
        target_topic = topic or self.topic
        if not self.connected or self.client is None:
            if self.client is None or not self.connect():
                print(
                    f"[mqtt] publish skipped because MQTT client is not connected: "
                    f"{_payload_context(payload, target_topic, connected=False, rc='n/a')}",
                    file=sys.stderr,
                )
                return False
        try:
            print(
                f"[mqtt] publishing: {_payload_context(payload, target_topic, connected=self.connected, rc='pending')}",
                flush=True,
            )
            result = self.client.publish(target_topic, json.dumps(payload, ensure_ascii=False), qos=0)
            if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
                print(
                    f"[mqtt] publish failed: {_payload_context(payload, target_topic, connected=self.connected, rc=result.rc)}",
                    file=sys.stderr,
                )
                return False
            print(
                f"[mqtt] published: {_payload_context(payload, target_topic, connected=self.connected, rc=result.rc)}",
                flush=True,
            )
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.connected = False
            print(
                f"[mqtt] publish failed: {_payload_context(payload, target_topic, connected=False, rc='exception')}, error={exc}",
                file=sys.stderr,
            )
            return False

    def close(self):
        if not self.connected or self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False


def mqtt_settings_from_env():
    legacy_topic = os.getenv("MQTT_TOPIC")
    return {
        "host": os.getenv("MQTT_HOST", "localhost"),
        "port": _env_int("MQTT_PORT", 1883),
        "topic": legacy_topic or "event",
        "camera_topic": os.getenv("MQTT_CAMERA_TOPIC", "camera"),
        "event_topic": os.getenv("MQTT_EVENT_TOPIC") or legacy_topic or "event",
        "client_id": os.getenv("MQTT_CLIENT_ID", "strange-ai-local"),
        "username": os.getenv("MQTT_USERNAME") or None,
        "password": os.getenv("MQTT_PASSWORD") or None,
    }


def create_event_publisher(args):
    publisher_mode = getattr(args, "publisher", None) or ("console" if getattr(args, "dry_run", False) else "mqtt")
    if publisher_mode == "console":
        return ConsoleEventPublisher(), "console"
    settings = mqtt_settings_from_env()
    publisher = MqttEventPublisher(
        host=getattr(args, "mqtt_host", None) or settings["host"],
        port=getattr(args, "mqtt_port", None) or settings["port"],
        topic=getattr(args, "mqtt_event_topic", None)
        or getattr(args, "mqtt_topic", None)
        or settings["event_topic"],
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


def mqtt_topic_settings_from_args(args):
    settings = mqtt_settings_from_env()
    return {
        "camera_topic": getattr(args, "mqtt_camera_topic", None) or settings["camera_topic"],
        "event_topic": getattr(args, "mqtt_event_topic", None)
        or getattr(args, "mqtt_topic", None)
        or settings["event_topic"],
    }


def _payload_context(payload, topic, connected=None, rc=None):
    if not isinstance(payload, dict):
        return _format_payload_context(topic, "unknown", "unknown", "unknown", "unknown", connected, rc)
    message_type = payload.get("messageType") or payload.get("message_type") or payload.get("event_type") or "unknown"
    stream_id = payload.get("streamId") or payload.get("camera_login_id") or payload.get("camera_id") or "unknown"
    camera_login_id = payload.get("cameraLoginId") or payload.get("camera_login_id") or stream_id or "unknown"
    frame_id = payload.get("frameId") or payload.get("frame_id") or "unknown"
    event_id = payload.get("eventId")
    event_text = f", eventId={event_id}" if event_id else ""
    return (
        _format_payload_context(topic, message_type, stream_id, camera_login_id, frame_id, connected, rc)
        + event_text
    )


def _format_payload_context(topic, message_type, stream_id, camera_login_id, frame_id, connected, rc):
    connected_text = "unknown" if connected is None else str(bool(connected)).lower()
    rc_text = "unknown" if rc is None else str(rc)
    return (
        f"topic={topic}, messageType={message_type}, streamId={stream_id}, "
        f"cameraLoginId={camera_login_id}, frameId={frame_id}, rc={rc_text}, connected={connected_text}"
    )
