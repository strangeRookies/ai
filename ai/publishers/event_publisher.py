import json
import os
import sys
import time

from ai.publishers.mqtt_identity import build_mqtt_client_id


def _env_int(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


class EventPublisher:
    def publish(self, payload, topic=None, qos=0):
        raise NotImplementedError


class ConsoleEventPublisher(EventPublisher):
    def publish(self, payload, topic=None, qos=0):
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
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        if username:
            self.client.username_pw_set(username=username, password=password or None)

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties=None):
        # Paho v2 passes a ReasonCode object here, which is comparable to 0
        # but cannot be converted with int().
        self.connected = _mqtt_connect_succeeded(reason_code)

    def _on_disconnect(self, _client, _userdata, _disconnect_flags=None, _reason_code=None, _properties=None):
        self.connected = False

    def connect(self):
        if self.client is None:
            return False
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            print(f"[mqtt] connection requested: mqtt://{self.host}:{self.port}, topic={self.topic}, client_id={self.client_id}")
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.connected = False
            print(
                f"[mqtt] connection failed: host={self.host}, port={self.port}, client_id={self.client_id}, error={exc}",
                file=sys.stderr,
            )
            return False

    def publish(self, payload, topic=None, qos=0):
        target_topic = topic or self.topic
        if not self.connected or self.client is None:
            print(
                f"[mqtt] publish skipped because MQTT client is not connected: "
                f"{_payload_context(payload, target_topic, connected=False, rc='n/a')}",
                file=sys.stderr,
            )
            return False
        try:
            publish_started_at_ms = _current_timestamp_ms()
            _stamp_event_publish_attempt(payload, publish_started_at_ms)
            print(
                f"[mqtt] publishing: {_payload_context(payload, target_topic, connected=self.connected, rc='pending')}",
                flush=True,
            )
            result = self.client.publish(target_topic, json.dumps(payload, ensure_ascii=False), qos=int(qos))
            publish_returned_at_ms = _current_timestamp_ms()
            if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
                self.connected = False
                print(
                    f"[mqtt] publish failed: {_payload_context(payload, target_topic, connected=self.connected, rc=result.rc, publish_returned_at_ms=publish_returned_at_ms)}",
                    file=sys.stderr,
                )
                return False
            print(
                f"[mqtt] published: {_payload_context(payload, target_topic, connected=self.connected, rc=result.rc, publish_returned_at_ms=publish_returned_at_ms)}",
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
        "status_topic": os.getenv("MQTT_STATUS_TOPIC", "safety/cameras/status"),
        "client_id": os.getenv("MQTT_CLIENT_ID", "strange-ai-local"),
        "username": os.getenv("MQTT_USERNAME") or None,
        "password": os.getenv("MQTT_PASSWORD") or None,
    }


def create_event_publisher(args, *, role="inference"):
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
        client_id=build_mqtt_client_id(
            getattr(args, "mqtt_client_id", None) or settings["client_id"],
            getattr(args, "camera_login_id", None) or getattr(args, "camera_id", "unknown-camera"),
            role,
        ),
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


def _payload_context(payload, topic, connected=None, rc=None, publish_returned_at_ms=None):
    if not isinstance(payload, dict):
        return _format_payload_context(topic, "unknown", "unknown", "unknown", "unknown", connected, rc, "none", "unknown")
    message_type = payload.get("messageType") or payload.get("message_type") or payload.get("event_type") or "unknown"
    stream_id = payload.get("streamId") or payload.get("camera_login_id") or payload.get("camera_id") or "unknown"
    camera_login_id = payload.get("cameraLoginId") or payload.get("camera_login_id") or stream_id or "unknown"
    frame_id = payload.get("frameId") or payload.get("frame_id") or "unknown"
    event_id = payload.get("eventId") or "none"
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return _format_payload_context(
        topic,
        message_type,
        stream_id,
        camera_login_id,
        frame_id,
        connected,
        rc,
        event_id,
        payload_bytes,
        payload=payload,
        publish_returned_at_ms=publish_returned_at_ms,
    )


def _format_payload_context(
    topic,
    message_type,
    stream_id,
    camera_login_id,
    frame_id,
    connected,
    rc,
    event_id,
    payload_bytes,
    payload=None,
    publish_returned_at_ms=None,
):
    connected_text = "unknown" if connected is None else str(bool(connected)).lower()
    rc_text = "unknown" if rc is None else str(rc)
    context = (
        f"topic={topic}, messageType={message_type}, streamId={stream_id}, "
        f"cameraLoginId={camera_login_id}, frameId={frame_id}, eventId={event_id}, "
        f"rc={rc_text}, connected={connected_text}, payloadBytes={payload_bytes}"
    )
    if isinstance(payload, dict) and _is_event_payload(payload):
        started_at = _optional_int(payload.get("mqttPublishStartedAtMs"))
        processed_at = _optional_int(payload.get("processedAtMs"))
        captured_at = _optional_int(payload.get("capturedAtMs"))
        returned_at = _optional_int(publish_returned_at_ms)
        if started_at is not None:
            context += f", mqttPublishStartedAtMs={started_at}"
        if processed_at is not None and started_at is not None:
            context += f", processedToMqttMs={max(0, started_at - processed_at)}"
        if captured_at is not None and started_at is not None:
            context += f", capturedToMqttMs={max(0, started_at - captured_at)}"
        if returned_at is not None and started_at is not None:
            context += f", mqttPublishCallMs={max(0, returned_at - started_at)}"
    return context


def _stamp_event_publish_attempt(payload, timestamp_ms):
    if not isinstance(payload, dict) or not _is_event_payload(payload):
        return
    payload["mqttPublishStartedAtMs"] = int(timestamp_ms)
    payload["mqttPublishedAtMs"] = int(timestamp_ms)


def _is_event_payload(payload):
    message_type = payload.get("messageType") or payload.get("message_type")
    if message_type == "event":
        return True
    return payload.get("eventId") is not None and payload.get("clip_url") is not None


def _current_timestamp_ms():
    return int(time.time() * 1000)


def _mqtt_connect_succeeded(reason_code):
    """Handle both legacy integer and Paho v2 ReasonCode connect results."""
    if reason_code == 0:
        return True
    return getattr(reason_code, "value", None) == 0


def _optional_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
