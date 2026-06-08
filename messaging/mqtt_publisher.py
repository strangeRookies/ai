import json
import sys

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(self, host, port, topic, client_id, username="", password=""):
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
        if username:
            self.client.username_pw_set(username=username, password=password or None)
        self.connected = False

    def connect(self):
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self.connected = True
            print(f"[mqtt] connected: mqtt://{self.host}:{self.port}, topic={self.topic}")
        except Exception as exc:
            self.connected = False
            print(
                f"[mqtt] connection failed: host={self.host}, port={self.port}, client_id={self.client_id}, error={exc}",
                file=sys.stderr,
            )

    def publish_event(self, event):
        payload = json.dumps(event, ensure_ascii=True)
        print(f"[mqtt] event: {payload}", flush=True)
        if not self.connected:
            print("[mqtt] publish skipped because MQTT client is not connected", file=sys.stderr)
            return False
        try:
            result = self.client.publish(self.topic, payload, qos=0)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"[mqtt] publish failed: topic={self.topic}, rc={result.rc}", file=sys.stderr)
                return False
            return True
        except Exception as exc:
            print(f"[mqtt] publish failed: topic={self.topic}, error={exc}", file=sys.stderr)
            self.connected = False
            return False

    def close(self):
        if not self.connected:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
