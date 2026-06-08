import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.publishers.event_publisher import mqtt_settings_from_env


def main():
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[mqtt-sub] paho-mqtt is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    try:
        settings = mqtt_settings_from_env()
    except ValueError as exc:
        print(f"[mqtt-sub] configuration error: {exc}", file=sys.stderr)
        return 1

    def on_connect(client, userdata, flags, reason_code, properties):
        del userdata, flags, properties
        print(f"[mqtt-sub] connected: host={settings['host']}, port={settings['port']}, topic={settings['topic']}")
        if reason_code == 0:
            client.subscribe(settings["topic"])

    def on_message(client, userdata, message):
        del client, userdata
        print(message.payload.decode("utf-8", errors="replace"), flush=True)

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{settings['client_id']}-subscriber",
        protocol=mqtt.MQTTv311,
    )
    if settings["username"]:
        client.username_pw_set(username=settings["username"], password=settings["password"] or None)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(settings["host"], settings["port"], keepalive=60)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[mqtt-sub] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
