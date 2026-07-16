from __future__ import annotations

import argparse
import json
import signal
import statistics
import threading
import time
from pathlib import Path


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 3)


def stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }

    return {
        "count": len(values),
        "avg_ms": round(statistics.fmean(values), 3),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": round(max(values), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="event")
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise SystemExit("pip install paho-mqtt 실행 필요") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "alert_latency_events.jsonl"

    samples: list[dict] = []
    lock = threading.Lock()
    stopped = threading.Event()

    def stop_handler(*_):
        stopped.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"[mqtt-probe] connected reason={reason_code}")
        client.subscribe(args.topic, qos=1)

    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        if payload.get("messageType") != "event":
            return

        received_at_ms = time.time_ns() // 1_000_000

        captured = payload.get("capturedAtMs")
        processed = payload.get("processedAtMs")
        mqtt_started = payload.get("mqttPublishStartedAtMs")

        if captured is None:
            return

        captured = int(captured)
        processed = int(processed) if processed is not None else None
        mqtt_started = int(mqtt_started) if mqtt_started is not None else None

        row = {
            "eventId": payload.get("eventId"),
            "cameraLoginId": payload.get("cameraLoginId"),
            "frameId": payload.get("frameId"),
            "capturedAtMs": captured,
            "processedAtMs": processed,
            "mqttPublishStartedAtMs": mqtt_started,
            "subscriberReceivedAtMs": received_at_ms,
            "aiLatencyMs": (
                max(0, processed - captured)
                if processed is not None
                else None
            ),
            "processedToMqttMs": (
                max(0, mqtt_started - processed)
                if mqtt_started is not None and processed is not None
                else None
            ),
            "mqttTransportMs": (
                max(0, received_at_ms - mqtt_started)
                if mqtt_started is not None
                else None
            ),
            "e2eSubscriberMs": max(0, received_at_ms - captured),
        }

        with lock:
            samples.append(row)
            with jsonl_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(
                "[alert-latency] "
                f"event={row['eventId']} "
                f"camera={row['cameraLoginId']} "
                f"ai={row['aiLatencyMs']}ms "
                f"mqtt={row['mqttTransportMs']}ms "
                f"e2e={row['e2eSubscriberMs']}ms",
                flush=True,
            )

            if len(samples) >= args.max_events:
                stopped.set()

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"latency-probe-{int(time.time())}",
        )
    except (TypeError, AttributeError):
        client = mqtt.Client(client_id=f"latency-probe-{int(time.time())}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    deadline = time.monotonic() + args.duration_seconds

    while not stopped.is_set() and time.monotonic() < deadline:
        time.sleep(0.2)

    client.loop_stop()
    client.disconnect()

    with lock:
        rows = list(samples)

    ai = [float(r["aiLatencyMs"]) for r in rows if r["aiLatencyMs"] is not None]
    mqtt_queue = [
        float(r["processedToMqttMs"])
        for r in rows
        if r["processedToMqttMs"] is not None
    ]
    transport = [
        float(r["mqttTransportMs"])
        for r in rows
        if r["mqttTransportMs"] is not None
    ]
    e2e = [float(r["e2eSubscriberMs"]) for r in rows]

    summary = {
        "topic": args.topic,
        "events_received": len(rows),
        "ai_latency": stats(ai),
        "processed_to_mqtt": stats(mqtt_queue),
        "mqtt_transport": stats(transport),
        "e2e_subscriber": stats(e2e),
        "within_1_second_rate": (
            round(sum(v <= 1000 for v in e2e) / len(e2e), 6)
            if e2e
            else None
        ),
    }

    summary_path = output_dir / "alert_latency_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()