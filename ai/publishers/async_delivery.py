"""Bounded MQTT delivery isolated from inference threads."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _Message:
    payload: dict[str, Any]
    topic: str
    qos: int


class AsyncMqttDelivery:
    """Own MQTT connect/retry/publish work; callers only enqueue payloads."""

    def __init__(self, publisher: Any, *, event_capacity: int = 100, retry_count: int = 2) -> None:
        self._publisher = publisher
        self._events: queue.Queue[_Message] = queue.Queue(maxsize=event_capacity)
        self._overlay: _Message | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._retry_count = max(0, retry_count)
        self._thread = threading.Thread(target=self._run, name="mqtt-delivery", daemon=True)
        self._thread.start()

    def publish(self, payload: dict[str, Any], topic: str | None = None, qos: int = 0) -> bool:
        target_topic = topic or ""
        if not target_topic:
            return False
        if int(qos) == 1:
            return self.enqueue_event(payload, target_topic)
        return self.enqueue_overlay(payload, target_topic)

    def enqueue_overlay(self, payload: dict[str, Any], topic: str) -> bool:
        with self._lock:
            self._overlay = _Message(payload, topic, 0)
        return True

    def enqueue_event(self, payload: dict[str, Any], topic: str) -> bool:
        try:
            self._events.put_nowait(_Message(payload, topic, 1))
            return True
        except queue.Full:
            print("[mqtt][error] event queue full; event was not queued", flush=True)
            return False

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        close = getattr(self._publisher, "close", None)
        if callable(close):
            close()

    def _next_overlay(self) -> _Message | None:
        with self._lock:
            message = self._overlay
            self._overlay = None
            return message

    def _run(self) -> None:
        while not self._stop.is_set() or not self._events.empty():
            try:
                message = self._events.get(timeout=0.05)
            except queue.Empty:
                message = self._next_overlay()
                if message is None:
                    continue
            self._publish(message)

    def _publish(self, message: _Message) -> None:
        for attempt in range(self._retry_count + 1):
            try:
                if not getattr(self._publisher, "connected", False):
                    self._publisher.connect()
                if self._publisher.publish(message.payload, topic=message.topic, qos=message.qos) is True:
                    return
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[mqtt][error] delivery failed: {exc}", flush=True)
            if attempt < self._retry_count:
                time.sleep(0.05 * (attempt + 1))
        print("[mqtt][error] delivery exhausted retries", flush=True)
