"""Bounded MQTT delivery isolated from inference threads."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.publishers.event_outbox import SqliteEventOutbox


@dataclass(frozen=True, slots=True)
class _Message:
    payload: dict[str, Any]
    topic: str
    qos: int
    outbox_id: int | None = None


class AsyncMqttDelivery:
    """Own MQTT work; QoS 1 events survive retries and process restarts."""

    def __init__(self, publisher: Any, *, event_capacity: int = 100, retry_count: int = 2, outbox_path: str | Path | None = None) -> None:
        self._publisher = publisher
        self._events: queue.Queue[_Message] = queue.Queue(maxsize=event_capacity)
        self._outbox = SqliteEventOutbox(outbox_path) if outbox_path else None
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
        message = _Message(payload, topic, 1)
        if self._outbox is not None:
            try:
                message = _Message(payload, topic, 1, self._outbox.enqueue(payload, topic))
            except (OSError, sqlite3.Error):
                print("[mqtt][error] event outbox write failed", flush=True)
                return False
        try:
            self._events.put_nowait(message)
            return True
        except queue.Full:
            if self._outbox is not None:
                print("[mqtt][warning] event queue full; durable event deferred", flush=True)
                return True
            print("[mqtt][error] event queue full; event was not queued", flush=True)
            return False

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        close = getattr(self._publisher, "close", None)
        if callable(close):
            close()
        if self._outbox is not None:
            self._outbox.close()

    def _next_overlay(self) -> _Message | None:
        with self._lock:
            message = self._overlay
            self._overlay = None
            return message

    def _next_event(self) -> _Message | None:
        try:
            return self._events.get(timeout=0.05)
        except queue.Empty:
            if self._outbox is None:
                return None
            pending = self._outbox.next_pending()
            if pending is None:
                return None
            return _Message(pending.payload, pending.topic, 1, pending.outbox_id)

    def _run(self) -> None:
        while not self._stop.is_set() or (self._outbox is None and not self._events.empty()):
            message = self._next_event()
            if message is None:
                message = self._next_overlay()
                if message is None:
                    continue
            if self._publish(message):
                if message.outbox_id is not None and self._outbox is not None:
                    self._outbox.acknowledge(message.outbox_id)
            elif message.outbox_id is not None:
                time.sleep(0.05)

    def _publish(self, message: _Message) -> bool:
        for attempt in range(self._retry_count + 1):
            try:
                if not getattr(self._publisher, "connected", True):
                    connect = getattr(self._publisher, "connect", None)
                    if callable(connect):
                        connect()
                publish = self._publisher.publish
                try:
                    published = publish(message.payload, topic=message.topic, qos=message.qos)
                except TypeError:
                    published = publish(message.payload, topic=message.topic)
                if published is True or published is None:
                    return True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"[mqtt][error] delivery failed: {exc}", flush=True)
            if attempt < self._retry_count:
                time.sleep(0.05 * (attempt + 1))
        print("[mqtt][error] delivery exhausted retries", flush=True)
        return False
