"""Durable FIFO storage for MQTT QoS 1 safety events."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PendingEvent:
    outbox_id: int
    payload: dict[str, Any]
    topic: str


class SqliteEventOutbox:
    """Persist events before network delivery and remove them only after success."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS mqtt_event_outbox "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._connection.commit()
        self._lock = threading.Lock()

    def enqueue(self, payload: dict[str, Any], topic: str) -> int:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO mqtt_event_outbox(topic, payload) VALUES (?, ?)",
                (str(topic), encoded),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def next_pending(self) -> PendingEvent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, topic, payload FROM mqtt_event_outbox ORDER BY id LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return PendingEvent(outbox_id=int(row[0]), topic=str(row[1]), payload=json.loads(row[2]))

    def acknowledge(self, outbox_id: int) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM mqtt_event_outbox WHERE id = ?", (int(outbox_id),))
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def event_outbox_path(camera_login_id: str) -> Path:
    """Return the camera-scoped durable outbox path for a runtime MQTT worker."""
    root = Path(os.getenv("MQTT_EVENT_OUTBOX_DIR", "runs/mqtt_outbox"))
    safe_camera_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(camera_login_id))
    return root / f"{safe_camera_id or 'unknown-camera'}.sqlite3"
