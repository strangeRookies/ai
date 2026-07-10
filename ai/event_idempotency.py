"""In-process eventId / idempotency key de-duplication for publish/store paths."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class EventIdempotencyStore:
    """Bounded LRU of seen event IDs. Thread-safe."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        self.max_entries = max(1, int(max_entries))
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self.duplicate_total = 0
        self.accepted_total = 0

    def _key(self, event_id: str | None, *, camera_login_id: str | None = None) -> str | None:
        if event_id is None or str(event_id).strip() == "":
            return None
        if camera_login_id:
            return f"{camera_login_id}:{event_id}"
        return str(event_id)

    def accept(self, event_id: str | None, *, camera_login_id: str | None = None) -> bool:
        """Return True if first time; False if duplicate (do not store again)."""
        key = self._key(event_id, camera_login_id=camera_login_id)
        if key is None:
            # No id → cannot de-dupe; allow through
            return True
        with self._lock:
            if key in self._seen:
                self.duplicate_total += 1
                self._seen.move_to_end(key)
                return False
            self._seen[key] = time.time()
            self.accepted_total += 1
            while len(self._seen) > self.max_entries:
                self._seen.popitem(last=False)
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "accepted_total": self.accepted_total,
                "duplicate_total": self.duplicate_total,
                "size": len(self._seen),
                "max_entries": self.max_entries,
            }
