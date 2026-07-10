"""Tracking runtime metrics collector (safe for live inference path).

Collection/save failures never raise into the inference loop.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrackingRuntimeMetrics:
    track_created_total: int = 0
    track_removed_total: int = 0
    match_hard_total: int = 0
    match_soft_total: int = 0
    match_sole_total: int = 0
    match_rejected_total: int = 0
    active_track_count: int = 0
    track_lost_total: int = 0
    suspected_fragmentation_total: int = 0
    _track_durations: list[float] = field(default_factory=list)
    _track_started_at: dict[int, float] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def record_safe(self, fn_name: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(self, fn_name)(*args, **kwargs)
        except Exception as exc:  # never break inference
            logger.warning("tracking metrics %s failed: %s", fn_name, exc)

    def on_track_created(self, track_id: int, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        self.track_created_total += 1
        self._track_started_at[int(track_id)] = now
        self.active_track_count = max(self.active_track_count, len(self._track_started_at))

    def on_track_removed(self, track_id: int, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        tid = int(track_id)
        started = self._track_started_at.pop(tid, None)
        self.track_removed_total += 1
        self.track_lost_total += 1
        if started is not None:
            self._track_durations.append(max(0.0, now - started))

    def on_match(self, mode: str | None) -> None:
        label = (mode or "").lower()
        if label in {"hard", "hard_iou", "hard_match"}:
            self.match_hard_total += 1
        elif label in {"soft", "soft_match", "soft_center"}:
            self.match_soft_total += 1
        elif label in {"sole", "sole_track", "sole_detection"}:
            self.match_sole_total += 1
        else:
            # detector_track_id / generic match
            if "sole" in label:
                self.match_sole_total += 1
            elif "soft" in label:
                self.match_soft_total += 1
            else:
                self.match_hard_total += 1

    def on_match_rejected(self, reason: str | None = None) -> None:
        self.match_rejected_total += 1
        key = reason or "unknown"
        self.rejection_reasons[key] = self.rejection_reasons.get(key, 0) + 1

    def on_fragmentation_suspected(self, count: int = 1) -> None:
        self.suspected_fragmentation_total += int(count)

    def observe_active(self, count: int) -> None:
        self.active_track_count = max(self.active_track_count, int(count))

    def ingest_tracker_events(self, events: list[dict], now: float | None = None) -> None:
        """Map SimpleTrackAssigner last_events into counters."""
        now = time.time() if now is None else float(now)
        try:
            for event in events or []:
                kind = event.get("event")
                if kind == "new_track":
                    tid = event.get("trackId")
                    if tid is not None:
                        self.on_track_created(int(tid), now=now)
                elif kind == "lost":
                    tid = event.get("trackId")
                    if tid is not None:
                        self.on_track_removed(int(tid), now=now)
                elif kind == "match":
                    self.on_match(str(event.get("reason") or "match"))
                elif kind == "id_switch_like":
                    self.on_fragmentation_suspected(1)
                elif kind in {"reject", "filtered", "filter"}:
                    # SimpleTrackAssigner emits event="filter" (low_confidence / tiny_box).
                    self.on_match_rejected(str(event.get("reason") or kind))
        except Exception as exc:
            logger.warning("ingest_tracker_events failed: %s", exc)

    @property
    def average_track_duration(self) -> float | None:
        if not self._track_durations:
            return None
        return sum(self._track_durations) / len(self._track_durations)

    def summary(self) -> dict[str, Any]:
        return {
            "track_created_total": self.track_created_total,
            "track_removed_total": self.track_removed_total,
            "match_hard_total": self.match_hard_total,
            "match_soft_total": self.match_soft_total,
            "match_sole_total": self.match_sole_total,
            "match_rejected_total": self.match_rejected_total,
            "active_track_count": self.active_track_count,
            "track_lost_total": self.track_lost_total,
            "suspected_fragmentation_total": self.suspected_fragmentation_total,
            "average_track_duration": self.average_track_duration,
            "rejection_reasons": dict(self.rejection_reasons),
            "runtime_seconds": max(0.0, time.time() - self.started_at),
        }

    def save_session_summary(self, output_dir: str | Path, *, run_id: str | None = None) -> dict[str, Path]:
        """Write JSON + CSV summary. Failures are logged only."""
        paths: dict[str, Path] = {}
        try:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            rid = run_id or f"session-{int(time.time())}"
            summary = self.summary()
            summary["runId"] = rid
            json_path = out / f"{rid}_tracking_metrics.json"
            csv_path = out / f"{rid}_tracking_metrics.csv"
            json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["metric", "value"])
                for key, value in summary.items():
                    if key == "rejection_reasons":
                        continue
                    writer.writerow([key, value])
                for reason, count in summary.get("rejection_reasons", {}).items():
                    writer.writerow([f"rejection_reason:{reason}", count])
            paths["json"] = json_path
            paths["csv"] = csv_path
        except Exception as exc:
            logger.warning("tracking metrics save failed: %s", exc)
        return paths
