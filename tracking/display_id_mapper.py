"""Camera-local display ID mapper for overlay labels.

Raw track IDs from the tracker are globally incrementing integers that can
grow large (e.g. 70+) over the lifetime of a camera session. This module
maintains a stable *display_id* (starting from 1) per camera that is
friendlier for operator reading.

Rules
-----
- display_id values start at 1 for the first visible track.
- A display_id is stable for as long as the raw track remains active.
- When a raw track is removed, its display_id is freed and may be reused
  by a future track (smallest available ID is reused first).
- The mapping is never reset between frames; it is reset only if the mapper
  object is recreated (e.g. on camera reconnect).
- Raw track_id is preserved internally and in diagnostics; only the overlay
  label substitutes display_id.
"""

from __future__ import annotations


class DisplayIdMapper:
    """Map raw tracker IDs to compact camera-local display IDs."""

    def __init__(self) -> None:
        # raw_track_id -> display_id
        self._raw_to_display: dict[int, int] = {}
        # display_id -> raw_track_id (reverse for diagnostics)
        self._display_to_raw: dict[int, int] = {}
        # Freed display IDs available for reuse (sorted heap-like behaviour)
        self._free_ids: list[int] = []
        # Next display_id to allocate when no freed IDs are available
        self._next_display_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, active_raw_ids: set[int]) -> None:
        """Synchronise the mapping with the currently active raw track IDs.

        - New IDs in *active_raw_ids* that are not yet mapped receive a
          display_id (smallest available, then next counter).
        - IDs previously mapped but absent from *active_raw_ids* are freed.

        Parameters
        ----------
        active_raw_ids:
            The set of raw track IDs currently alive in the tracker.
        """
        # Free tracks that are no longer active
        gone = set(self._raw_to_display) - active_raw_ids
        for raw_id in gone:
            display_id = self._raw_to_display.pop(raw_id)
            del self._display_to_raw[display_id]
            self._free_ids.append(display_id)
            self._free_ids.sort()

        # Assign display IDs to newly seen raw tracks
        for raw_id in sorted(active_raw_ids):
            if raw_id not in self._raw_to_display:
                display_id = self._allocate()
                self._raw_to_display[raw_id] = display_id
                self._display_to_raw[display_id] = raw_id

    def display_id(self, raw_track_id: int) -> int | None:
        """Return the display_id for a raw track ID, or None if unmapped."""
        return self._raw_to_display.get(int(raw_track_id))

    def mapping_snapshot(self) -> dict[str, dict]:
        """Return a JSON-serialisable snapshot for diagnostics/summary.

        Returns a dict with two sub-dicts keyed by string for easy JSON:
        ``raw_to_display`` and ``display_to_raw``.
        """
        return {
            "raw_to_display": {str(k): v for k, v in self._raw_to_display.items()},
            "display_to_raw": {str(k): v for k, v in self._display_to_raw.items()},
        }

    def reset(self) -> None:
        """Reset all mappings (e.g. on camera reconnect)."""
        self._raw_to_display.clear()
        self._display_to_raw.clear()
        self._free_ids.clear()
        self._next_display_id = 1

    def transfer_raw_id(self, old_raw_id: int, new_raw_id: int) -> bool:
        """Move display mapping from one raw track id to another (recovery migrate)."""
        old_id, new_id = int(old_raw_id), int(new_raw_id)
        if old_id == new_id:
            return False
        if old_id not in self._raw_to_display:
            return False
        display_id = self._raw_to_display.pop(old_id)
        if new_id in self._raw_to_display:
            freed = self._raw_to_display.pop(new_id)
            self._display_to_raw.pop(freed, None)
            self._free_ids.append(freed)
            self._free_ids.sort()
        self._raw_to_display[new_id] = display_id
        self._display_to_raw[display_id] = new_id
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _allocate(self) -> int:
        if self._free_ids:
            return self._free_ids.pop(0)
        display_id = self._next_display_id
        self._next_display_id += 1
        return display_id
