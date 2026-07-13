"""Safe migration of per-track runtime state when recovery assigns a new track_id.

Recovery must not force the tracker to reuse a lost source track_id. Instead the
pipeline mints (or continues) a recovery track_id and moves sequence buffers,
Fall/Faint lifecycle, display/overlay mapping, and incident binding to that id.
"""
from __future__ import annotations

from typing import Any


def _move_dict_key(mapping: dict, old_key: Any, new_key: Any) -> bool:
    if old_key == new_key:
        return False
    if old_key not in mapping:
        return False
    if new_key in mapping:
        # Preserve both sides until a compatible merge policy is available.
        return False
    mapping[new_key] = mapping.pop(old_key)
    return True


def migrate_sequence_buffer(sequence_buffer: Any, old_track_id: int, new_track_id: int) -> bool:
    """Move per-track sequence buffer entries from old_track_id → new_track_id."""
    if sequence_buffer is None or int(old_track_id) == int(new_track_id):
        return False
    old_id, new_id = int(old_track_id), int(new_track_id)
    if hasattr(sequence_buffer, "migrate_track_id"):
        return bool(sequence_buffer.migrate_track_id(old_id, new_id))

    moved = False
    for attr in ("_buffers", "_last_seen_at", "_last_detection_by_track", "last_sequence_diagnostics", "sequences_generated_by_track"):
        mapping = getattr(sequence_buffer, attr, None)
        if isinstance(mapping, dict):
            moved = _move_dict_key(mapping, old_id, new_id) or moved
    return moved


def migrate_display_id_mapper(display_id_mapper: Any, old_track_id: int, new_track_id: int) -> bool:
    """Keep the same operator display_id under the new raw track_id."""
    if display_id_mapper is None or int(old_track_id) == int(new_track_id):
        return False
    old_id, new_id = int(old_track_id), int(new_track_id)
    if hasattr(display_id_mapper, "transfer_raw_id"):
        return bool(display_id_mapper.transfer_raw_id(old_id, new_id))
    raw_to = getattr(display_id_mapper, "_raw_to_display", None)
    disp_to = getattr(display_id_mapper, "_display_to_raw", None)
    if not isinstance(raw_to, dict) or not isinstance(disp_to, dict):
        return False
    if old_id not in raw_to:
        return False
    display_id = raw_to.pop(old_id)
    # Free new_id mapping if it already exists
    if new_id in raw_to:
        old_disp = raw_to.pop(new_id)
        disp_to.pop(old_disp, None)
        free = getattr(display_id_mapper, "_free_ids", None)
        if isinstance(free, list):
            free.append(old_disp)
            free.sort()
    raw_to[new_id] = display_id
    disp_to[display_id] = new_id
    return True


def migrate_overlay_publish_state(overlay_publish_state: Any, old_track_id: int, new_track_id: int) -> bool:
    if overlay_publish_state is None or int(old_track_id) == int(new_track_id):
        return False
    old_id, new_id = int(old_track_id), int(new_track_id)
    if hasattr(overlay_publish_state, "migrate_track"):
        return bool(overlay_publish_state.migrate_track(old_id, new_id))
    signals = getattr(overlay_publish_state, "signals_by_track", None)
    if not isinstance(signals, dict):
        return False
    return _move_dict_key(signals, old_id, new_id)


def migrate_faint_post_processor(post_processor: Any, camera_id: str, old_track_id: int, new_track_id: int) -> bool:
    if post_processor is None or int(old_track_id) == int(new_track_id):
        return False
    old_id, new_id = int(old_track_id), int(new_track_id)
    if hasattr(post_processor, "migrate_track"):
        return bool(post_processor.migrate_track(camera_id, old_id, new_id))

    from ai.action.faint_post_processing import event_state_key

    moved = False
    old_key = event_state_key(camera_id, old_id)
    new_key = event_state_key(camera_id, new_id)
    for attr in ("_consecutive_by_camera", "_last_seen_ts"):
        mapping = getattr(post_processor, attr, None)
        if isinstance(mapping, dict):
            moved = _move_dict_key(mapping, old_key, new_key) or moved

    sm = getattr(post_processor, "_state_machine", None)
    if sm is not None:
        moved = migrate_fall_state_machine(sm, camera_id, old_id, new_id) or moved

    pe = getattr(post_processor, "_posture_estimator", None)
    if pe is not None and hasattr(pe, "reset_track"):
        # posture history is keyed by track_key strings
        hist = getattr(pe, "_history", None)
        cy = getattr(pe, "_center_y", None)
        if isinstance(hist, dict):
            moved = _move_dict_key(hist, old_key, new_key) or moved
        if isinstance(cy, dict):
            moved = _move_dict_key(cy, old_key, new_key) or moved
    return moved


def migrate_fall_state_machine(state_machine: Any, camera_id: str, old_track_id: int, new_track_id: int) -> bool:
    if state_machine is None or int(old_track_id) == int(new_track_id):
        return False
    old_id, new_id = int(old_track_id), int(new_track_id)
    if hasattr(state_machine, "migrate_track"):
        return bool(state_machine.migrate_track(camera_id, old_id, new_id))
    from ai.action.fall_event_state import track_state_key

    tracks = getattr(state_machine, "_tracks", None)
    if not isinstance(tracks, dict):
        return False
    return _move_dict_key(tracks, track_state_key(camera_id, old_id), track_state_key(camera_id, new_id))


def migrate_track_runtime_state(
    *,
    camera_id: str,
    old_track_id: int,
    new_track_id: int,
    sequence_buffer: Any = None,
    post_processor: Any = None,
    display_id_mapper: Any = None,
    overlay_publish_state: Any = None,
) -> dict[str, bool]:
    """Migrate all known runtime maps for a track id change. Returns per-target flags."""
    return {
        "sequence_buffer": migrate_sequence_buffer(sequence_buffer, old_track_id, new_track_id),
        "post_processor": migrate_faint_post_processor(post_processor, camera_id, old_track_id, new_track_id),
        "display_id_mapper": migrate_display_id_mapper(display_id_mapper, old_track_id, new_track_id),
        "overlay_publish_state": migrate_overlay_publish_state(overlay_publish_state, old_track_id, new_track_id),
    }


def finalize_recovery_detections(
    detections: list[dict],
    *,
    camera_login_id: str,
    incident_recovery: Any,
    tracker: Any = None,
    now: float | None = None,
    sequence_buffer: Any = None,
    post_processor: Any = None,
    display_id_mapper: Any = None,
    overlay_publish_state: Any = None,
) -> tuple[list[dict], list[dict]]:
    """Assign track ids for recovery_relink rows and migrate state to the new id.

    Does not force the lost source track_id. First recovery mints a new id and
    migrates; later recovery frames for the same incident continue under the
    already-linked active_track_id.
    """
    out: list[dict] = []
    migrations: list[dict] = []
    for det in detections:
        if not det.get("recovery_relink"):
            out.append(det)
            continue

        item = dict(det)
        incident_id = item.get("incident_id")
        from_id = item.get("recovered_from_track_id")
        from_id_int = int(from_id) if from_id is not None else None

        assigned = incident_recovery.assign_recovery_track(
            camera_login_id=camera_login_id,
            incident_id=incident_id,
            detection=item,
            tracker=tracker,
            now=now,
            recovered_from_track_id=from_id_int,
        )
        if assigned.get("recovery_rejected") or assigned.get("track_id") is None:
            continue
        new_id = int(assigned["track_id"])
        item.update(assigned)

        if from_id_int is not None and from_id_int != new_id:
            flags = migrate_track_runtime_state(
                camera_id=camera_login_id,
                old_track_id=from_id_int,
                new_track_id=new_id,
                sequence_buffer=sequence_buffer,
                post_processor=post_processor,
                display_id_mapper=display_id_mapper,
                overlay_publish_state=overlay_publish_state,
            )
            migrations.append(
                {
                    "incident_id": incident_id,
                    "from_track_id": from_id_int,
                    "to_track_id": new_id,
                    "migrated": flags,
                }
            )
        out.append(item)
    return out, migrations
