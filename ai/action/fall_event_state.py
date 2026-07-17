"""Per-track Fall/Faint lifecycle state machine.

Policies:
- NEW_FALL only on first confirm (NORMAL/RECOVERED → … → POST_FALL_LYING).
- While POST_FALL_LYING, never re-emit NEW_FALL (cooldown alone is not enough).
- After ``unrecovered_after_seconds`` still alert/lying → emit UNRECOVERED
  (FAINT_SUSPECTED / FALL_UNRECOVERED), not a second Fall.
- 10s camera cooldown remains an auxiliary gate in the post-processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class FallState(str, Enum):
    NORMAL = "NORMAL"
    FALL_CANDIDATE = "FALL_CANDIDATE"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    POST_FALL_LYING = "POST_FALL_LYING"
    RECOVERED = "RECOVERED"


class LifecycleKind(str, Enum):
    NONE = "none"
    NEW_FALL = "new_fall"
    SUPPRESS_NEW_FALL = "suppress_new_fall"  # still lying; no second Fall
    UNRECOVERED = "unrecovered"  # sustained risk after unrecovered_after_seconds
    # backward-compatible alias used by older tests/call sites
    SUPPRESS = "suppress_new_fall"


# Public MQTT / payload type strings
EVENT_TYPE_FAINT_SUSPECTED = "FAINT_SUSPECTED"
EVENT_TYPE_FALL_UNRECOVERED = "FALL_UNRECOVERED"


@dataclass
class LifecycleDecision:
    kind: LifecycleKind
    state: FallState
    event_id: str | None = None
    reason: str | None = None
    original_event_id: str | None = None
    duration_sec: float | None = None
    event_type: str | None = None  # payload type override for unrecovered
    faint_prob: float | None = None
    consecutive_count: int | None = None
    prev_lifecycle_state: str | None = None
    next_lifecycle_state: str | None = None

    @property
    def allow_new_fall_alert(self) -> bool:
        return self.kind == LifecycleKind.NEW_FALL

    @property
    def allow_unrecovered_alert(self) -> bool:
        return self.kind == LifecycleKind.UNRECOVERED

    @property
    def should_publish(self) -> bool:
        return self.kind in (LifecycleKind.NEW_FALL, LifecycleKind.UNRECOVERED)


@dataclass
class FallTrackState:
    state: FallState = FallState.NORMAL
    last_transition_ts: float = 0.0
    candidate_count: int = 0
    confirmed_ts: float | None = None
    lying_since_ts: float | None = None
    last_event_id: str | None = None
    last_unrecovered_ts: float | None = None
    last_persistent_ts: float | None = None  # alias for last_unrecovered_ts
    recover_count: int = 0
    saw_upright: bool = False
    saw_upright_to_lying: bool = False
    last_posture: str | None = None
    confirm_faint_prob: float | None = None
    confirm_consecutive: int | None = None
    confirm_prev_state: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def track_state_key(camera_id: str, track_id: Any | None = None) -> str:
    return f"{camera_id}:track:{track_id}" if track_id is not None else str(camera_id)


def unrecovered_event_type_for_prediction(
    prediction: dict | None,
    *,
    movement_level: str | None = None,
    posture_label: str | None = None,
) -> str:
    """Choose sustained-risk event type.

    Policy (OBJECTIVE):
    - movement still/low → FAINT_SUSPECTED
    - otherwise (or uncertain motion) → FALL_UNRECOVERED
    Prediction label can still bias when movement is unknown.
    """
    ml = (movement_level or "").lower()
    if ml in {"still", "low"}:
        return EVENT_TYPE_FAINT_SUSPECTED
    if ml in {"high"}:
        return EVENT_TYPE_FALL_UNRECOVERED
    # uncertain movement
    label = str((prediction or {}).get("label") or "Faint").strip().lower()
    if label in {"fall", "fall_detected"} or (posture_label == "lying_like" and ml in {"", "unknown"}):
        return EVENT_TYPE_FALL_UNRECOVERED
    if ml in {"", "unknown"}:
        return EVENT_TYPE_FALL_UNRECOVERED
    return EVENT_TYPE_FAINT_SUSPECTED


class FallEventStateMachine:
    """Track-level fall lifecycle."""

    def __init__(
        self,
        *,
        min_consecutive_faint: int = 3,
        recover_consecutive: int = 3,
        require_upright_to_lying: bool = False,
        block_upright_faint: bool = True,
        unrecovered_after_seconds: float = 10.0,
        unrecovered_repeat_seconds: float = 30.0,
    ):
        self.min_consecutive_faint = max(1, int(min_consecutive_faint))
        self.recover_consecutive = max(1, int(recover_consecutive))
        self.require_upright_to_lying = bool(require_upright_to_lying)
        # Block NEW_FALL while currently upright_like without upright→lying transition.
        self.block_upright_faint = bool(block_upright_faint)
        self.unrecovered_after_seconds = max(0.0, float(unrecovered_after_seconds))
        # 0 => emit unrecovered at most once until recover
        self.unrecovered_repeat_seconds = max(0.0, float(unrecovered_repeat_seconds))
        self._tracks: dict[str, FallTrackState] = {}

    def get_state(self, camera_id: str, track_id: Any | None = None) -> FallState:
        track = self._tracks.get(track_state_key(camera_id, track_id))
        return track.state if track is not None else FallState.NORMAL

    def get_track(self, camera_id: str, track_id: Any | None = None) -> FallTrackState | None:
        return self._tracks.get(track_state_key(camera_id, track_id))

    def reset_all(self) -> None:
        self._tracks.clear()

    def reset_track(self, camera_id: str, track_id: Any | None = None) -> None:
        self._tracks.pop(track_state_key(camera_id, track_id), None)

    def migrate_track(self, camera_id: str, old_track_id: Any, new_track_id: Any) -> bool:
        """Move lifecycle state from old_track_id to new_track_id (incident recovery)."""
        if old_track_id is None or new_track_id is None:
            return False
        if str(old_track_id) == str(new_track_id):
            return False
        old_key = track_state_key(camera_id, old_track_id)
        new_key = track_state_key(camera_id, new_track_id)
        if old_key not in self._tracks:
            return False
        if new_key in self._tracks:
            self._tracks.pop(old_key, None)
            return True
        self._tracks[new_key] = self._tracks.pop(old_key)
        return True

    def set_last_event_id(self, camera_id: str, track_id: Any | None, event_id: str) -> None:
        """Rewrite last_event_id only (lineage rewrite for spatial dedup).

        Does not touch state/confirmed_ts/lying_since_ts — the track's own lifecycle
        keeps progressing normally; this only makes future SUPPRESS_NEW_FALL/UNRECOVERED
        decisions report the originally-published event as their original_event_id
        instead of an event_id that was never actually published.
        """
        st = self._tracks.get(track_state_key(camera_id, track_id))
        if st is None:
            return
        st.last_event_id = event_id

    def revert_confirm_to_candidate(self, camera_id: str, track_id: Any | None = None) -> None:
        """If NEW_FALL was decided but camera cooldown blocked publish, roll back."""
        st = self._tracks.get(track_state_key(camera_id, track_id))
        if st is None or st.state != FallState.POST_FALL_LYING:
            return
        st.state = FallState.FALL_CANDIDATE
        st.candidate_count = max(0, self.min_consecutive_faint - 1)
        st.confirmed_ts = None
        st.lying_since_ts = None
        st.last_event_id = None
        st.last_unrecovered_ts = None
        st.recover_count = 0

    def update(
        self,
        camera_id: str,
        timestamp: float,
        *,
        track_id: Any | None = None,
        is_alert: bool = False,
        posture_label: str | None = None,
        upright_to_lying: bool | None = None,
        prediction: dict | None = None,
        movement_level: str | None = None,
        lying_like: bool | None = None,
    ) -> LifecycleDecision:
        key = track_state_key(camera_id, track_id)
        st = self._tracks.setdefault(key, FallTrackState())
        ts = float(timestamp)

        if posture_label == "upright_like":
            st.saw_upright = True
            st.last_posture = posture_label
        elif posture_label is not None:
            st.last_posture = posture_label
        if upright_to_lying:
            st.saw_upright_to_lying = True

        if st.state == FallState.FALL_CONFIRMED:
            st.state = FallState.POST_FALL_LYING
            st.lying_since_ts = st.lying_since_ts or ts

        # --- POST_FALL_LYING ---
        if st.state == FallState.POST_FALL_LYING:
            still_lying = lying_like if lying_like is not None else (posture_label == "lying_like" or is_alert)
            if still_lying or is_alert:
                st.recover_count = 0
                return self._post_fall_alert_decision(
                    st,
                    ts,
                    prediction,
                    movement_level=movement_level,
                    posture_label=posture_label,
                    lying_like=bool(still_lying),
                )

            # non-alert: recovery path
            st.recover_count += 1
            if st.recover_count >= self.recover_consecutive:
                st.state = FallState.RECOVERED
                st.last_transition_ts = ts
                st.candidate_count = 0
                st.recover_count = 0
                return LifecycleDecision(LifecycleKind.NONE, st.state, reason="recovered_from_lying")
            return LifecycleDecision(
                LifecycleKind.SUPPRESS_NEW_FALL,
                st.state,
                event_id=st.last_event_id,
                reason="post_fall_recovering",
            )

        if st.state == FallState.RECOVERED and not is_alert:
            st.candidate_count = 0
            st.state = FallState.NORMAL
            st.last_transition_ts = ts
            return LifecycleDecision(LifecycleKind.NONE, st.state, reason="recovered_to_normal")

        if not is_alert:
            if st.state == FallState.FALL_CANDIDATE:
                st.state = FallState.NORMAL
                st.last_transition_ts = ts
            st.candidate_count = 0
            st.recover_count = 0
            return LifecycleDecision(LifecycleKind.NONE, st.state, reason="non_alert")

        if st.state in (FallState.NORMAL, FallState.RECOVERED):
            st.state = FallState.FALL_CANDIDATE
            st.last_transition_ts = ts
            st.candidate_count = 1
            st.recover_count = 0
            if st.candidate_count < self.min_consecutive_faint:
                return LifecycleDecision(LifecycleKind.NONE, st.state, reason="fall_candidate")
            return self._try_confirm(st, ts, prediction=prediction)

        if st.state == FallState.FALL_CANDIDATE:
            st.candidate_count += 1
            if st.candidate_count < self.min_consecutive_faint:
                return LifecycleDecision(LifecycleKind.NONE, st.state, reason="fall_candidate")
            return self._try_confirm(st, ts, prediction=prediction)

        return LifecycleDecision(LifecycleKind.SUPPRESS_NEW_FALL, st.state, reason="defensive_suppress")

    def _post_fall_alert_decision(
        self,
        st: FallTrackState,
        ts: float,
        prediction: dict | None,
        *,
        movement_level: str | None = None,
        posture_label: str | None = None,
        lying_like: bool = True,
    ) -> LifecycleDecision:
        """Still lying/alert after confirm: never NEW_FALL; maybe UNRECOVERED."""
        confirmed_at = float(st.confirmed_ts if st.confirmed_ts is not None else st.lying_since_ts or ts)
        duration = max(0.0, ts - confirmed_at)

        if duration < self.unrecovered_after_seconds:
            return LifecycleDecision(
                LifecycleKind.SUPPRESS_NEW_FALL,
                st.state,
                event_id=st.last_event_id,
                original_event_id=st.last_event_id,
                duration_sec=round(duration, 3),
                reason="post_fall_within_unrecovered_delay_suppress_new_fall_only",
            )

        # Require still unrecovered (lying / not recovered) for persistent emit
        if not lying_like and not (prediction and str(prediction.get("label", "")).lower() in {"faint", "fall", "fall_detected"}):
            return LifecycleDecision(
                LifecycleKind.SUPPRESS_NEW_FALL,
                st.state,
                event_id=st.last_event_id,
                original_event_id=st.last_event_id,
                duration_sec=round(duration, 3),
                reason="post_fall_not_lying_enough_for_unrecovered",
            )

        last_u = st.last_unrecovered_ts
        if last_u is not None:
            if self.unrecovered_repeat_seconds <= 0:
                return LifecycleDecision(
                    LifecycleKind.SUPPRESS_NEW_FALL,
                    st.state,
                    event_id=st.last_event_id,
                    original_event_id=st.last_event_id,
                    duration_sec=round(duration, 3),
                    reason="unrecovered_already_emitted_once",
                )
            if ts - float(last_u) < self.unrecovered_repeat_seconds:
                return LifecycleDecision(
                    LifecycleKind.SUPPRESS_NEW_FALL,
                    st.state,
                    event_id=st.last_event_id,
                    original_event_id=st.last_event_id,
                    duration_sec=round(duration, 3),
                    reason="unrecovered_repeat_cooldown",
                )

        st.last_unrecovered_ts = ts
        st.last_persistent_ts = ts
        # Same incident id as NEW_FALL — never mint a second uuid for unrecovered.
        event_id = st.last_event_id or str(uuid4())
        if st.last_event_id is None:
            st.last_event_id = event_id
        event_type = unrecovered_event_type_for_prediction(
            prediction,
            movement_level=movement_level,
            posture_label=posture_label,
        )
        return LifecycleDecision(
            LifecycleKind.UNRECOVERED,
            st.state,
            event_id=event_id,
            original_event_id=event_id,
            duration_sec=round(duration, 3),
            event_type=event_type,
            reason="fall_unrecovered_sustained_lying",
            faint_prob=st.confirm_faint_prob,
            consecutive_count=st.confirm_consecutive,
            prev_lifecycle_state=st.confirm_prev_state or FallState.POST_FALL_LYING.value,
            next_lifecycle_state=st.state.value if hasattr(st.state, "value") else str(st.state),
        )

    def _try_confirm(self, st: FallTrackState, ts: float, prediction: dict | None = None) -> LifecycleDecision:
        # Standing Faint false-positive guard (default on):
        # currently upright_like without an upright→lying transition → never NEW_FALL.
        if (
            self.block_upright_faint
            and st.last_posture == "upright_like"
            and not st.saw_upright_to_lying
        ):
            st.candidate_count = 0
            if st.state == FallState.FALL_CANDIDATE:
                st.state = FallState.NORMAL
                st.last_transition_ts = ts
            return LifecycleDecision(
                LifecycleKind.NONE,
                st.state,
                reason="blocked_currently_upright_without_transition",
            )

        # Optional stricter gate: already-lying without upright history.
        # - lying-only (no upright ever) → block when require_upright_to_lying
        # - upright→lying transition seen → allow
        if self.require_upright_to_lying and not st.saw_upright_to_lying:
            if not st.saw_upright and st.last_posture == "lying_like":
                return LifecycleDecision(
                    LifecycleKind.NONE,
                    st.state,
                    reason="blocked_no_upright_to_lying_transition",
                )
        prev_state = st.state.value if hasattr(st.state, "value") else str(st.state)
        confirm_prob = _confirm_faint_prob(prediction)
        confirm_consecutive = int(st.candidate_count) if st.candidate_count else self.min_consecutive_faint
        st.confirm_faint_prob = confirm_prob
        st.confirm_consecutive = confirm_consecutive
        st.confirm_prev_state = prev_state
        event_id = str(uuid4())
        st.confirmed_ts = ts
        st.last_event_id = event_id
        st.last_transition_ts = ts
        st.state = FallState.POST_FALL_LYING
        st.lying_since_ts = ts
        st.candidate_count = 0
        st.recover_count = 0
        st.last_unrecovered_ts = None
        return LifecycleDecision(
            LifecycleKind.NEW_FALL,
            st.state,
            event_id=event_id,
            reason="fall_confirmed",
            faint_prob=confirm_prob,
            consecutive_count=confirm_consecutive,
            prev_lifecycle_state=prev_state,
            next_lifecycle_state=FallState.POST_FALL_LYING.value,
        )

def _confirm_faint_prob(prediction: dict | None) -> float | None:
    """Freeze faint probability at NEW_FALL confirm time (not later overlay frames)."""
    if not prediction:
        return None
    try:
        from ai.action.faint_post_processing import faint_probability

        value = faint_probability(prediction)
        if value is not None:
            return float(value)
    except Exception:
        pass
    # Only use explicit score when present; missing score must stay None (not 0.0).
    if "score" not in prediction or prediction.get("score") is None:
        return None
    try:
        return float(prediction.get("score"))
    except (TypeError, ValueError):
        return None
