from __future__ import annotations

from dataclasses import dataclass

from ai.action.fall_event_state import FallEventStateMachine, LifecycleDecision, LifecycleKind

DEFAULT_FAINT_THRESHOLD = 0.6
DEFAULT_MIN_CONSECUTIVE_FAINT = 2
DEFAULT_CAMERA_COOLDOWN_SECONDS = 10.0
DEFAULT_RECOVER_CONSECUTIVE = 4
DEFAULT_PERSISTENT_DELAY_SEC = 10.0
DEFAULT_PERSISTENT_REPEAT_SEC = 30.0
DEFAULT_REQUIRE_UPRIGHT_TO_LYING = False
DEFAULT_BLOCK_UPRIGHT_FAINT = True
DEFAULT_ACTION_MODEL = (
    "benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/"
    "YOLO26n-pose=./yolo26n-pose.pt/best.pt"
)


MEMO_NEW_FALL = "쓰러짐 의심!"
MEMO_UNRECOVERED = "낙상 후 미회복/실신 의심"
MEMO_UNRECOVERED_LYING = "낙상 후 계속 누워 있음"


def movement_level_from_delta(center_y_delta: float | None, low_threshold: float = 12.0) -> str:
    """Coarse motion class for unrecovered risk context."""
    if center_y_delta is None:
        return "unknown"
    mag = abs(float(center_y_delta))
    if mag < 2.0:
        return "still"
    if mag < float(low_threshold):
        return "low"
    return "high"


@dataclass
class AlertEmitDecision:
    """Result of evaluate(): what (if anything) to publish."""

    emit: bool
    kind: str  # new_fall | unrecovered | none
    event_type: str | None = None
    event_id: str | None = None
    original_event_id: str | None = None
    duration_sec: float | None = None
    lifecycle: LifecycleDecision | None = None
    memo_text: str | None = None
    camera_login_id: str | None = None
    track_id: int | str | None = None
    posture_label: str | None = None
    movement_level: str | None = None
    state: str | None = None

    @property
    def is_new_fall(self) -> bool:
        return self.kind == "new_fall"

    @property
    def is_unrecovered(self) -> bool:
        return self.kind == "unrecovered"


class FaintEventPostProcessor:
    """LSTM 실신 후처리: 연속 감지 + 카메라 cooldown + track 상태머신.

    - NEW_FALL: 최초 확정 1회 (cooldown 보조)
    - POST_FALL_LYING: NEW_FALL 재발행 금지
    - cooldown(기본 10s) 이후에도 누워/Faint 유지 시 FAINT_SUSPECTED / FALL_UNRECOVERED 발행
    """

    def __init__(
        self,
        min_consecutive_faint=DEFAULT_MIN_CONSECUTIVE_FAINT,
        cooldown_seconds=DEFAULT_CAMERA_COOLDOWN_SECONDS,
        *,
        use_fall_state_machine: bool = True,
        recover_consecutive: int = DEFAULT_RECOVER_CONSECUTIVE,
        require_upright_to_lying: bool = DEFAULT_REQUIRE_UPRIGHT_TO_LYING,
        block_upright_faint: bool = DEFAULT_BLOCK_UPRIGHT_FAINT,
        unrecovered_after_seconds: float | None = None,
        unrecovered_repeat_seconds: float = DEFAULT_PERSISTENT_REPEAT_SEC,
        use_posture_estimator: bool = True,
        lying_aspect_ratio: float = 1.2,
        upright_aspect_ratio: float = 1.3,
        min_keypoint_conf: float = 0.3,
        lying_frames_required: int = 2,
        upright_frames_required: int = 2,
        movement_low_threshold: float = 12.0,
        faint_threshold: float | None = None,
        fall_threshold: float | None = None,
        track_lost_grace_sec: float = 3.0,
    ):
        self.min_consecutive_faint = max(1, int(min_consecutive_faint))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.use_fall_state_machine = bool(use_fall_state_machine)
        self.use_posture_estimator = bool(use_posture_estimator)
        self.require_upright_to_lying = bool(require_upright_to_lying)
        self.block_upright_faint = bool(block_upright_faint)
        self.movement_low_threshold = float(movement_low_threshold)
        self.faint_threshold = float(faint_threshold) if faint_threshold is not None else DEFAULT_FAINT_THRESHOLD
        self.fall_threshold = float(fall_threshold) if fall_threshold is not None else self.faint_threshold
        self.track_lost_grace_sec = max(0.0, float(track_lost_grace_sec))
        unrecovered_after = (
            float(DEFAULT_PERSISTENT_DELAY_SEC)
            if unrecovered_after_seconds is None
            else float(unrecovered_after_seconds)
        )
        self._consecutive_by_camera = {}
        self._last_event_time_by_camera = {}
        self._last_seen_ts: dict[str, float] = {}
        self._last_lifecycle_decision = None
        self._last_emit_decision: AlertEmitDecision | None = None
        self._posture_estimator = None
        if self.use_posture_estimator:
            from ai.action.posture_estimator import PostureEstimator

            self._posture_estimator = PostureEstimator(
                aspect_lying_wh=lying_aspect_ratio,
                aspect_upright_hw=upright_aspect_ratio,
                conf_min=min_keypoint_conf,
                lying_frames_required=lying_frames_required,
                upright_frames_required=upright_frames_required,
                movement_low_threshold=movement_low_threshold,
            )
        self._state_machine = (
            FallEventStateMachine(
                min_consecutive_faint=self.min_consecutive_faint,
                recover_consecutive=max(1, int(recover_consecutive)),
                require_upright_to_lying=bool(require_upright_to_lying),
                block_upright_faint=bool(block_upright_faint),
                unrecovered_after_seconds=unrecovered_after,
                unrecovered_repeat_seconds=max(0.0, float(unrecovered_repeat_seconds)),
            )
            if self.use_fall_state_machine
            else None
        )

    def reset(self) -> None:
        self._consecutive_by_camera.clear()
        self._last_event_time_by_camera.clear()
        self._last_seen_ts.clear()
        self._last_lifecycle_decision = None
        self._last_emit_decision = None
        if self._state_machine is not None:
            self._state_machine.reset_all()
        if self._posture_estimator is not None:
            self._posture_estimator.reset_all()

    def note_track_seen(self, camera_id, track_id, timestamp: float) -> None:
        """Record last-seen time for track-lost grace handling."""
        key = event_state_key(camera_id, track_id)
        self._last_seen_ts[key] = float(timestamp)

    def prune_lost_tracks(self, camera_id, active_track_ids, timestamp: float) -> list[str]:
        """Drop lifecycle state for tracks missing longer than track_lost_grace_sec.

        Returns list of pruned track keys.
        """
        if self.track_lost_grace_sec <= 0:
            return []
        active = {event_state_key(camera_id, tid) for tid in (active_track_ids or [])}
        prefix = f"{camera_id}:track:"
        pruned: list[str] = []
        ts = float(timestamp)
        for key, seen_at in list(self._last_seen_ts.items()):
            if not key.startswith(prefix) and key != str(camera_id):
                continue
            if key in active:
                continue
            if ts - float(seen_at) < self.track_lost_grace_sec:
                continue
            pruned.append(key)
            self._last_seen_ts.pop(key, None)
            self._consecutive_by_camera.pop(key, None)
            # parse track id from key "cam:track:7"
            track_id = None
            if ":track:" in key:
                try:
                    track_id = int(float(key.rsplit(":track:", 1)[1]))
                except (TypeError, ValueError):
                    track_id = key.rsplit(":track:", 1)[-1]
            if self._state_machine is not None:
                self._state_machine.reset_track(camera_id, track_id)
            if self._posture_estimator is not None:
                self._posture_estimator.reset_track(key)
        return pruned

    def last_lifecycle_decision(self):
        return self._last_lifecycle_decision

    def last_emit_decision(self) -> AlertEmitDecision | None:
        return self._last_emit_decision

    def evaluate(
        self,
        camera_id,
        prediction,
        timestamp,
        track_id=None,
        *,
        posture_label=None,
        upright_to_lying=None,
        detection=None,
    ) -> AlertEmitDecision:
        """Full emit decision (new fall or unrecovered). Prefer this over should_trigger.

        If ``detection`` is provided and posture estimator is enabled, posture_label /
        upright_to_lying are derived automatically (Phase B).
        """
        key = event_state_key(camera_id, track_id)
        cooldown_key = event_cooldown_key(camera_id)
        is_alert = is_alert_prediction(
            prediction,
            faint_threshold=self.faint_threshold,
            fall_threshold=self.fall_threshold,
        )
        movement_level = "unknown"
        estimate = None
        self.note_track_seen(camera_id, track_id, timestamp)

        if detection is not None and self._posture_estimator is not None:
            from ai.action.posture_estimator import track_posture_key

            estimate = self._posture_estimator.estimate(
                track_posture_key(str(camera_id), track_id),
                detection,
                timestamp=float(timestamp),
            )
            if posture_label is None:
                posture_label = estimate.label
            if upright_to_lying is None:
                upright_to_lying = estimate.upright_to_lying_transition
            movement_level = movement_level_from_delta(
                estimate.center_y_delta,
                low_threshold=self.movement_low_threshold,
            )

        # Standing Faint FP: do not feed alert into consecutive counters / lifecycle confirm.
        standing_blocked = bool(
            self.block_upright_faint
            and is_alert
            and posture_label == "upright_like"
            and not upright_to_lying
        )
        effective_alert = bool(is_alert) and not standing_blocked

        def _context(**kwargs) -> AlertEmitDecision:
            state = None
            if kwargs.get("lifecycle") is not None:
                state = kwargs["lifecycle"].state.value if hasattr(kwargs["lifecycle"].state, "value") else str(kwargs["lifecycle"].state)
            base = dict(
                camera_login_id=str(camera_id),
                track_id=track_id,
                posture_label=posture_label,
                movement_level=movement_level,
                state=state,
            )
            base.update(kwargs)
            return AlertEmitDecision(**base)

        if self._state_machine is not None:
            decision = self._state_machine.update(
                camera_id,
                timestamp,
                track_id=track_id,
                is_alert=effective_alert,
                posture_label=posture_label,
                upright_to_lying=upright_to_lying,
                prediction=prediction if isinstance(prediction, dict) else None,
                movement_level=movement_level,
                lying_like=(posture_label == "lying_like") if posture_label else None,
            )
            self._last_lifecycle_decision = decision

            if standing_blocked:
                # Blocked upright Faint must not accumulate consecutive counts.
                self._consecutive_by_camera[key] = 0
                blocked_decision = LifecycleDecision(
                    LifecycleKind.NONE,
                    decision.state,
                    reason="blocked_currently_upright_without_transition",
                )
                self._last_lifecycle_decision = blocked_decision
                out = _context(
                    emit=False,
                    kind="none",
                    lifecycle=blocked_decision,
                    memo_text="blocked_currently_upright_without_transition",
                )
                self._last_emit_decision = out
                _maybe_log_faint_diagnostic(
                    camera_id=camera_id,
                    track_id=track_id,
                    prediction=prediction,
                    posture_label=posture_label,
                    upright_to_lying=bool(upright_to_lying),
                    standing_blocked=True,
                    lifecycle=blocked_decision,
                    event_emit=False,
                    sequence=detection if isinstance(detection, dict) else None,
                )
                return out

            if not effective_alert:
                self._consecutive_by_camera[key] = 0
                out = _context(emit=False, kind="none", lifecycle=decision)
                self._last_emit_decision = out
                return out

            consecutive = int(self._consecutive_by_camera.get(key, 0)) + 1
            self._consecutive_by_camera[key] = consecutive

            if decision.kind == LifecycleKind.NEW_FALL:
                last_event_time = self._last_event_time_by_camera.get(cooldown_key)
                if last_event_time is not None and float(timestamp) - float(last_event_time) < self.cooldown_seconds:
                    self._state_machine.revert_confirm_to_candidate(camera_id, track_id)
                    out = _context(
                        emit=False,
                        kind="none",
                        lifecycle=decision,
                        memo_text="new_fall_blocked_by_camera_cooldown",
                    )
                    self._last_emit_decision = out
                    return out
                self._last_event_time_by_camera[cooldown_key] = float(timestamp)
                out = _context(
                    emit=True,
                    kind="new_fall",
                    event_type=None,  # keep prediction-based faint/fall
                    event_id=decision.event_id,
                    lifecycle=decision,
                    memo_text=MEMO_NEW_FALL,
                )
                self._last_emit_decision = out
                _maybe_log_faint_diagnostic(
                    camera_id=camera_id,
                    track_id=track_id,
                    prediction=prediction,
                    posture_label=posture_label,
                    upright_to_lying=bool(upright_to_lying),
                    standing_blocked=False,
                    lifecycle=decision,
                    event_emit=True,
                    sequence=detection if isinstance(detection, dict) else None,
                )
                return out

            if decision.kind == LifecycleKind.UNRECOVERED:
                # Prefer lying-specific copy when posture says lying_like.
                memo = MEMO_UNRECOVERED_LYING if posture_label == "lying_like" else MEMO_UNRECOVERED
                out = _context(
                    emit=True,
                    kind="unrecovered",
                    event_type=decision.event_type,
                    event_id=decision.event_id,
                    original_event_id=decision.original_event_id,
                    duration_sec=decision.duration_sec,
                    lifecycle=decision,
                    memo_text=memo,
                )
                self._last_emit_decision = out
                return out

            # Faint prediction still active (candidate path) — optional diagnostic
            if is_alert:
                _maybe_log_faint_diagnostic(
                    camera_id=camera_id,
                    track_id=track_id,
                    prediction=prediction,
                    posture_label=posture_label,
                    upright_to_lying=bool(upright_to_lying),
                    standing_blocked=False,
                    lifecycle=decision,
                    event_emit=False,
                    sequence=detection if isinstance(detection, dict) else None,
                )
            out = _context(emit=False, kind="none", lifecycle=decision)
            self._last_emit_decision = out
            return out

        # Legacy path
        if not is_alert:
            self._consecutive_by_camera[key] = 0
            out = AlertEmitDecision(emit=False, kind="none")
            self._last_emit_decision = out
            return out

        consecutive = int(self._consecutive_by_camera.get(key, 0)) + 1
        self._consecutive_by_camera[key] = consecutive
        if consecutive < self.min_consecutive_faint:
            out = AlertEmitDecision(emit=False, kind="none")
            self._last_emit_decision = out
            return out
        last_event_time = self._last_event_time_by_camera.get(cooldown_key)
        if last_event_time is not None and float(timestamp) - float(last_event_time) < self.cooldown_seconds:
            out = AlertEmitDecision(emit=False, kind="none")
            self._last_emit_decision = out
            return out
        self._last_event_time_by_camera[cooldown_key] = float(timestamp)
        out = AlertEmitDecision(emit=True, kind="new_fall", memo_text="쓰러짐 의심!")
        self._last_emit_decision = out
        return out

    def should_trigger(
        self,
        camera_id,
        prediction,
        timestamp,
        track_id=None,
        *,
        posture_label=None,
        upright_to_lying=None,
        detection=None,
    ):
        """Backward-compatible: True only for NEW_FALL publish.

        Unrecovered events are available via :meth:`evaluate`.
        """
        decision = self.evaluate(
            camera_id,
            prediction,
            timestamp,
            track_id=track_id,
            posture_label=posture_label,
            upright_to_lying=upright_to_lying,
            detection=detection,
        )
        return bool(decision.emit and decision.is_new_fall)

    def consecutive_count(self, camera_id, track_id=None):
        return int(self._consecutive_by_camera.get(event_state_key(camera_id, track_id), 0))

    def cooldown_active(self, camera_id, timestamp, track_id=None):
        key = event_cooldown_key(camera_id)
        last_event_time = self._last_event_time_by_camera.get(key)
        if last_event_time is None:
            return False
        return float(timestamp) - float(last_event_time) < self.cooldown_seconds

    def migrate_track(self, camera_id, old_track_id, new_track_id) -> bool:
        """Move Fall/Faint lifecycle maps from old_track_id to new_track_id (recovery)."""
        if old_track_id is None or new_track_id is None:
            return False
        if str(old_track_id) == str(new_track_id):
            return False
        old_key = event_state_key(camera_id, old_track_id)
        new_key = event_state_key(camera_id, new_track_id)
        moved = False
        if old_key in self._consecutive_by_camera:
            if new_key not in self._consecutive_by_camera:
                self._consecutive_by_camera[new_key] = self._consecutive_by_camera.pop(old_key)
            else:
                self._consecutive_by_camera.pop(old_key, None)
            moved = True
        if old_key in self._last_seen_ts:
            if new_key not in self._last_seen_ts:
                self._last_seen_ts[new_key] = self._last_seen_ts.pop(old_key)
            else:
                self._last_seen_ts.pop(old_key, None)
            moved = True
        if self._state_machine is not None:
            moved = self._state_machine.migrate_track(camera_id, old_track_id, new_track_id) or moved
        if self._posture_estimator is not None:
            hist = getattr(self._posture_estimator, "_history", None)
            cy = getattr(self._posture_estimator, "_center_y", None)
            if isinstance(hist, dict) and old_key in hist:
                if new_key not in hist:
                    hist[new_key] = hist.pop(old_key)
                else:
                    hist.pop(old_key, None)
                moved = True
            if isinstance(cy, dict) and old_key in cy:
                if new_key not in cy:
                    cy[new_key] = cy.pop(old_key)
                else:
                    cy.pop(old_key, None)
                moved = True
        return moved


def _maybe_log_faint_diagnostic(
    *,
    camera_id,
    track_id,
    prediction,
    posture_label,
    upright_to_lying: bool,
    standing_blocked: bool,
    lifecycle,
    event_emit: bool,
    sequence=None,
) -> None:
    """Structured sparse log for Faint FP diagnosis (not per-frame)."""
    pred = prediction if isinstance(prediction, dict) else {}
    label = pred.get("label")
    probs = pred.get("probabilities") if isinstance(pred.get("probabilities"), dict) else {}
    faint_p = probs.get("Faint")
    normal_p = probs.get("Normal")
    if faint_p is None:
        try:
            faint_p = float(pred.get("score")) if pred.get("score") is not None else None
        except (TypeError, ValueError):
            faint_p = None
    motion = {}
    if isinstance(sequence, dict):
        motion = sequence.get("motion_validity") or {}
    reason = getattr(lifecycle, "reason", None) if lifecycle is not None else None
    state = None
    if lifecycle is not None and getattr(lifecycle, "state", None) is not None:
        st = lifecycle.state
        state = st.value if hasattr(st, "value") else str(st)
    motion_ratio = motion.get("motion_valid_ratio")
    low_conf_frames = motion.get("low_conf_motion_frames")
    should_log = bool(
        standing_blocked
        or event_emit
        or str(label).lower() == "faint"
        or (motion_ratio is not None and float(motion_ratio) < 0.5)
        or (low_conf_frames is not None and int(low_conf_frames) > 0 and str(label).lower() == "faint")
    )
    if not should_log:
        return
    print(
        "[faint-diagnostic] "
        f"cameraLoginId={camera_id} "
        f"trackId={track_id} "
        f"tensor_shape={pred.get('tensor_shape') or ''} "
        f"feature_schema={pred.get('feature_schema') or ''} "
        f"sequence_length={pred.get('sequence_length') or ''} "
        f"sequence_stride={pred.get('sequence_stride') or ''} "
        f"faint_probability={faint_p} "
        f"normal_probability={normal_p} "
        f"predicted_label={label} "
        f"posture_label={posture_label} "
        f"upright_to_lying={str(bool(upright_to_lying)).lower()} "
        f"standing_blocked={str(bool(standing_blocked)).lower()} "
        f"motion_valid_ratio={motion.get('motion_valid_ratio', '')} "
        f"low_conf_motion_frames={motion.get('low_conf_motion_frames', '')} "
        f"center_drop_min={motion.get('center_drop_min', '')} "
        f"center_drop_max={motion.get('center_drop_max', '')} "
        f"velocity_min={motion.get('velocity_min', '')} "
        f"velocity_max={motion.get('velocity_max', '')} "
        f"torso_angle_min={motion.get('torso_angle_min', '')} "
        f"torso_angle_max={motion.get('torso_angle_max', '')} "
        f"lifecycle_state={state or ''} "
        f"lifecycle_reason={reason or ''} "
        f"event_emit={str(bool(event_emit)).lower()}",
        flush=True,
    )


def event_cooldown_key(camera_id):
    return str(camera_id)


def event_state_key(camera_id, track_id=None):
    return f"{camera_id}:track:{track_id}" if track_id is not None else str(camera_id)


def is_alert_prediction(prediction, faint_threshold=None, fall_threshold=None):
    """True when prediction is an alert class, optionally gated by score thresholds.

    If the prediction has no score/probability fields, label-only alerts still fire
    (backward compatible with unit tests and sparse mocks).
    """
    if not prediction:
        return False
    label = prediction.get("label")
    if label is None or label == "Normal":
        return False
    label_l = str(label).strip().lower()
    has_prob = "probabilities" in prediction and prediction.get("probabilities")
    has_score = prediction.get("score") is not None
    if not has_prob and not has_score:
        return True
    score = faint_probability(prediction)
    if score is None:
        try:
            score = float(prediction.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
    if label_l == "faint" and faint_threshold is not None:
        return float(score) >= float(faint_threshold)
    if label_l in {"fall", "fall_detected"} and fall_threshold is not None:
        return float(score) >= float(fall_threshold)
    if faint_threshold is not None and label_l not in {"fall", "fall_detected"}:
        return float(score) >= float(faint_threshold)
    return True


def faint_probability(prediction):
    if not prediction:
        return None
    probabilities = prediction.get("probabilities") or {}
    if "Faint" in probabilities:
        return float(probabilities["Faint"])
    if prediction.get("label") == "Faint":
        return float(prediction.get("score", 0.0))
    return None


DEFAULT_EXIT_MIN_CONSECUTIVE = 2
DEFAULT_EXIT_COOLDOWN_SECONDS = 60.0


class ExitEventPostProcessor:
    def __init__(self, min_consecutive=DEFAULT_EXIT_MIN_CONSECUTIVE, cooldown_seconds=DEFAULT_EXIT_COOLDOWN_SECONDS):
        self.min_consecutive = max(1, int(min_consecutive))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_by_track = {}
        self._last_event_time = {}
        self._inside_by_track = {}

    def observe(self, camera_id, track_id, *, inside_safe_zone, timestamp):
        """Emit only after a tracked person crosses from inside to outside and stays outside for min_consecutive frames."""
        key = f"{camera_id}:track:{track_id}"
        
        if not inside_safe_zone:
            consecutive = self._consecutive_by_track.get(key, 0) + 1
            self._consecutive_by_track[key] = consecutive
            self._inside_by_track[key] = False
            
            if consecutive >= self.min_consecutive:
                cooldown_key = f"{camera_id}:track:{track_id}"
                last = self._last_event_time.get(cooldown_key)
                if last is None or (float(timestamp) - last >= self.cooldown_seconds):
                    self._last_event_time[cooldown_key] = float(timestamp)
                    return True
        else:
            self._consecutive_by_track[key] = 0
            self._inside_by_track[key] = True
            
        return False
    def should_trigger(self, camera_id, track_id, timestamp):
        key = f"{camera_id}:track:{track_id}"
        consecutive = self._consecutive_by_track.get(key, 0) + 1
        self._consecutive_by_track[key] = consecutive
        if consecutive < self.min_consecutive:
            return False
        cooldown_key = str(camera_id)
        last = self._last_event_time.get(cooldown_key)
        if last is not None and float(timestamp) - last < self.cooldown_seconds:
            return False
        self._last_event_time[cooldown_key] = float(timestamp)
        return True

    def reset_track(self, camera_id, track_id):
        self._consecutive_by_track[f"{camera_id}:track:{track_id}"] = 0


DEFAULT_HAZARD_MIN_CONSECUTIVE = 2
DEFAULT_HAZARD_COOLDOWN_SECONDS = 60.0


class HazardEventPostProcessor:
    def __init__(self, min_consecutive=DEFAULT_HAZARD_MIN_CONSECUTIVE, cooldown_seconds=DEFAULT_HAZARD_COOLDOWN_SECONDS):
        self.min_consecutive = max(1, int(min_consecutive))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_by_track = {}
        self._last_event_time = {}
        self._inside_by_track = {}

    def observe(self, camera_id, track_id, *, inside_safe_zone, timestamp):
        """Emit only after a tracked person crosses from inside to outside."""
        key = f"{camera_id}:track:{track_id}"
        if inside_safe_zone:
            self._inside_by_track[key] = True
            self.reset_track(camera_id, track_id)
            return False
        if not self._inside_by_track.get(key, False):
            return False
        return self.should_trigger(camera_id, track_id, timestamp)
    def should_trigger(self, camera_id, track_id, timestamp):
        key = f"{camera_id}:track:{track_id}"
        consecutive = self._consecutive_by_track.get(key, 0) + 1
        self._consecutive_by_track[key] = consecutive
        if consecutive < self.min_consecutive:
            return False
        cooldown_key = str(camera_id)
        last = self._last_event_time.get(cooldown_key)
        if last is not None and float(timestamp) - last < self.cooldown_seconds:
            return False
        self._last_event_time[cooldown_key] = float(timestamp)
        return True

    def reset_track(self, camera_id, track_id):
        self._consecutive_by_track[f"{camera_id}:track:{track_id}"] = 0
