"""Configurable multi-camera supervisor restart policy (offline-safe defaults).

Process model: one OS process (overlay worker) per cameraLoginId under
`registered_camera_workers` — not N runtimes in a single process.

Defaults are conservative scaffolding for GPU-PC tuning via env vars.
Do not treat defaults as measured production optima.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """Backoff / failure caps for per-camera process restarts."""

    initial_delay_sec: float = 2.0
    maximum_delay_sec: float = 60.0
    maximum_consecutive_failures: int = 10
    stable_runtime_reset_threshold_sec: float = 120.0
    startup_stagger_sec: float = 0.5

    @classmethod
    def from_env(cls) -> "RestartPolicy":
        return cls(
            initial_delay_sec=max(0.0, _env_float("SUPERVISOR_RESTART_INITIAL_DELAY_SEC", 2.0)),
            maximum_delay_sec=max(0.0, _env_float("SUPERVISOR_RESTART_MAX_DELAY_SEC", 60.0)),
            maximum_consecutive_failures=max(1, _env_int("SUPERVISOR_MAX_CONSECUTIVE_FAILURES", 10)),
            stable_runtime_reset_threshold_sec=max(
                0.0, _env_float("SUPERVISOR_STABLE_RUNTIME_RESET_SEC", 120.0)
            ),
            startup_stagger_sec=max(0.0, _env_float("SUPERVISOR_STARTUP_STAGGER_SEC", 0.5)),
        )


@dataclass
class CameraRestartState:
    camera_login_id: str
    consecutive_failures: int = 0
    restart_count: int = 0
    last_start_mono: float | None = None
    last_exit_mono: float | None = None
    blocked: bool = False
    last_block_reason: str | None = None

    def mark_started(self, *, now_mono: float | None = None) -> None:
        self.last_start_mono = float(time.monotonic() if now_mono is None else now_mono)

    def note_stable_if_elapsed(self, policy: RestartPolicy, *, now_mono: float | None = None) -> None:
        """Clear consecutive_failures after running past stable threshold."""
        if self.last_start_mono is None:
            return
        now = float(time.monotonic() if now_mono is None else now_mono)
        if now - self.last_start_mono >= policy.stable_runtime_reset_threshold_sec:
            self.consecutive_failures = 0
            self.blocked = False
            self.last_block_reason = None

    def next_delay_sec(self, policy: RestartPolicy) -> float:
        if self.consecutive_failures <= 0:
            return float(policy.initial_delay_sec)
        # exponential: initial * 2^(failures-1), capped
        delay = float(policy.initial_delay_sec) * (2 ** max(0, self.consecutive_failures - 1))
        return min(delay, float(policy.maximum_delay_sec))

    def record_exit_and_decide(
        self,
        policy: RestartPolicy,
        *,
        now_mono: float | None = None,
    ) -> dict[str, Any]:
        """Record a worker exit; return whether restart is allowed and delay."""
        now = float(time.monotonic() if now_mono is None else now_mono)
        self.last_exit_mono = now
        # If process lived long enough, treat as stable before counting this exit
        if self.last_start_mono is not None:
            runtime = now - self.last_start_mono
            if runtime >= policy.stable_runtime_reset_threshold_sec:
                self.consecutive_failures = 0
                self.blocked = False

        self.consecutive_failures += 1
        self.restart_count += 1
        if self.consecutive_failures > policy.maximum_consecutive_failures:
            self.blocked = True
            self.last_block_reason = "max_consecutive_failures"
            return {
                "allow_restart": False,
                "delay_sec": None,
                "consecutive_failures": self.consecutive_failures,
                "restart_count": self.restart_count,
                "blocked": True,
                "reason": self.last_block_reason,
            }
        delay = self.next_delay_sec(policy)
        return {
            "allow_restart": True,
            "delay_sec": delay,
            "consecutive_failures": self.consecutive_failures,
            "restart_count": self.restart_count,
            "blocked": False,
            "reason": None,
        }


@dataclass
class SupervisorRestartBook:
    policy: RestartPolicy = field(default_factory=RestartPolicy.from_env)
    states: dict[str, CameraRestartState] = field(default_factory=dict)

    def state_for(self, camera_login_id: str) -> CameraRestartState:
        key = str(camera_login_id)
        if key not in self.states:
            self.states[key] = CameraRestartState(camera_login_id=key)
        return self.states[key]

    def on_worker_started(self, camera_login_id: str, *, now_mono: float | None = None) -> None:
        self.state_for(camera_login_id).mark_started(now_mono=now_mono)

    def on_worker_exited(self, camera_login_id: str, *, now_mono: float | None = None) -> dict[str, Any]:
        return self.state_for(camera_login_id).record_exit_and_decide(self.policy, now_mono=now_mono)

    def startup_stagger_delay(self, index: int) -> float:
        return max(0.0, float(index) * float(self.policy.startup_stagger_sec))

    def instrumentation(self, camera_login_id: str, *, pid: int | None = None) -> dict[str, Any]:
        st = self.state_for(camera_login_id)
        return {
            "camera_login_id": st.camera_login_id,
            "pid": pid,
            "restart_count": st.restart_count,
            "consecutive_failures": st.consecutive_failures,
            "blocked": st.blocked,
            "restart_blocked": st.blocked,
            "block_reason": st.last_block_reason,
        }

    def blocked_cameras(self) -> list[dict[str, Any]]:
        """External-facing list of cameras currently blocked from restart."""
        return [
            {
                "camera_login_id": st.camera_login_id,
                "restart_blocked": True,
                "blocked": True,
                "reason": st.last_block_reason,
                "consecutive_failures": st.consecutive_failures,
                "restart_count": st.restart_count,
            }
            for st in self.states.values()
            if st.blocked
        ]


class AssignmentConflictError(ValueError):
    """Raised when camera ID / port / output path assignments collide."""


@dataclass(frozen=True, slots=True)
class CameraAssignment:
    camera_login_id: str
    source: str | None = None
    overlay_port: int | None = None
    output_path: str | None = None


def validate_camera_assignments(assignments: Sequence[CameraAssignment]) -> None:
    """Fail-fast on duplicate camera ID, port, or output path (dynamic sets OK)."""
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    seen_paths: set[str] = set()
    for item in assignments:
        cam = str(item.camera_login_id or "").strip()
        if not cam:
            raise AssignmentConflictError("empty camera_login_id")
        if cam in seen_ids:
            raise AssignmentConflictError(f"duplicate camera_login_id={cam}")
        seen_ids.add(cam)
        if item.overlay_port is not None:
            port = int(item.overlay_port)
            if port in seen_ports:
                raise AssignmentConflictError(f"duplicate overlay_port={port}")
            seen_ports.add(port)
        if item.output_path:
            path = str(item.output_path).replace("\\", "/").rstrip("/")
            if path in seen_paths:
                raise AssignmentConflictError(f"duplicate output_path={path}")
            seen_paths.add(path)


def plan_source_change_restarts(
    current_signatures: Mapping[str, str],
    desired_signatures: Mapping[str, str],
) -> dict[str, str]:
    """Return camera_login_id -> reason for workers that must restart.

    Only cameras whose signature changed (or are new) are included.
    Unchanged cameras are omitted (must not restart).
    """
    plan: dict[str, str] = {}
    for cam, sig in desired_signatures.items():
        prev = current_signatures.get(cam)
        if prev is None:
            plan[str(cam)] = "new_camera"
        elif prev != sig:
            plan[str(cam)] = "source_changed"
    for cam in current_signatures:
        if cam not in desired_signatures:
            plan[str(cam)] = "removed"
    return plan
