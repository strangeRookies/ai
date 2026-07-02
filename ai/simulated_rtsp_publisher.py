from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


FFMPEG_MODE_CHOICES: Final = ("auto", "copy", "cpu", "libx264", "nvenc")
NVENC_FAILURE_HINTS: Final = (
    "h264_nvenc",
    "No NVENC capable devices found",
    "Cannot load libcuda",
    "OpenEncodeSessionEx failed",
    "Device busy",
)
PUBLISH_FAILURE_HINTS: Final = (
    "No such file",
    "Connection refused",
    "already publishing",
    "Broken pipe",
    "Could not write header",
    "Server returned 4XX",
)


def normalize_ffmpeg_mode(ffmpeg_mode: str) -> str:
    mode = ffmpeg_mode.lower()
    if mode == "libx264":
        return "cpu"
    if mode not in FFMPEG_MODE_CHOICES:
        return "auto"
    return mode


def build_ffmpeg_cmd(video_path: Path, rtsp_url: str, loop: bool, ffmpeg_mode: str = "cpu") -> list[str]:
    cmd = ["ffmpeg", "-re"]
    if loop:
        cmd.extend(["-stream_loop", "-1"])
    cmd.extend(["-i", str(video_path), "-an"])

    mode = normalize_ffmpeg_mode(ffmpeg_mode)
    if mode == "auto":
        mode = "copy"
    if mode == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(
            [
                "-vf",
                "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=15",
                "-c:v",
                "h264_nvenc" if mode == "nvenc" else "libx264",
                "-preset",
                "p1" if mode == "nvenc" else "ultrafast",
                "-tune",
                "ull" if mode == "nvenc" else "zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "30",
                "-keyint_min",
                "30",
                "-sc_threshold",
                "0",
                "-b:v",
                "1500k",
                "-maxrate",
                "1800k",
                "-bufsize",
                "3000k",
            ]
        )
    cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", rtsp_url])
    return cmd


def h264_nvenc_available(ffmpeg_binary: str = "ffmpeg") -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "h264_nvenc" in result.stdout


def nvidia_smi_available(nvidia_smi_binary: str = "nvidia-smi") -> bool:
    try:
        result = subprocess.run(
            [nvidia_smi_binary],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def select_initial_ffmpeg_mode(requested_mode: str, nvenc_available: bool) -> str:
    mode = normalize_ffmpeg_mode(requested_mode)
    match mode:
        case "auto":
            return "nvenc" if nvenc_available else "copy"
        case "libx264":
            return "cpu"
        case "copy" | "cpu" | "nvenc":
            return mode
        case unreachable:
            raise AssertionError(f"Unhandled ffmpeg mode: {unreachable}")


@dataclass(slots=True)
class FfmpegRestartPolicy:
    requested_mode: str
    initial_mode: str
    max_restarts_per_mode: int = 2
    fallback_enabled: bool = True
    restart_count: int = 0
    mode_restart_count: int = 0
    active_mode: str = field(init=False)

    def __post_init__(self) -> None:
        self.active_mode = normalize_ffmpeg_mode(self.initial_mode)

    def record_exit(self, exit_code: int, stderr_tail: str = "") -> str:
        self.restart_count += 1
        self.mode_restart_count += 1
        if not self.fallback_enabled:
            return self.active_mode
        nvenc_hint_seen = any(hint in stderr_tail for hint in NVENC_FAILURE_HINTS)
        if self.active_mode == "nvenc" and (
            exit_code == 255 or nvenc_hint_seen or self.mode_restart_count > self.max_restarts_per_mode
        ):
            return self._switch_mode("copy")
        if self.active_mode == "copy" and self.mode_restart_count > self.max_restarts_per_mode:
            return self._switch_mode("cpu")
        return self.active_mode

    def _switch_mode(self, next_mode: str) -> str:
        self.active_mode = normalize_ffmpeg_mode(next_mode)
        self.mode_restart_count = 0
        return self.active_mode


@dataclass(slots=True)
class PublisherLock:
    path: Path
    acquired: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_lock()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"simulated RTSP publisher is already locked by {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current_pid = int(self.path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, OSError, ValueError):
            self.acquired = False
            return
        if current_pid == os.getpid():
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def _remove_stale_lock(self) -> None:
        try:
            existing_pid = int(self.path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            self.path.unlink(missing_ok=True)
            return

        from ai.worker_registry import check_pid_alive, check_process_signature_matches

        if check_pid_alive(existing_pid) and check_process_signature_matches(
            existing_pid,
            ["start_simulated_rtsp_from_folder.py"],
        ):
            raise RuntimeError(f"simulated RTSP publisher already running with PID {existing_pid}")
        self.path.unlink(missing_ok=True)


def tail_text_file(path: Path, max_lines: int = 50) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-max_lines:])


def summarize_ffmpeg_exit(camera_login_id: str, exit_code: int, log_path: Path) -> str:
    tail = tail_text_file(log_path)
    hints = [hint for hint in (*NVENC_FAILURE_HINTS, *PUBLISH_FAILURE_HINTS) if hint in tail]
    hint_text = ", ".join(hints) if hints else "no known failure hint detected"
    return (
        f"[simulated-rtsp][ffmpeg-exit] camera={camera_login_id} exit_code={exit_code} "
        f"log={log_path} hints={hint_text}\n"
        f"----- last ffmpeg log lines -----\n{tail}\n"
        f"---------------------------------"
    )
