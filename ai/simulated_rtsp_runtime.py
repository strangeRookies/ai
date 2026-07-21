from __future__ import annotations

import argparse
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ai.registered_cameras import camera_rtsp_url, load_active_cameras
from ai.simulated_rtsp_publisher import (
    FfmpegRestartPolicy,
    PublisherLock,
    build_ffmpeg_cmd,
    h264_nvenc_available,
    nvidia_smi_available,
    select_initial_ffmpeg_mode,
    summarize_ffmpeg_exit,
    tail_text_file,
)
from ai.simulated_rtsp_sources import (
    filter_stream_video_pools,
    scan_video_directory,
)


ASSIGNMENT_STATE_PATH = Path("runs/simulated_rtsp/video_assignments.json")
VideoFingerprint = tuple[str, int, int]


def video_pool_for_camera_position(
    camera_position: int,
    outdoor_files: list[Path],
    chromakey_files: list[Path],
) -> tuple[list[Path], bool]:
    # Pair cameras in sorted order: 1st/3rd/... use outdoor footage and
    # 2nd/4th/... use chromakey footage. This keeps cam_03/cam_04 split even
    # when cam_01/cam_02 are also active in the backend.
    if camera_position % 2 == 1:
        return chromakey_files, True
    return outdoor_files, False


def scan_stream_video_directories(video_dir: str, chromakey_video_dir: str | None) -> list[Path]:
    video_files = scan_video_directory(video_dir)
    if chromakey_video_dir:
        video_files.extend(scan_video_directory(chromakey_video_dir))

    # The chromakey directory may be nested under --video-dir. Keep one entry
    # per physical path so rotating camera assignment is not skewed by duplicates.
    unique_files = {video_path.resolve(): video_path for video_path in video_files}
    return sorted(unique_files.values(), key=lambda video_path: str(video_path).lower())


def video_file_fingerprint(video_path: Path) -> VideoFingerprint:
    stat = video_path.stat()
    return str(video_path.resolve()), stat.st_size, stat.st_mtime_ns


def video_pool_fingerprint(video_files: list[Path]) -> tuple[VideoFingerprint, ...]:
    return tuple(sorted(video_file_fingerprint(video_path) for video_path in video_files))


def load_video_assignments(state_path: Path) -> dict[str, Path]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(camera_login_id): Path(video_path)
        for camera_login_id, video_path in payload.items()
        if isinstance(camera_login_id, str) and isinstance(video_path, str) and video_path
    }


def save_video_assignments(state_path: Path, assignments: dict[str, Path]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    payload = {camera_login_id: str(video_path.resolve()) for camera_login_id, video_path in assignments.items()}
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(state_path)


def select_rotating_video(
    video_files: list[Path],
    previous_video: Path | None,
    used_videos: set[Path],
    rng: random.Random | random.SystemRandom | None = None,
) -> Path:
    if not video_files:
        raise ValueError("video_files must not be empty")

    chooser = rng or random.SystemRandom()
    previous_resolved = previous_video.resolve() if previous_video is not None else None
    resolved_used = {video_path.resolve() for video_path in used_videos}

    candidates = [
        video_path
        for video_path in video_files
        if video_path.resolve() != previous_resolved and video_path.resolve() not in resolved_used
    ]
    if not candidates:
        candidates = [video_path for video_path in video_files if video_path.resolve() != previous_resolved]
    if not candidates:
        candidates = [video_path for video_path in video_files if video_path.resolve() not in resolved_used]
    if not candidates:
        candidates = list(video_files)
    return chooser.choice(candidates)


def stream_config_changed(
    stream_info: dict[str, Any],
    assigned_video: Path,
    assigned_fingerprint: VideoFingerprint,
    target_rtsp_url: str,
) -> bool:
    return (
        stream_info["video_path"] != assigned_video
        or stream_info["video_fingerprint"] != assigned_fingerprint
        or stream_info["rtsp_url"] != target_rtsp_url
    )


def run_simulated_rtsp_publisher(args: argparse.Namespace, repo_root: Path) -> None:
    video_dir = args.video_dir
    chromakey_video_dir = getattr(args, "chromakey_video_dir", None)
    backend_url = args.backend_url
    rtsp_base_url = f"rtsp://{args.rtsp_host}:{args.rtsp_port}"
    poll_interval = args.poll_interval
    loop_playback = args.loop
    nvenc_encoder_supported = h264_nvenc_available()
    nvidia_smi_supported = nvidia_smi_available()
    nvenc_supported = nvenc_encoder_supported and nvidia_smi_supported
    initial_ffmpeg_mode = select_initial_ffmpeg_mode(args.ffmpeg_mode, nvenc_supported)
    lock = PublisherLock(Path("runs/simulated_rtsp/start_simulated_rtsp_from_folder.lock"))

    print("==================================================", flush=True)
    print("Starting simulated RTSP stream publisher", flush=True)
    print(f"Video Directory: {video_dir}", flush=True)
    print(f"Chromakey Directory: {chromakey_video_dir or '(from video directory)'}", flush=True)
    print(f"Backend URL:     {backend_url}", flush=True)
    print(f"RTSP Base URL:   {rtsp_base_url}", flush=True)
    print(f"Poll Interval:   {poll_interval}s", flush=True)
    print(f"Loop Playback:   {loop_playback}", flush=True)
    print(f"FFmpeg Mode:     requested={args.ffmpeg_mode}, active={initial_ffmpeg_mode}", flush=True)
    if args.ffmpeg_mode in {"auto", "nvenc"}:
        print(
            f"[NVENC CHECK] h264_nvenc={'available' if nvenc_encoder_supported else 'unavailable'}, "
            f"nvidia-smi={'available' if nvidia_smi_supported else 'unavailable'}",
            flush=True,
        )
    if args.ffmpeg_mode == "auto" and initial_ffmpeg_mode != "nvenc":
        print("[NVENC CHECK] h264_nvenc unavailable. Falling back to copy/libx264.", flush=True)
    print("==================================================", flush=True)

    all_scanned_files = scan_stream_video_directories(video_dir, chromakey_video_dir)
    if not all_scanned_files:
        print(f"[simulated-rtsp][error] No video files (mp4, avi, mov, mkv) found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        outdoor_files, chromakey_files, excluded_files = filter_stream_video_pools(
            all_scanned_files, args.domain, args.label, args.video_filter
        )
    except ValueError as exc:
        print(f"[simulated-rtsp][error] Filter mismatch: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned {len(all_scanned_files)} video files. Filter results:")
    print(f"  Outdoor/non-chromakey ({len(outdoor_files)}):")
    for vf in outdoor_files:
        print(f"    - {vf.name}")
    print(f"  Chromakey ({len(chromakey_files)}):")
    for vf in chromakey_files:
        print(f"    - {vf.name}")
    if excluded_files:
        print(f"  Excluded ({len(excluded_files)}):")
        for vf in excluded_files:
            print(f"    - {vf.name}")
    print("--------------------------------------------------", flush=True)

    running_streams: dict[str, dict[str, Any]] = {}
    session_assignments: dict[str, Path] = {}
    assignment_history = load_video_assignments(ASSIGNMENT_STATE_PATH)
    pool_fingerprints = {
        "outdoor": video_pool_fingerprint(outdoor_files),
        "chromakey": video_pool_fingerprint(chromakey_files),
    }
    pending_pool_changes: set[str] = set()

    def stop_stream(camera_login_id: str, stream_info: dict[str, Any]) -> None:
        process = stream_info["process"]
        print(f"[simulated-rtsp] Stopping stream for camera={camera_login_id} (pid={process.pid})", flush=True)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"[simulated-rtsp][warning] ffmpeg did not exit after kill for camera={camera_login_id} "
                    f"(pid={process.pid})",
                    flush=True,
                )
        from ai.worker_registry import unregister_publisher_by_path

        try:
            unregister_publisher_by_path(stream_info["rtsp_url"])
        except (OSError, RuntimeError):
            pass

    def cleanup_all_streams() -> None:
        if not running_streams:
            return
        print("\n[simulated-rtsp] Terminating all streaming processes...", flush=True)
        for camera_login_id, stream_info in list(running_streams.items()):
            stop_stream(camera_login_id, stream_info)
        running_streams.clear()

    def signal_handler(sig: int, frame: Any) -> None:
        cleanup_all_streams()
        lock.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_backend_poll_time = 0.0
    last_status_log_time = 0.0
    try:
        lock.acquire()
        while True:
            now = time.monotonic()
            simulated_cameras = None
            if now - last_backend_poll_time >= poll_interval:
                try:
                    all_scanned_files = scan_stream_video_directories(video_dir, chromakey_video_dir)
                    outdoor_files, chromakey_files, excluded_files = filter_stream_video_pools(
                        all_scanned_files, args.domain, args.label, args.video_filter
                    )
                    scanned_pool_fingerprints = {
                        "outdoor": video_pool_fingerprint(outdoor_files),
                        "chromakey": video_pool_fingerprint(chromakey_files),
                    }
                    for pool_name, fingerprint in scanned_pool_fingerprints.items():
                        if fingerprint != pool_fingerprints[pool_name]:
                            pending_pool_changes.add(pool_name)
                            print(
                                f"[simulated-rtsp] Video pool changed: {pool_name}. "
                                "Affected streams will be reassigned or restarted.",
                                flush=True,
                            )
                    pool_fingerprints = scanned_pool_fingerprints
                except Exception as exc:
                    print(f"[simulated-rtsp][error] Failed during directory scan or filtering: {exc}", file=sys.stderr)
                    cleanup_all_streams()
                    sys.exit(1)

                try:
                    simulated_cameras = load_active_cameras(backend_url, None, timeout_seconds=10.0)
                    last_backend_poll_time = now
                except RuntimeError as exc:
                    print(f"[simulated-rtsp][warning] Failed to load active cameras from backend: {exc}", file=sys.stderr)

            if simulated_cameras is not None:
                current_active_ids: set[str] = set()
                used_videos_by_pool: dict[str, set[Path]] = {"outdoor": set(), "chromakey": set()}
                assignment_history_changed = False
                sorted_cameras = sorted(simulated_cameras, key=lambda item: item.camera_login_id)
                for camera_position, camera in enumerate(sorted_cameras):
                    camera_login_id = camera.camera_login_id
                    current_active_ids.add(camera_login_id)
                    camera_video_pool, use_chromakey = video_pool_for_camera_position(
                        camera_position, outdoor_files, chromakey_files
                    )
                    pool_name = "chromakey" if use_chromakey else "outdoor"
                    used_videos = used_videos_by_pool[pool_name]
                    current_assignment = session_assignments.get(camera_login_id)
                    current_is_available = current_assignment in camera_video_pool
                    current_is_duplicate = current_assignment is not None and current_assignment in used_videos
                    needs_new_assignment = (
                        current_assignment is None
                        or not current_is_available
                        or pool_name in pending_pool_changes
                        or current_is_duplicate
                    )
                    if needs_new_assignment:
                        previous_video = current_assignment or assignment_history.get(camera_login_id)
                        assigned_video = select_rotating_video(camera_video_pool, previous_video, used_videos)
                        session_assignments[camera_login_id] = assigned_video
                        assignment_history[camera_login_id] = assigned_video
                        assignment_history_changed = True
                    else:
                        assigned_video = current_assignment
                    used_videos.add(assigned_video)
                    assigned_fingerprint = video_file_fingerprint(assigned_video)
                    target_rtsp_url = camera_rtsp_url(rtsp_base_url, camera_login_id)

                    existing = running_streams.get(camera_login_id)
                    if existing is not None and stream_config_changed(
                        existing, assigned_video, assigned_fingerprint, target_rtsp_url
                    ):
                        print(f"[simulated-rtsp] Stream config changed for camera={camera_login_id}. Restarting.", flush=True)
                        stop_stream(camera_login_id, existing)
                        del running_streams[camera_login_id]
                        existing = None

                    if existing is None:
                        from ai.worker_registry import force_kill_existing_publisher, register_publisher

                        force_kill_existing_publisher(target_rtsp_url)
                        policy = FfmpegRestartPolicy(
                            requested_mode=args.ffmpeg_mode,
                            initial_mode=initial_ffmpeg_mode,
                            fallback_enabled=not args.no_ffmpeg_fallback,
                        )
                        cmd = build_ffmpeg_cmd(assigned_video, target_rtsp_url, loop_playback, policy.active_mode)
                        from ai.simulated_rtsp_sources import estimate_video_metadata
                        meta = estimate_video_metadata(assigned_video)
                        print(
                            f"[simulated-rtsp] Mapping camera={camera_login_id} "
                            f"pool={'chromakey' if use_chromakey else 'outdoor'} to video={assigned_video.name} "
                            f"(domain={meta['domain']}, label={meta['label']}, source={meta['source']})",
                            flush=True
                        )
                        print(f"  FFmpeg mode: requested={args.ffmpeg_mode}, active={policy.active_mode}", flush=True)
                        print(f"  CMD: {' '.join(cmd)}", flush=True)
                        try:
                            log_dir = Path("runs/simulated_rtsp")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            log_path = log_dir / f"{camera_login_id}-ffmpeg.log"
                            with log_path.open("a", encoding="utf-8") as log_file:
                                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                            register_publisher(target_rtsp_url, process.pid, str(assigned_video), camera_login_id)
                            running_streams[camera_login_id] = {
                                "process": process,
                                "video_path": assigned_video,
                                "video_fingerprint": assigned_fingerprint,
                                "rtsp_url": target_rtsp_url,
                                "policy": policy,
                                "log_path": log_path,
                            }
                            print(f"  Started process (pid={process.pid}), logs redirected to {log_path}", flush=True)
                        except (OSError, RuntimeError) as exc:
                            print(f"[simulated-rtsp][error] Failed to start ffmpeg for camera={camera_login_id}: {exc}", file=sys.stderr)

                if assignment_history_changed:
                    try:
                        save_video_assignments(ASSIGNMENT_STATE_PATH, assignment_history)
                    except OSError as exc:
                        print(
                            f"[simulated-rtsp][warning] Failed to save video assignments: {exc}",
                            file=sys.stderr,
                        )
                pending_pool_changes.clear()

                for camera_login_id in list(running_streams.keys()):
                    if camera_login_id not in current_active_ids:
                        print(f"[simulated-rtsp] Camera={camera_login_id} is no longer active. Stopping stream.", flush=True)
                        stop_stream(camera_login_id, running_streams[camera_login_id])
                        del running_streams[camera_login_id]
                        session_assignments.pop(camera_login_id, None)

            for camera_login_id, stream_info in list(running_streams.items()):
                process = stream_info["process"]
                exit_code = process.poll()
                if exit_code is not None:
                    _restart_stream(camera_login_id, stream_info, exit_code, loop_playback, running_streams)

            if now - last_status_log_time >= 10.0:
                from ai.worker_registry import get_active_publisher_count, get_active_worker_count

                print(
                    f"[simulated-rtsp] Active camera workers: {get_active_worker_count()} "
                    f"| Active simulated publishers: {get_active_publisher_count()}",
                    flush=True,
                )
                last_status_log_time = now

            time.sleep(1.0)
    except KeyboardInterrupt:
        cleanup_all_streams()
        print("[simulated-rtsp] Exiting.", flush=True)
    finally:
        lock.release()


def _restart_stream(
    camera_login_id: str,
    stream_info: dict[str, Any],
    exit_code: int,
    loop_playback: bool,
    running_streams: dict[str, dict[str, Any]],
) -> None:
    process = stream_info["process"]
    print(
        f"[simulated-rtsp][warning] ffmpeg for camera={camera_login_id} (pid={process.pid}) "
        f"exited with code {exit_code}. Restarting.",
        flush=True,
    )
    try:
        process.wait(timeout=0)
    except subprocess.TimeoutExpired:
        pass

    from ai.worker_registry import register_publisher, unregister_publisher_by_path

    unregister_publisher_by_path(stream_info["rtsp_url"])
    stderr_tail = tail_text_file(stream_info["log_path"])
    print(summarize_ffmpeg_exit(camera_login_id, exit_code, stream_info["log_path"]), flush=True)
    policy = stream_info["policy"]
    previous_mode = policy.active_mode
    next_mode = policy.record_exit(exit_code, stderr_tail=stderr_tail)
    if next_mode != previous_mode:
        print(
            f"[simulated-rtsp][warning] camera={camera_login_id} switching ffmpeg mode "
            f"{previous_mode} -> {next_mode} after exit_code={exit_code}",
            flush=True,
        )
    cmd = build_ffmpeg_cmd(stream_info["video_path"], stream_info["rtsp_url"], loop_playback, policy.active_mode)
    try:
        log_path = Path("runs/simulated_rtsp") / f"{camera_login_id}-ffmpeg.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            new_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        register_publisher(stream_info["rtsp_url"], new_process.pid, str(stream_info["video_path"]), camera_login_id)
        running_streams[camera_login_id]["process"] = new_process
        running_streams[camera_login_id]["log_path"] = log_path
        print(f"  Restarted process (pid={new_process.pid}), logs redirected to {log_path}", flush=True)
    except (OSError, RuntimeError) as exc:
        print(f"[simulated-rtsp][error] Failed to restart ffmpeg for camera={camera_login_id}: {exc}", file=sys.stderr)
        del running_streams[camera_login_id]
