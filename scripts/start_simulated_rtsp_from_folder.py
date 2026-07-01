#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure that the root directory is on the python search path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.registered_cameras import load_active_cameras, RegisteredCamera, camera_rtsp_url


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-scan a folder of videos and publish them as simulated RTSP streams based on active cameras."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing video files (mp4, avi, mov, mkv)."
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8080",
        help="Backend base URL."
    )
    parser.add_argument(
        "--rtsp-host",
        default="127.0.0.1",
        help="RTSP server publish host."
    )
    parser.add_argument(
        "--rtsp-port",
        type=int,
        default=8554,
        help="RTSP server publish port."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Interval in seconds to poll the backend for active cameras."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Indefinitely loop video playback."
    )
    parser.add_argument(
        "--no-loop",
        action="store_false",
        dest="loop",
        help="Do not loop video playback."
    )
    parser.add_argument(
        "--ffmpeg-mode",
        choices=["copy", "cpu", "nvenc"],
        default=os.environ.get("FFMPEG_MODE", "copy"),
        help="FFmpeg encoding mode. CPU/NVENC modes apply the browser-safe output profile."
    )
    return parser.parse_args()


def scan_video_directory(directory_path: str) -> list[Path]:
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"[simulated-rtsp][error] Specified --video-dir is not a directory: {directory_path}", file=sys.stderr)
        sys.exit(1)
    
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    video_files.sort(key=lambda x: x.name)
    return video_files


def build_ffmpeg_cmd(video_path: Path, rtsp_url: str, loop: bool, ffmpeg_mode: str = "cpu") -> list[str]:
    cmd = ["ffmpeg", "-re"]
    if loop:
        cmd.extend(["-stream_loop", "-1"])
    cmd.extend(["-i", str(video_path), "-an"])

    mode = ffmpeg_mode.lower()
    if mode == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend([
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=15",
            "-c:v", "h264_nvenc" if mode == "nvenc" else "libx264",
            "-preset", "p1" if mode == "nvenc" else "ultrafast",
            "-tune", "ull" if mode == "nvenc" else "zerolatency",
        ])

    if mode != "copy":
        cmd.extend([
        "-pix_fmt", "yuv420p",
        "-g", "30",
        "-keyint_min", "30",
        "-sc_threshold", "0",
        "-b:v", "1500k",
        "-maxrate", "1800k",
        "-bufsize", "3000k",
        ])
    cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", rtsp_url])
    return cmd


def resolve_assigned_video_path(camera: RegisteredCamera) -> Path | None:
    if not camera.assigned_video_path:
        return None

    assigned = Path(camera.assigned_video_path)
    if assigned.exists():
        return assigned

    repo_relative = Path(__file__).resolve().parents[1] / assigned
    if repo_relative.exists():
        return repo_relative

    return None


def video_for_camera(camera: RegisteredCamera, video_files: list[Path], index: int) -> Path:
    assigned_video = resolve_assigned_video_path(camera)
    if assigned_video is not None:
        return assigned_video
    return video_files[index % len(video_files)]


def stable_video_index(camera_login_id: str, video_count: int) -> int:
    if video_count <= 0:
        raise ValueError("video_count must be positive")
    digest = hashlib.sha256(camera_login_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % video_count


def main() -> None:
    args = parse_arguments()
    video_dir = args.video_dir
    backend_url = args.backend_url
    rtsp_base_url = f"rtsp://{args.rtsp_host}:{args.rtsp_port}"
    poll_interval = args.poll_interval
    loop_playback = args.loop

    print("==================================================", flush=True)
    print(f"Starting simulated RTSP stream publisher", flush=True)
    print(f"Video Directory: {video_dir}", flush=True)
    print(f"Backend URL:     {backend_url}", flush=True)
    print(f"RTSP Base URL:   {rtsp_base_url}", flush=True)
    print(f"Poll Interval:   {poll_interval}s", flush=True)
    print(f"Loop Playback:   {loop_playback}", flush=True)
    print("==================================================", flush=True)

    # Scan directory initially
    video_files = scan_video_directory(video_dir)
    if not video_files:
        print(f"[simulated-rtsp][error] No video files (mp4, avi, mov, mkv) found in {video_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanned {len(video_files)} video files:")
    for vf in video_files:
        print(f"  - {vf.name}")
    print("--------------------------------------------------", flush=True)

    # running_streams maps cameraLoginId -> { 'process': Popen, 'video_path': Path, 'rtsp_url': str }
    running_streams: dict[str, dict[str, Any]] = {}

    def stop_stream(camera_login_id: str, stream_info: dict[str, Any]) -> None:
        p = stream_info['process']
        print(f"[simulated-rtsp] Stopping stream for camera={camera_login_id} (pid={p.pid})", flush=True)
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        from ai.worker_registry import unregister_publisher_by_path
        try:
            unregister_publisher_by_path(stream_info['rtsp_url'])
        except Exception:
            pass

    def cleanup_all_streams() -> None:
        if not running_streams:
            return
        print("\n[simulated-rtsp] Terminating all streaming processes...", flush=True)
        for cid, info in list(running_streams.items()):
            stop_stream(cid, info)
        running_streams.clear()

    # Graceful shutdown handler
    def signal_handler(sig: int, frame: Any) -> None:
        cleanup_all_streams()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_backend_poll_time = 0.0
    last_status_log_time = 0.0
    try:
        while True:
            now = time.monotonic()
            simulated_cameras = None
            if now - last_backend_poll_time >= poll_interval:
                # Re-scan video files to allow adding new videos dynamically
                try:
                    video_files = scan_video_directory(video_dir)
                except Exception as e:
                    print(f"[simulated-rtsp][warning] Failed to scan video directory: {e}", file=sys.stderr)
                
                if not video_files:
                    print(f"[simulated-rtsp][error] Video directory became empty! Exiting.", file=sys.stderr)
                    cleanup_all_streams()
                    sys.exit(1)

                # Query active cameras from backend
                try:
                    active_cameras = load_active_cameras(backend_url, None, timeout_seconds=10.0)
                    # Filter to only keep SIMULATED_RTSP cameras
                    simulated_cameras = active_cameras
                    last_backend_poll_time = now
                except Exception as e:
                    print(f"[simulated-rtsp][warning] Failed to load active cameras from backend: {e}", file=sys.stderr)
                    # Fallback to keep existing streams running if backend query fails
                    simulated_cameras = None

            if simulated_cameras is not None:
                current_active_ids = set()
                
                # Circularly assign scanned video files to active simulated cameras
                for camera in simulated_cameras:
                    cid = camera.camera_login_id
                    current_active_ids.add(cid)

                    video_index = stable_video_index(cid, len(video_files))
                    assigned_video = video_for_camera(camera, video_files, video_index)
                    target_rtsp_url = camera_rtsp_url(rtsp_base_url, cid)

                    # Check if stream is already running for this camera
                    existing = running_streams.get(cid)
                    if existing is not None:
                        # If video file or RTSP url changed, we must restart
                        if existing['video_path'] != assigned_video or existing['rtsp_url'] != target_rtsp_url:
                            print(f"[simulated-rtsp] Stream config changed for camera={cid}. Restarting.", flush=True)
                            stop_stream(cid, existing)
                            del running_streams[cid]
                            existing = None
                    
                    if existing is None:
                        # Start new streaming process
                        from ai.worker_registry import force_kill_existing_publisher, register_publisher
                        force_kill_existing_publisher(target_rtsp_url)
                        cmd = build_ffmpeg_cmd(assigned_video, target_rtsp_url, loop_playback, args.ffmpeg_mode)
                        print(f"[simulated-rtsp] Mapping camera={cid} to video={assigned_video.name}", flush=True)
                        print(f"  CMD: {' '.join(cmd)}", flush=True)
                        
                        try:
                            log_dir = Path("runs/simulated_rtsp")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            log_path = log_dir / f"{cid}-ffmpeg.log"
                            with log_path.open("a", encoding="utf-8") as log_file:
                                p = subprocess.Popen(
                                    cmd,
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL
                                )
                            register_publisher(target_rtsp_url, p.pid, str(assigned_video), cid)
                            running_streams[cid] = {
                                'process': p,
                                'video_path': assigned_video,
                                'rtsp_url': target_rtsp_url
                            }
                            print(f"  Started process (pid={p.pid}), logs redirected to {log_path}", flush=True)
                        except Exception as ex:
                            print(f"[simulated-rtsp][error] Failed to start ffmpeg for camera={cid}: {ex}", file=sys.stderr)

                # Stop cameras that are no longer active
                for cid in list(running_streams.keys()):
                    if cid not in current_active_ids:
                        print(f"[simulated-rtsp] Camera={cid} is no longer active. Stopping stream.", flush=True)
                        stop_stream(cid, running_streams[cid])
                        del running_streams[cid]

            # Check for crashed processes (every 1 second)
            for cid, info in list(running_streams.items()):
                p = info['process']
                exit_code = p.poll()
                if exit_code is not None:
                    print(f"[simulated-rtsp][warning] ffmpeg for camera={cid} (pid={p.pid}) exited with code {exit_code} in mode '{args.ffmpeg_mode}'. Restarting.", flush=True)
                    from ai.worker_registry import force_kill_existing_publisher, register_publisher
                    force_kill_existing_publisher(info['rtsp_url'])
                    cmd = build_ffmpeg_cmd(info['video_path'], info['rtsp_url'], loop_playback, args.ffmpeg_mode)
                    try:
                        log_dir = Path("runs/simulated_rtsp")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_path = log_dir / f"{cid}-ffmpeg.log"
                        with log_path.open("a", encoding="utf-8") as log_file:
                            new_p = subprocess.Popen(
                                cmd,
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL
                            )
                        register_publisher(info['rtsp_url'], new_p.pid, str(info['video_path']), cid)
                        running_streams[cid]['process'] = new_p
                        print(f"  Restarted process (pid={new_p.pid}), logs redirected to {log_path}", flush=True)
                    except Exception as ex:
                        print(f"[simulated-rtsp][error] Failed to restart ffmpeg for camera={cid}: {ex}", file=sys.stderr)
                        del running_streams[cid]

            # Periodic status logging (every 10 seconds)
            if now - last_status_log_time >= 10.0:
                from ai.worker_registry import get_active_worker_count, get_active_publisher_count
                print(
                    f"[simulated-rtsp] Active camera workers: {get_active_worker_count()} "
                    f"| Active simulated publishers: {get_active_publisher_count()}",
                    flush=True
                )
                last_status_log_time = now
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        cleanup_all_streams()
        print("[simulated-rtsp] Exiting.", flush=True)


if __name__ == "__main__":
    main()
