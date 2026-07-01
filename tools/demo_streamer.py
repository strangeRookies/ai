#!/usr/bin/env python3
import os
import subprocess
import time
import argparse
import sys
import re

def normalize_rtsp_url(url: str) -> str:
    # Rewrite endpoints ending in /cam1~/cam4 or similar to standard /cam_01~/cam_04
    stripped = url.rstrip('/')
    match = re.search(r'^(rtsp://.*?/)(cam_?\d+)$', stripped, re.IGNORECASE)
    if match:
        base, cam_id = match.groups()
        cam_match = re.match(r'^cam_?(\d+)$', cam_id, re.IGNORECASE)
        if cam_match:
            num = int(cam_match.group(1))
            return f"{base}cam_{num:02d}"
    return url

def main():
    parser = argparse.ArgumentParser(description="Demo Streamer for RTSP")
    parser.add_argument("--video", required=True, help="Path to mp4 file")
    parser.add_argument("--rtsp-url", required=True, help="RTSP URL to push to (e.g. rtsp://localhost:8554/cam1)")
    parser.add_argument("--no-resize", action="store_true", help="Do not resize video (use original 4K etc with -c:v copy)")
    parser.add_argument(
        "--ffmpeg-mode",
        choices=["copy", "cpu", "nvenc"],
        default=os.environ.get("FFMPEG_MODE", "cpu"),
        help="FFmpeg encoding mode (copy: direct copy, cpu: libx264 encoding, nvenc: h264_nvenc hardware acceleration)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    normalized_url = normalize_rtsp_url(args.rtsp_url)
    if normalized_url != args.rtsp_url:
        print(f"Normalized RTSP URL: {args.rtsp_url} -> {normalized_url}")

    # Determine mode: default is args.ffmpeg_mode. But if no-resize is requested and
    # ffmpeg-mode wasn't explicitly overridden by argument or env var, fallback to copy mode
    # to maintain legacy --no-resize behavior.
    mode = args.ffmpeg_mode.lower()
    has_explicit_mode = any(arg.startswith("--ffmpeg-mode") for arg in sys.argv)
    if args.no_resize and not has_explicit_mode and "FFMPEG_MODE" not in os.environ:
        mode = "copy"

    while True:
        cmd = ["ffmpeg", "-re", "-stream_loop", "-1", "-i", args.video]
        
        # Apply scaling filter if not in copy mode and no_resize is not requested
        if mode != "copy" and not args.no_resize:
            cmd.extend(["-vf", "scale=-2:720"])
            
        # Add video codec and encoding options
        if mode == "copy":
            cmd.extend(["-c:v", "copy"])
        elif mode == "nvenc":
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "ull",
                "-g", "30",
                "-keyint_min", "30",
                "-b:v", "1500k"
            ])
        else:  # cpu mode (default)
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-g", "30",
                "-keyint_min", "30",
                "-b:v", "1500k"
            ])
            
        cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", normalized_url])
        
        print(f"Running (mode={mode}): {' '.join(cmd)}")
        
        current_proc = None
        
        def signal_handler(sig, frame):
            nonlocal current_proc
            if current_proc is not None:
                print(f"\n[demo-streamer] Terminating child ffmpeg (pid={current_proc.pid})...", flush=True)
                current_proc.terminate()
                try:
                    current_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    current_proc.kill()
            sys.exit(0)

        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            current_proc = subprocess.Popen(cmd)
            exit_code = current_proc.wait()
            print(f"FFmpeg process exited with code {exit_code}, restarting in 2 seconds...")
        except Exception as e:
            print(f"Failed to run ffmpeg: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    main()
