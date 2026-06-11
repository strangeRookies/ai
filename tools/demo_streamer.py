#!/usr/bin/env python3
import os
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Demo Streamer for RTSP")
    parser.add_argument("--video", required=True, help="Path to mp4 file")
    parser.add_argument("--rtsp-url", required=True, help="RTSP URL to push to (e.g. rtsp://localhost:8554/cam1)")
    parser.add_argument("--no-resize", action="store_true", help="Do not resize video (use original 4K etc with -c:v copy)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    while True:
        if args.no_resize:
            cmd = [
                "ffmpeg", "-re", "-stream_loop", "-1", "-i", args.video,
                "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", args.rtsp_url
            ]
        else:
            cmd = [
                "ffmpeg", "-re", "-stream_loop", "-1", "-i", args.video,
                "-vf", "scale=-2:720",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", 
                "-g", "30", "-keyint_min", "30",
                "-b:v", "1500k", "-f", "rtsp", "-rtsp_transport", "tcp", args.rtsp_url
            ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        print("FFmpeg process exited, restarting in 2 seconds...")
        time.sleep(2)

if __name__ == "__main__":
    main()
