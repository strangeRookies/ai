import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_json_command(command):
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Command produced no stdout: {' '.join(command)}")
    return json.loads(stdout[stdout.find("{") :]), completed.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description="Run dataset split and 4-camera RTSP demo verification.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--config", default="configs/demo_4cams.yaml")
    parser.add_argument("--output-dir", default="runs/verification")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--detector-mode", choices=["mock", "yolo"], default="mock")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--start-rtsp-publishers", action="store_true")
    parser.add_argument("--read-from-rtsp", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rtsp_summary_path = output_dir / "rtsp_demo_summary.json"

    split_command = [
        sys.executable,
        "scripts/check_dataset_split.py",
        "--metadata-csv",
        args.metadata_csv,
    ]
    rtsp_command = [
        sys.executable,
        "scripts/run_rtsp_demo.py",
        "--config",
        args.config,
        "--dataset-csv",
        args.metadata_csv,
        "--dry-run",
        "--detector-mode",
        args.detector_mode,
        "--max-frames",
        str(args.max_frames),
        "--sequence-length",
        str(args.sequence_length),
        "--sequence-stride",
        str(args.sequence_stride),
        "--output",
        str(rtsp_summary_path),
    ]
    if args.start_rtsp_publishers:
        rtsp_command.append("--start-rtsp-publishers")
    if args.read_from_rtsp:
        rtsp_command.append("--read-from-rtsp")

    split_summary, split_stderr = run_json_command(split_command)
    rtsp_summary, rtsp_stderr = run_json_command(rtsp_command)

    final = {
        "dataset_split_summary": split_summary,
        "rtsp_demo_summary": rtsp_summary,
        "commands": {
            "split": " ".join(split_command),
            "rtsp_demo": " ".join(rtsp_command),
        },
        "stderr": {
            "split": split_stderr,
            "rtsp_demo": rtsp_stderr,
        },
        "remaining_blockers": [],
    }
    if not split_summary.get("ratio_ok_approx"):
        final["remaining_blockers"].append("Dataset split ratio is not approximately 70/15/15.")
    if split_summary.get("leakage_count", 0) > 0:
        final["remaining_blockers"].append("Source-video leakage detected across splits.")
    if rtsp_summary.get("totals", {}).get("failed_cameras", 0) > 0:
        final["remaining_blockers"].append("One or more demo cameras failed to decode.")
    if rtsp_summary.get("totals", {}).get("events_generated", 0) == 0:
        final["remaining_blockers"].append("No demo events were generated.")

    final_path = output_dir / "final_summary.json"
    final_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"[verification] saved {final_path}")


if __name__ == "__main__":
    main()
