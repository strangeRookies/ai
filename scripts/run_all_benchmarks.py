import subprocess
import sys
import time
from pathlib import Path

def run_cmd(args):
    print(f"\n[RUNNING] {' '.join(args)}")
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAILED] Error: {res.stderr}")
    else:
        print(f"[SUCCESS] Output:\n{res.stdout}")
    return res.returncode == 0

def main():
    print("==================================================")
    print("Starting smart safety dashboard streaming & event benchmark")
    print("==================================================")

    duration = 15  # duration in seconds for each scenario
    run_id = f"run_{int(time.time())}"

    # 1. Single Camera Scenarios
    print("\n--- 1. Single Camera Scenarios ---")
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "hls", "--cameras", "cam_01", "--duration", str(duration), "--run-id", run_id])
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "webrtc", "--cameras", "cam_01", "--duration", str(duration), "--run-id", run_id])

    # 2. 2-Camera Scenarios
    print("\n--- 2. 2-Camera Scenarios ---")
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "hls", "--cameras", "cam_01", "cam_02", "--duration", str(duration), "--run-id", run_id])
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "webrtc", "--cameras", "cam_01", "cam_02", "--duration", str(duration), "--run-id", run_id])

    # 3. 4-Camera Scenarios
    print("\n--- 3. 4-Camera Scenarios ---")
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "hls", "--cameras", "cam_01", "cam_02", "cam_03", "cam_04", "--duration", str(duration), "--run-id", run_id])
    run_cmd([sys.executable, "scripts/benchmark_streaming.py", "--mode", "webrtc", "--cameras", "cam_01", "cam_02", "cam_03", "cam_04", "--duration", str(duration), "--run-id", run_id])

    # 4. Run Event Notification Latency Test
    print("\n--- 4. Event Notification Latency Test ---")
    run_cmd([sys.executable, "scripts/benchmark_alert_latency.py"])

    # 5. Generate summary report
    print("\n--- 5. Generating Summary Report ---")
    run_cmd([sys.executable, "scripts/summarize_streaming_benchmark.py", "--run-id", run_id])

    print("\n==================================================")
    print("Benchmark completed successfully!")
    print(f"Run ID is: {run_id}")
    print("==================================================")

if __name__ == "__main__":
    main()
