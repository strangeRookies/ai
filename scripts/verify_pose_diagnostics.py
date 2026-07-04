from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def check_console_log(log_path: Path):
    print("=== [1] Console Log Analysis (ai_runner.log) ===")
    if not log_path.exists():
        print(f"[-] Log file not found at: {log_path}")
        print("    Please run the pipeline first or specify the correct log path.\n")
        return False

    has_config = False
    config_record = None
    has_diagnostics_lines = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[pose-tracking-config]" in line:
                has_config = True
                try:
                    # Extract JSON block
                    json_str = line.split("[pose-tracking-config]", 1)[1].strip()
                    config_record = json.loads(json_str)
                except Exception:
                    pass
            if "[pose-diagnostics]" in line:
                has_diagnostics_lines += 1

    # Check 1: [pose-tracking-config] printed at startup
    if has_config:
        print("[+] SUCCESS: [pose-tracking-config] log found at startup.")
    else:
        print("[-] FAILED: [pose-tracking-config] log NOT found at startup.")

    # Check 2: bytetrack_constructor_ignored display
    if config_record:
        ignored = config_record.get("bytetrack_constructor_ignored")
        print(f"[+] INFO: bytetrack_constructor_ignored field exists: {ignored}")
        if isinstance(ignored, dict) and len(ignored) > 0:
            print("    -> Some parameters were ignored by the installed supervision version.")
        else:
            print("    -> No parameters were ignored (all supported by current supervision version).")
    else:
        print("[-] FAILED: Could not parse pose-tracking-config JSON.")

    # Check 3: [pose-diagnostics] logs appearing
    if has_diagnostics_lines > 0:
        print(f"[+] SUCCESS: Found {has_diagnostics_lines} '[pose-diagnostics]' log lines in console output.")
    else:
        print("[-] FAILED: No '[pose-diagnostics]' log lines found. (Did you set POSE_DEBUG=true or TRACKING_DEBUG=true?)")

    print()
    return True


def check_jsonl_log(jsonl_path: Path):
    print("=== [2] JSONL File Analysis (pose_tracking_diag.jsonl) ===")
    if not jsonl_path.exists():
        print(f"[-] JSONL file not found at: {jsonl_path}")
        print("    Make sure POSE_TRACKING_DIAG_JSONL=true is set in your env.\n")
        return None

    records = []
    corrupted_lines = 0
    missing_fields = set()
    required_fields = [
        "raw_detection_count",
        "avg_keypoint_confidence",
        "active_tracks",
        "track_ids",
        "sequenceReadyCount",
    ]

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("stage") == "pose_frame":
                    records.append(rec)
                    # Check for fields presence
                    for field in required_fields:
                        if field not in rec:
                            missing_fields.add(field)
            except json.JSONDecodeError:
                corrupted_lines += 1

    # Check 8: JSONL parsed line-by-line successfully
    if corrupted_lines == 0 and len(records) > 0:
        print(f"[+] SUCCESS: JSONL file read successfully. Parsed {len(records)} valid records.")
    elif corrupted_lines > 0:
        print(f"[-] WARNING: Found {corrupted_lines} corrupted/invalid lines in JSONL.")
    else:
        print("[-] FAILED: JSONL file is empty or contains no 'pose_frame' stage records.")

    # Check 4-7: Fields existence
    for field in required_fields:
        if field in missing_fields:
            print(f"[-] FAILED: Field '{field}' is missing in some records.")
        else:
            print(f"[+] SUCCESS: Field '{field}' is present in records.")

    print()
    return records


def compare_cameras(records: list[dict]):
    print("=== [3] Camera Side-by-Side Comparison (cam_04 vs cam_05) ===")
    cam_records = {"cam_04": [], "cam_05": []}
    for r in records:
        cam_id = r.get("cameraLoginId")
        if cam_id in cam_records:
            cam_records[cam_id].append(r)

    len_04 = len(cam_records["cam_04"])
    len_05 = len(cam_records["cam_05"])

    print(f"[+] Record count: cam_04 = {len_04}, cam_05 = {len_05}")
    if len_04 == 0 or len_05 == 0:
        print("[-] FAILED: One or both cameras have 0 records in the log. Cannot perform comparison.\n")
        return

    # Helper to check matching sets
    assigned_04 = {r.get("assignedVideoPath") for r in cam_records["cam_04"] if r.get("assignedVideoPath")}
    assigned_05 = {r.get("assignedVideoPath") for r in cam_records["cam_05"] if r.get("assignedVideoPath")}
    source_04 = {r.get("sourceUrl") for r in cam_records["cam_04"] if r.get("sourceUrl")}
    source_05 = {r.get("sourceUrl") for r in cam_records["cam_05"] if r.get("sourceUrl")}

    # Comparison 1: assignedVideoPath
    is_assigned_same = (assigned_04 == assigned_05) and len(assigned_04) > 0
    print(f"[Comparison 1] assignedVideoPath 같은가? : {'예 (Same)' if is_assigned_same else '아니오 (Different)'}")
    print(f"    -> cam_04: {list(assigned_04)}")
    print(f"    -> cam_05: {list(assigned_05)}")

    # Comparison 2: sourceUrl
    is_source_same = (source_04 == source_05) and len(source_04) > 0
    print(f"[Comparison 2] sourceUrl 같은가? : {'예 (Same)' if is_source_same else '아니오 (Different)'}")
    print(f"    -> cam_04: {list(source_04)}")
    print(f"    -> cam_05: {list(source_05)}")

    # Comparison 3: frameId 기준으로 비교 가능한가?
    frames_04 = [r.get("frameId") for r in cam_records["cam_04"]]
    frames_05 = [r.get("frameId") for r in cam_records["cam_05"]]
    has_frame_ids = all(f is not None for f in frames_04 + frames_05)
    print(f"[Comparison 3] frameId 기준으로 비교 가능한가? : {'예 (Available)' if has_frame_ids else '아니오 (Missing/None)'}")

    # Build side-by-side comparison on frameId or timestampMs
    key_field = "frameId" if has_frame_ids else "timestampMs"
    cam_04_by_key = {r.get(key_field): r for r in cam_records["cam_04"]}
    cam_05_by_key = {r.get(key_field): r for r in cam_records["cam_05"]}
    common_keys = sorted(set(cam_04_by_key.keys()).intersection(set(cam_05_by_key.keys())))

    print(f"    -> Common comparison points (by {key_field}): {len(common_keys)}")

    if not common_keys:
        print("[-] WARNING: No common frameIds/timestamps found between cam_04 and cam_05 to align side-by-side.")
        common_keys = sorted(set(cam_04_by_key.keys()).union(set(cam_05_by_key.keys())))[:20]  # Take first 20 as sample
        print("    Showing first 20 sample records for both separately:")

    # Aggregates
    det_diffs = []
    kp_diffs = []
    yolo_detect_tracker_zero_04 = 0
    yolo_detect_tracker_zero_05 = 0
    tracker_active_seq_zero_04 = 0
    tracker_active_seq_zero_05 = 0

    for k in common_keys:
        r4 = cam_04_by_key.get(k)
        r5 = cam_05_by_key.get(k)

        if r4 and r5:
            d4 = int(r4.get("raw_detection_count") or 0)
            d5 = int(r5.get("raw_detection_count") or 0)
            det_diffs.append(abs(d4 - d5))

            kp4 = r4.get("avg_keypoint_confidence")
            kp5 = r5.get("avg_keypoint_confidence")
            if kp4 is not None and kp5 is not None:
                kp_diffs.append(abs(float(kp4) - float(kp5)))

        if r4:
            if int(r4.get("raw_detection_count") or 0) > 0 and int(r4.get("active_tracks") or 0) == 0:
                yolo_detect_tracker_zero_04 += 1
            if int(r4.get("active_tracks") or 0) > 0 and int(r4.get("sequenceReadyCount") or 0) == 0:
                tracker_active_seq_zero_04 += 1

        if r5:
            if int(r5.get("raw_detection_count") or 0) > 0 and int(r5.get("active_tracks") or 0) == 0:
                yolo_detect_tracker_zero_05 += 1
            if int(r5.get("active_tracks") or 0) > 0 and int(r5.get("sequenceReadyCount") or 0) == 0:
                tracker_active_seq_zero_05 += 1

    # Comparison 4: raw_detection_count 차이
    avg_det_diff = sum(det_diffs) / len(det_diffs) if det_diffs else 0.0
    print(f"[Comparison 4] raw_detection_count 차이가 큰가? : {'예 (Diff >= 1.0)' if avg_det_diff >= 1.0 else '아니오 (Diff < 1.0)'}")
    print(f"    -> 평균 감지 갯수 차이: {avg_det_diff:.4f}")

    # Comparison 5: avg_keypoint_confidence 차이
    avg_kp_diff = sum(kp_diffs) / len(kp_diffs) if kp_diffs else 0.0
    print(f"[Comparison 5] avg_keypoint_confidence 차이가 큰가? : {'예 (Diff >= 0.15)' if avg_kp_diff >= 0.15 else '아니오 (Diff < 0.15)'}")
    print(f"    -> 평균 키포인트 신뢰도 차이: {avg_kp_diff:.4f}")

    # Comparison 6: YOLO는 잡는데 active_tracks만 0인가?
    print("[Comparison 6] YOLO는 잡는데 active_tracks만 0인 프레임 수:")
    print(f"    -> cam_04: {yolo_detect_tracker_zero_04} frames")
    print(f"    -> cam_05: {yolo_detect_tracker_zero_05} frames")

    # Comparison 7: active_tracks는 있는데 sequenceReadyCount가 0인가?
    print("[Comparison 7] active_tracks는 있는데 sequenceReadyCount가 0인 프레임 수:")
    print(f"    -> cam_04: {tracker_active_seq_zero_04} frames")
    print(f"    -> cam_05: {tracker_active_seq_zero_05} frames")

    print("\n--- 5개 샘플 프레임 비교 데이터 ---")
    header = f"{key_field:<10} | {'cam_04 det':<10} | {'cam_05 det':<10} | {'cam_04 track':<12} | {'cam_05 track':<12} | {'cam_04 seq':<10} | {'cam_05 seq':<10}"
    print(header)
    print("-" * len(header))
    sampled_keys = common_keys[:5]
    for k in sampled_keys:
        r4 = cam_04_by_key.get(k)
        r5 = cam_05_by_key.get(k)
        det4 = r4.get("raw_detection_count", "-") if r4 else "-"
        det5 = r5.get("raw_detection_count", "-") if r5 else "-"
        tr4 = r4.get("active_tracks", "-") if r4 else "-"
        tr5 = r5.get("active_tracks", "-") if r5 else "-"
        seq4 = r4.get("sequenceReadyCount", "-") if r4 else "-"
        seq5 = r5.get("sequenceReadyCount", "-") if r5 else "-"
        print(f"{k:<10} | {det4:<10} | {det5:<10} | {tr4:<12} | {tr5:<12} | {seq4:<10} | {seq5:<10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO Pose/Tracking diagnostics checklist and compare cam_04 vs cam_05.")
    parser.add_argument("--log-path", type=Path, default=Path("ai_runner.log"), help="Path to console stdout log (ai_runner.log)")
    parser.add_argument("--jsonl-path", type=Path, default=Path("runs/diagnostics/pose_tracking_diag.jsonl"), help="Path to JSONL file")
    args = parser.parse_args()

    print("====================================================")
    print("Pose and Tracking Diagnostics Checker for GPU PC")
    print("====================================================\n")

    check_console_log(args.log_path)
    records = check_jsonl_log(args.jsonl_path)
    if records:
        compare_cameras(records)


if __name__ == "__main__":
    main()
