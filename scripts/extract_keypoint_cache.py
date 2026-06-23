import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.streams.video_reader import VideoReader
from detector.yolo_pose_detector import YoloPoseDetector
from scripts.run_dataset_evaluation import read_dataset_rows

try:
    from benchmark.keypoint_cache_loader import resolve_keypoint_cache_path, load_keypoint_cache
except ModuleNotFoundError:
    from keypoint_cache_loader import resolve_keypoint_cache_path, load_keypoint_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Extract YOLO keypoints to npz cache files for faster LSTM training.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--cache-dir", default="../ai_fall_experiments/data/keypoints/yolo26n-pose")
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--faint-only", action="store_true", help="Only process Faint clips")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model: {args.model} on {args.device}")
    detector = YoloPoseDetector(args.model, args.device)
    
    print(f"Loading metadata from: {args.metadata_csv}")
    rows = read_dataset_rows(args.metadata_csv)
    
    if args.faint_only:
        rows = [r for r in rows if r.get("label", "").lower() == "faint"]
        print(f"Filtered for Faint only. Total: {len(rows)}")
    else:
        # Sort so Faint comes first to prioritize generating missing minority class
        rows = sorted(rows, key=lambda r: 0 if r.get("label", "").lower() == "faint" else 1)
        print(f"Sorted rows to prioritize Faint. Total: {len(rows)}")
        
    success = 0
    skipped = 0
    errors = 0
    
    for idx, row in enumerate(rows):
        clip_id = row.get("clip_id") or Path(row.get("video_path") or row.get("clip_path") or "").stem
        video_path = row.get("_resolved_video_path") or row.get("video_path") or row.get("clip_path")
        
        if not video_path:
            errors += 1
            continue
            
        label = row.get("label", "Unknown")
        
        # Check existing cache
        existing = resolve_keypoint_cache_path(row, args.cache_dir)
        target_path = None
        
        if existing and existing.exists():
            if not args.overwrite:
                # Basic validation to ensure it's not a short/broken clip
                try:
                    data = load_keypoint_cache(existing)
                    if len(data) >= 10:  # arbitrary valid minimum length
                        skipped += 1
                        continue
                except Exception:
                    pass  # Broken cache, proceed to overwrite
            target_path = existing
        
        if not target_path:
            target_dir = cache_dir / label
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / f"{clip_id}.npz"
            
        print(f"[{idx+1}/{len(rows)}] Extracting {clip_id} ({label})")
        frames = []
        try:
            with VideoReader(video_path) as reader:
                while True:
                    packet = reader.read()
                    if packet is None:
                        break
                    detections = detector.detect(packet.frame)
                    frames.append({
                        "frame_idx": packet.frame_idx,
                        "detections": detections,
                        "frame_shape": packet.frame.shape
                    })
            if frames:
                np.savez_compressed(target_path, data=np.array(frames, dtype=object))
                success += 1
            else:
                errors += 1
                print(f"  -> Error: No frames read.")
        except Exception as e:
            errors += 1
            print(f"  -> Error: {e}")
            
    print(f"\nExtraction Done! Success: {success}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
