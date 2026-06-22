import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_dataset_evaluation import read_dataset_rows
try:
    from benchmark.keypoint_cache_loader import resolve_keypoint_cache_path, load_keypoint_cache
except ModuleNotFoundError:
    from keypoint_cache_loader import resolve_keypoint_cache_path, load_keypoint_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Check the coverage of keypoint cache against metadata.")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--cache-dir", default="../ai_fall_experiments/data/keypoints/yolo26n-pose")
    parser.add_argument("--deep", action="store_true", help="Load each npz to verify it is not empty/short")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading metadata from: {args.metadata_csv}")
    rows = read_dataset_rows(args.metadata_csv)
    
    stats = {
        "Total": {"count": 0, "hit": 0},
        "Normal": {"count": 0, "hit": 0},
        "Faint": {"count": 0, "hit": 0},
    }
    
    print(f"Checking coverage in: {args.cache_dir}")
    print(f"Deep verification (loading files): {'ON' if args.deep else 'OFF (fast mode)'}")
    
    for idx, row in enumerate(rows):
        label = row.get("label", "Unknown")
        if label not in stats:
            stats[label] = {"count": 0, "hit": 0}
            
        stats["Total"]["count"] += 1
        stats[label]["count"] += 1
        
        existing = resolve_keypoint_cache_path(row, args.cache_dir)
        is_hit = False
        
        if existing and existing.exists():
            is_hit = True
            if args.deep:
                try:
                    data = load_keypoint_cache(existing)
                    if len(data) < 10:
                        is_hit = False
                except Exception:
                    is_hit = False
                    
        if is_hit:
            stats["Total"]["hit"] += 1
            stats[label]["hit"] += 1
            
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} / {len(rows)} clips...")
            
    print("\n" + "="*50)
    print("🎯 Keypoint Cache Coverage Report")
    print("="*50)
    for key, data in stats.items():
        if data["count"] > 0:
            coverage = (data["hit"] / data["count"]) * 100
            print(f"{key:<10}: {data['hit']:>6} / {data['count']:>6} ({coverage:5.1f}% coverage)")
    print("="*50)


if __name__ == "__main__":
    main()
