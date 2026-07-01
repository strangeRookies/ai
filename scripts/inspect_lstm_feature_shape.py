#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import sequence_to_lstm_features


def main():
    parser = argparse.ArgumentParser(description="Inspect keypoint LSTM feature shape.")
    parser.add_argument("--feature-schema", default="keypoint51", choices=["keypoint51", "keypoint_motion54", "keypoint_bbox54"])
    parser.add_argument("--expected-input-size", type=int, default=51)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    print(f"=== Inspecting LSTM Feature Shapes ===")
    print(f"Schema: {args.feature_schema}")
    print(f"Expected Input Size: {args.expected_input_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Sample size to print: {args.sample}")

    # Build dummy sequence
    detections = []
    frame_shapes = []
    for t in range(args.sequence_length):
        # mock keypoints moving downwards slightly
        kps = []
        for i in range(17):
            kps.append({
                "x": 100.0 + i * 5 + t * 2,
                "y": 200.0 + i * 5 + t * 4,
                "confidence": 0.9 if i % 2 == 0 else 0.5
            })
        
        # mock bbox coordinates: x1, y1, x2, y2
        # width: 100 + t, height: 200 + t * 2
        bbox = [50.0, 60.0, 150.0 + t, 260.0 + t * 2]
        
        detections.append({
            "bbox": bbox,
            "keypoints": kps
        })
        frame_shapes.append((1080, 1920, 3))

    sequence = {
        "detections": detections,
        "frame_shapes": frame_shapes
    }

    # Extract features
    features = sequence_to_lstm_features(
        sequence, 
        input_size=args.expected_input_size, 
        feature_schema=args.feature_schema
    )

    print(f"\nResulting Feature Array:")
    print(f"  Shape: {features.shape}")
    print(f"  Dtype: {features.dtype}")
    
    expected_shape = (args.sequence_length, args.expected_input_size)
    if features.shape == expected_shape:
        print("  Shape match check: PASS")
    else:
        print(f"  Shape match check: FAIL (expected {expected_shape}, got {features.shape})")

    print(f"\nFirst {args.sample} frames features:")
    for t_idx in range(min(args.sample, args.sequence_length)):
        frame_feat = features[t_idx]
        print(f"  Frame {t_idx}:")
        print(f"    Base keypoint coords (first 3 values): {frame_feat[:3]}")
        if args.expected_input_size == 54:
            print(f"    Additional 3 features (dims 51-53): {frame_feat[51:]}")


if __name__ == "__main__":
    main()
