#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add repository root to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import LSTMActionModel
from ai.action.fight_vs_normal_metrics import classification_metrics


CLASS_TO_ID = {"Normal": 0, "Faint": 1}
ID_TO_CLASS = {0: "Normal", 1: "Faint"}


def load_npz_sequences(row: dict, seq_length: int = 30, seq_stride: int = 15, input_size: int = 54, feature_schema: str = "keypoint_bbox54") -> list[np.ndarray]:
    npz_path_str = row.get("npz_path", "")
    if not npz_path_str:
        return []
    npz_path = Path(npz_path_str)
    if not npz_path.exists():
        return []
    try:
        data = np.load(npz_path, allow_pickle=True)
        raw = data.get("data", None)
        if raw is None or len(raw) == 0:
            return []
        
        array = np.asarray(raw, dtype=np.float32)
        target_dim = int(input_size)
        
        if array.ndim == 3:
            actual_dim = array.shape[-1]
            if actual_dim == target_dim:
                return [array[i] for i in range(array.shape[0])]
            elif target_dim == 51:
                return [array[i, :, :51] for i in range(array.shape[0])]
            else: # target_dim == 54 and actual_dim == 51
                if feature_schema == "keypoint_motion54":
                    from ai.action.motion_features import append_motion_features
                    return [append_motion_features(array[i]) for i in range(array.shape[0])]
                else: # keypoint_bbox54 or unknown
                    raise ValueError(f"Cannot pad 51-dim keypoint to keypoint_bbox54 feature dim: actual_dim={actual_dim} -> target_dim={target_dim}")
                    
        elif array.ndim == 2:
            actual_dim = array.shape[-1]
            windows = []
            for start in range(0, max(0, array.shape[0] - seq_length + 1), seq_stride):
                window = array[start : start + seq_length]
                if actual_dim == target_dim:
                    windows.append(window)
                elif target_dim == 51:
                    windows.append(window[:, :51])
                else: # target_dim == 54 and actual_dim == 51
                    if feature_schema == "keypoint_motion54":
                        from ai.action.motion_features import append_motion_features
                        windows.append(append_motion_features(window))
                    else: # keypoint_bbox54 or unknown
                        raise ValueError(f"Cannot pad 51-dim keypoint to keypoint_bbox54 feature dim: actual_dim={actual_dim} -> target_dim={target_dim}")
            return windows
    except ValueError:
        raise
    except Exception:
        pass
    return []


def collect_eval_sequences(rows: list[dict], input_size: int = 54, feature_schema: str = "keypoint_bbox54") -> list[dict]:
    dataset = []
    for row in rows:
        label_val = int(row.get("label", "0"))
        sequences = load_npz_sequences(row, input_size=input_size, feature_schema=feature_schema)
        for seq in sequences:
            dataset.append({
                "features": seq,
                "label": label_val,
                "scenario_tag": row.get("scenario_tag", "unknown"),
                "source_video": row.get("source_video", "unknown")
            })
    return dataset


def evaluate_checkpoint(checkpoint_path: Path, dataset: list[dict], device: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata from checkpoint
    meta = checkpoint.get("metadata", {})
    input_size = meta.get("input_size", 54)
    hidden_size = meta.get("hidden_size", 64)
    num_layers = meta.get("num_layers", 2)
    num_classes = meta.get("num_classes", 2)
    
    model = LSTMActionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    y_true = []
    y_probs = []
    meta_results = []
    
    with torch.no_grad():
        for item in dataset:
            features = item["features"]
            # Add batch and seq dimensions
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            outputs = model(x)
            prob = torch.softmax(outputs, dim=1)[0, 1].item() # Probability of Faint (class 1)
            
            y_true.append(item["label"])
            y_probs.append(prob)
            
            meta_results.append({
                "label": item["label"],
                "probability": prob,
                "scenario_tag": item["scenario_tag"],
                "source_video": item["source_video"]
            })
            
    return np.array(y_true), np.array(y_probs), meta_results


def calculate_metrics_for_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    met = classification_metrics(y_true, y_pred)
    met["threshold"] = threshold
    return met


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare multiple LSTM checkpoints on the same split.")
    parser.add_argument("--eval-split", required=True, help="Path to evaluation split CSV (e.g. data/splits/final_source_video_split/val.csv)")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated paths to model checkpoints (best.pt)")
    parser.add_argument("--labels", required=True, help="Comma-separated labels for checkpoints (e.g. baseline,hn_0.05)")
    parser.add_argument("--feature-schema", default="keypoint_bbox54", help="Enforce feature schema.")
    parser.add_argument("--feature-dim", type=int, default=54, help="Enforce feature dimension.")
    parser.add_argument("--output-dir", default="runs/hard_negative_retraining_comparison", help="Output metrics directory.")
    parser.add_argument("--report-path", default="docs/hard_negative_retraining_performance_comparison.md", help="Markdown report path.")
    parser.add_argument("--device", default="auto", help="Device to run inference (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    eval_path = Path(args.eval_split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not eval_path.exists():
        print(f"[ERROR] Evaluation split not found: {eval_path}")
        sys.exit(1)
        
    checkpoint_paths = [Path(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    
    if len(checkpoint_paths) != len(labels):
        print("[ERROR] Length of checkpoints and labels must match.")
        sys.exit(1)
        
    # Check that all checkpoints exist
    for idx, cp in enumerate(checkpoint_paths):
        if not cp.exists():
            print(f"[ERROR] Checkpoint {labels[idx]} not found: {cp}")
            sys.exit(1)
            
    # Read evaluation metadata
    rows = []
    with eval_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
            
    print(f"Loading features from {len(rows)} evaluation files...")
    eval_dataset = collect_eval_sequences(rows, input_size=args.feature_dim, feature_schema=args.feature_schema)
    print(f"Total evaluation sequences collected: {len(eval_dataset)}")
    
    if len(eval_dataset) == 0:
        print("[ERROR] Evaluation dataset is empty. Cannot continue comparison.")
        sys.exit(1)
        
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    comparison_summary = {}
    
    # Store scenario metadata for breakdown
    scenarios_data = {}
    
    for idx, cp in enumerate(checkpoint_paths):
        label = labels[idx]
        print(f"Evaluating model '{label}' from {cp}...")
        y_true, y_probs, meta_results = evaluate_checkpoint(cp, eval_dataset, device)
        
        # Save predictions to CSV
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_csv_path = pred_dir / f"{label}_eval_predictions.csv"
        with pred_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ground_truth", "probability", "scenario_tag", "source_video"])
            for r in meta_results:
                writer.writerow([r["label"], r["probability"], r["scenario_tag"], r["source_video"]])
                
        # Calculate thresholds
        model_metrics = {}
        for th in thresholds:
            th_metrics = calculate_metrics_for_threshold(y_true, y_probs, th)
            model_metrics[str(th)] = th_metrics
            
        comparison_summary[label] = {
            "metrics": model_metrics,
            "predictions_csv": str(pred_csv_path)
        }
        
        # Scenario breakdown (at primary threshold 0.5)
        for th_key, th_metrics in model_metrics.items():
            if float(th_key) == 0.5:
                # Group by scenario
                for r in meta_results:
                    scen = r["scenario_tag"]
                    is_correct = int((r["probability"] >= 0.5)) == r["label"]
                    if scen not in scenarios_data:
                        scenarios_data[scen] = {}
                    if label not in scenarios_data[scen]:
                        scenarios_data[scen][label] = {"correct": 0, "total": 0}
                    scenarios_data[scen][label]["correct"] += int(is_correct)
                    scenarios_data[scen][label]["total"] += 1

    # Write scenario breakdown to CSV
    scen_csv_path = output_dir / "metrics" / "scenario_breakdown.csv"
    scen_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with scen_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["Scenario Tag"]
        for label in labels:
            header.append(f"{label}_accuracy")
        header.append("Samples")
        writer.writerow(header)
        
        for scen, models_dict in scenarios_data.items():
            row = [scen]
            samples = 0
            for label in labels:
                scen_stats = models_dict.get(label, {"correct": 0, "total": 0})
                acc = scen_stats["correct"] / scen_stats["total"] if scen_stats["total"] > 0 else 0.0
                row.append(f"{acc:.2%}")
                samples = max(samples, scen_stats["total"])
            row.append(samples)
            writer.writerow(row)
            
    # Save overall summary JSON
    summary_json_path = output_dir / "metrics" / "comparison_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
        
    # Generate MD Report
    report_lines = []
    report_lines.append("# Hard Negative Retraining Performance Comparison Report\n")
    report_lines.append("This report presents a direct performance comparison between the baseline and hard-negative retrained variants under identical evaluation splits.\n")
    
    # 0.5 Threshold Comparison Table
    report_lines.append("## Overall Performance Summary (Threshold = 0.5)\n")
    
    table_header = "| Metric | " + " | ".join(labels) + " |"
    table_divider = "| :--- | " + " | ".join([":---:"] * len(labels)) + " |"
    report_lines.append(table_header)
    report_lines.append(table_divider)
    
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "FP", "FN", "TP", "TN"]
    for mkey in metric_keys:
        row_str = f"| **{mkey.upper()}** | "
        vals = []
        for label in labels:
            val = comparison_summary[label]["metrics"]["0.5"].get(mkey, 0.0)
            if mkey in {"FP", "FN", "TP", "TN"}:
                vals.append(f"{int(val)}")
            else:
                vals.append(f"{val:.2%}")
        row_str += " | ".join(vals) + " |"
        report_lines.append(row_str)
    report_lines.append("\n")
    
    # Threshold sweep comparison
    report_lines.append("## Threshold Sweep Analysis (0.3 ~ 0.7)\n")
    report_lines.append("| Threshold | Model | F1 Score | Recall | Precision | FP | FN |")
    report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for th in thresholds:
        for label in labels:
            m = comparison_summary[label]["metrics"][str(th)]
            report_lines.append(f"| {th} | `{label}` | {m.get('f1_score', 0.0):.2%} | {m.get('recall', 0.0):.2%} | {m.get('precision', 0.0):.2%} | {int(m.get('FP', 0))} | {int(m.get('FN', 0))} |")
    report_lines.append("\n")
    
    # Scenario breakdown
    report_lines.append("## Performance by Scenario Tag\n")
    scen_header = "| Scenario Tag | " + " | ".join([f"{l} Accuracy" for l in labels]) + " | Samples |"
    scen_divider = "| :--- | " + " | ".join([":---:"] * len(labels)) + " | :---: |"
    report_lines.append(scen_header)
    report_lines.append(scen_divider)
    
    for scen, models_dict in sorted(scenarios_data.items()):
        row = f"| `{scen}` | "
        vals = []
        samples = 0
        for label in labels:
            scen_stats = models_dict.get(label, {"correct": 0, "total": 0})
            acc = scen_stats["correct"] / scen_stats["total"] if scen_stats["total"] > 0 else 0.0
            vals.append(f"{acc:.2%}")
            samples = max(samples, scen_stats["total"])
        row += " | ".join(vals) + f" | {samples} |"
        report_lines.append(row)
    report_lines.append("\n")
    
    report_lines.append("## Decision Criteria & Recommendation\n")
    report_lines.append("1. **Recall Priority**: Retraining variants must not trigger material Recall drop.\n")
    report_lines.append("2. **FP Reduction**: Select the smallest ratio that achieves the most significant False Positive decrease.\n")
    report_lines.append("3. **Overfitting Warning**: Check if accuracy gains are concentrated in one scenario tag or spread across all scenarios.\n")
    
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Generated comparison markdown report at {report_path}")


if __name__ == "__main__":
    main()
