#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare 51-dim vs 54-dim LSTM models.")
    parser.add_argument("--metrics-a", required=True, help="Path to metrics JSON for model A")
    parser.add_argument("--metrics-b", required=True, help="Path to metrics JSON for model B")
    parser.add_argument("--name-a", default="keypoint51", help="Name of model A")
    parser.add_argument("--name-b", default="keypoint_bbox54", help="Name of model B")
    parser.add_argument("--output-csv", default="runs/evaluation_feature_dim/comparison_51_vs_54.csv", help="Path to output comparison CSV")
    parser.add_argument("--report-path", default="reports/lstm_feature_51_vs_54_eval.md", help="Path to output Markdown report")
    args = parser.parse_args()

    metrics_a_path = Path(args.metrics_a)
    metrics_b_path = Path(args.metrics_b)

    if not metrics_a_path.exists():
        print(f"[ERROR] Metrics A path not found: {metrics_a_path}")
        return
    if not metrics_b_path.exists():
        print(f"[ERROR] Metrics B path not found: {metrics_b_path}")
        return

    with metrics_a_path.open("r", encoding="utf-8") as fp:
        data_a = json.load(fp)
    with metrics_b_path.open("r", encoding="utf-8") as fp:
        data_b = json.load(fp)

    # 1. Base comparison metrics
    comparison_metrics = ["accuracy", "precision", "recall", "f1_score", "FP", "FN", "TP", "TN"]
    comparison_rows = []

    for metric in comparison_metrics:
        val_a = data_a.get(metric, 0.0)
        val_b = data_b.get(metric, 0.0)
        diff = val_b - val_a
        comparison_rows.append({
            "Metric": metric,
            args.name_a: val_a,
            args.name_b: val_b,
            "Diff": diff
        })

    # Save to CSV
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["Metric", args.name_a, args.name_b, "Diff"])
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    print(f"Saved comparison CSV to {output_csv_path}")

    # Generate MD Report
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append(f"# Performance Comparison: {args.name_a} vs {args.name_b}\n")
    report_lines.append("This report presents a direct performance comparison between the 51-dimensional baseline model and the 54-dimensional feature model using the same balanced evaluation split.\n")
    
    # Model Metadata Table
    report_lines.append("## Model Configurations\n")
    report_lines.append("| Property | Model A (Baseline) | Model B (Retrained) |")
    report_lines.append("| :--- | :--- | :--- |")
    report_lines.append(f"| **Name** | `{args.name_a}` | `{args.name_b}` |")
    report_lines.append(f"| **Input size** | `{data_a.get('input_size', 51)}` | `{data_b.get('input_size', 54)}` |")
    report_lines.append(f"| **Feature Schema** | `{data_a.get('feature_schema_version', 'keypoint51')}` | `{data_b.get('feature_schema_version', 'keypoint_bbox54')}` |")
    report_lines.append("\n")

    # Metrics Summary Table
    report_lines.append("## Overall Performance Summary\n")
    report_lines.append(f"| Metric | {args.name_a} | {args.name_b} | Difference |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for r in comparison_rows:
        m = r["Metric"]
        v_a = r[args.name_a]
        v_b = r[args.name_b]
        diff = r["Diff"]
        
        # format float or integer
        if m in {"FP", "FN", "TP", "TN"}:
            fmt_a = f"{int(v_a)}"
            fmt_b = f"{int(v_b)}"
            fmt_diff = f"{int(diff):+d}"
        else:
            fmt_a = f"{v_a:.2%}"
            fmt_b = f"{v_b:.2%}"
            fmt_diff = f"{diff:+.2%}"
            
        report_lines.append(f"| {m.upper()} | {fmt_a} | {fmt_b} | {fmt_diff} |")
    report_lines.append("\n")

    # Threshold sweeps comparison
    report_lines.append("## Threshold Sweep Analysis (0.3 ~ 0.7)\n")
    report_lines.append(f"| Threshold | {args.name_a} F1 | {args.name_b} F1 | Difference | {args.name_a} Recall | {args.name_b} Recall |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    thresholds_a = data_a.get("thresholds", {})
    thresholds_b = data_b.get("thresholds", {})
    for th in sorted(list(set(list(thresholds_a.keys()) + list(thresholds_b.keys())))):
        th_data_a = thresholds_a.get(str(th), thresholds_a.get(float(th), {}))
        th_data_b = thresholds_b.get(str(th), thresholds_b.get(float(th), {}))
        
        f1_a = th_data_a.get("f1", 0.0)
        f1_b = th_data_b.get("f1", 0.0)
        rec_a = th_data_a.get("recall", 0.0)
        rec_b = th_data_b.get("recall", 0.0)
        
        report_lines.append(f"| {th} | {f1_a:.2%} | {f1_b:.2%} | {f1_b-f1_a:+.2%} | {rec_a:.2%} | {rec_b:.2%} |")
    report_lines.append("\n")

    # Scenario Tags Comparison
    report_lines.append("## Performance by Scenario Tag\n")
    report_lines.append(f"| Scenario Tag | {args.name_a} Accuracy | {args.name_b} Accuracy | Difference | Samples |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    tags_a = data_a.get("tags", {})
    tags_b = data_b.get("tags", {})
    for tag in sorted(list(set(list(tags_a.keys()) + list(tags_b.keys())))):
        tag_data_a = tags_a.get(tag, {})
        tag_data_b = tags_b.get(tag, {})
        
        acc_a = tag_data_a.get("accuracy", 0.0)
        acc_b = tag_data_b.get("accuracy", 0.0)
        total = tag_data_a.get("total", tag_data_b.get("total", 0))
        
        report_lines.append(f"| `{tag}` | {acc_a:.2%} | {acc_b:.2%} | {acc_b-acc_a:+.2%} | {total} |")
    report_lines.append("\n")

    # Source Types Comparison
    report_lines.append("## Performance by Source Type\n")
    report_lines.append(f"| Source Type | {args.name_a} Accuracy | {args.name_b} Accuracy | Difference | Samples |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    augs_a = data_a.get("augs", {})
    augs_b = data_b.get("augs", {})
    for src in sorted(list(set(list(augs_a.keys()) + list(augs_b.keys())))):
        src_data_a = augs_a.get(src, {})
        src_data_b = augs_b.get(src, {})
        
        acc_a = src_data_a.get("accuracy", 0.0)
        acc_b = src_data_b.get("accuracy", 0.0)
        total = src_data_a.get("total", src_data_b.get("total", 0))
        
        report_lines.append(f"| `{src}` | {acc_a:.2%} | {acc_b:.2%} | {acc_b-acc_a:+.2%} | {total} |")
    report_lines.append("\n")

    report_lines.append("## Conclusion & Rationale\n")
    report_lines.append("- Analyze the performance changes carefully. Check whether adding bounding box dimensions (`keypoint_bbox54`) or motion details (`keypoint_motion54`) helps the classifier distinguish true fainting actions from negative actions (like sitting down or bending over).\n")
    report_lines.append("> [!IMPORTANT]\n")
    report_lines.append("> These performance metrics are based on the balanced evaluation split. Make sure to run full real mode validation on the target GPU server before shipping to production.\n")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Generated comparison markdown report at {report_path}")


if __name__ == "__main__":
    main()
