import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
LABEL_NAMES = ("Normal", "Faint")
CHROMAKEY_DOMAIN = "indoor_chromakey"
CHROMAKEY_PATTERN = re.compile(r"(indoor_chromakey|chroma|chromakey|green[_ -]?screen|green|studio|chm|croki|크로마키)", re.IGNORECASE)


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def label_name(row):
    value = (row.get("label_name") or "").strip()
    if value:
        return value
    return "Faint" if (row.get("label") or "").strip() == "1" else "Normal"


def split_name(row, fallback):
    return (row.get("split") or fallback or "unknown").strip()


def classify_domain(row):
    domain = (row.get("domain") or "").strip()
    if domain:
        return domain
    text = " ".join(
        str(row.get(key, ""))
        for key in ("source_video", "video_path", "clip_path", "clip_id", "_resolved_video_path")
    )
    if CHROMAKEY_PATTERN.search(text):
        return CHROMAKEY_DOMAIN
    if "outdoor" in text.lower():
        return "outdoor"
    if "indoor_background" in text.lower():
        return "indoor_background"
    return "unknown"


def is_chromakey(row):
    domain = classify_domain(row)
    if domain == CHROMAKEY_DOMAIN:
        return True
    text = " ".join(
        str(row.get(key, ""))
        for key in ("source_video", "video_path", "clip_path", "clip_id", "_resolved_video_path")
    )
    return bool(CHROMAKEY_PATTERN.search(text))


def attach_audit_columns(rows, fallback_split):
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split_name(item, fallback_split)
        item["domain"] = classify_domain(item)
        item["is_chromakey"] = "1" if is_chromakey(item) else "0"
        item["eval_group"] = "chromakey" if item["is_chromakey"] == "1" else "non_chromakey"
        item["label_name"] = label_name(item)
        output.append(item)
    return output


def summarize(rows):
    summary = {
        "total_rows": len(rows),
        "by_split": {},
        "by_domain": dict(Counter(row["domain"] for row in rows)),
        "by_eval_group": dict(Counter(row["eval_group"] for row in rows)),
        "recommendation": "Use non_chromakey test rows for final threshold audit; keep chromakey rows as training/augmentation only.",
    }
    for split in sorted({row["split"] for row in rows} | set(SPLITS)):
        split_rows = [row for row in rows if row["split"] == split]
        summary["by_split"][split] = {
            "rows": len(split_rows),
            "labels": dict(Counter(row["label_name"] for row in split_rows)),
            "domains": dict(Counter(row["domain"] for row in split_rows)),
            "eval_groups": dict(Counter(row["eval_group"] for row in split_rows)),
            "non_chromakey_labels": dict(Counter(row["label_name"] for row in split_rows if row["eval_group"] == "non_chromakey")),
            "chromakey_labels": dict(Counter(row["label_name"] for row in split_rows if row["eval_group"] == "chromakey")),
        }
    return summary


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary):
    lines = [
        "# Chromakey Split Audit",
        "",
        "Final service validation should use non-chromakey rows. Chromakey rows can remain in training as auxiliary data.",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- By domain: {summary['by_domain']}",
        f"- By eval group: {summary['by_eval_group']}",
        "",
        "| split | rows | non_chromakey | chromakey | non_chromakey_labels | chromakey_labels |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for split, item in summary["by_split"].items():
        groups = item["eval_groups"]
        lines.append(
            f"| {split} | {item['rows']} | {groups.get('non_chromakey', 0)} | {groups.get('chromakey', 0)} | "
            f"{item['non_chromakey_labels']} | {item['chromakey_labels']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(input_paths, output_dir):
    rows = []
    for input_path in input_paths:
        fallback = Path(input_path).stem if Path(input_path).stem in SPLITS else ""
        rows.extend(attach_audit_columns(read_rows(input_path), fallback))
    if not rows:
        raise RuntimeError("No rows found for chromakey audit")
    fieldnames = list(dict.fromkeys([key for row in rows for key in row.keys()]))
    output = Path(output_dir)
    write_csv(output / "all_with_chromakey_flags.csv", rows, fieldnames)
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        write_csv(output / f"{split}_non_chromakey.csv", [row for row in split_rows if row["eval_group"] == "non_chromakey"], fieldnames)
        write_csv(output / f"{split}_chromakey.csv", [row for row in split_rows if row["eval_group"] == "chromakey"], fieldnames)
    summary = summarize(rows)
    (output / "chromakey_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output / "chromakey_audit_report.md", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Separate chromakey and non-chromakey rows for final LSTM validation.")
    parser.add_argument("--input", nargs="+", default=["data/splits/final_source_video_split/all.csv"])
    parser.add_argument("--output-dir", default="data/splits/final_source_video_split/chromakey_audit")
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
