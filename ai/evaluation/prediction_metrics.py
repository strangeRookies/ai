import itertools
import json
from pathlib import Path

from ai.evaluation.prediction_log import NEGATIVE_LABEL, POSITIVE_LABEL, normalize_ground_truth


DEFAULT_EVAL_FOLDERS = ("normal_basic", "faint", "hard_negative", "rtsp_real")


def read_prediction_logs(sample_root):
    root = Path(sample_root)
    rows = []
    for folder in DEFAULT_EVAL_FOLDERS:
        directory = root / folder
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            rows.extend(read_jsonl(path, folder))
        for path in sorted(directory.rglob("*.json")):
            rows.extend(read_json(path, folder))
    return rows


def read_jsonl(path, folder_label):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(with_folder_ground_truth(json.loads(line), folder_label))
    return rows


def read_json(path, folder_label):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    return [with_folder_ground_truth(dict(item), folder_label) for item in items]


def with_folder_ground_truth(row, folder_label):
    if not row.get("ground_truth"):
        row["ground_truth"] = folder_ground_truth(folder_label)
    row["ground_truth"] = normalize_ground_truth(row.get("ground_truth"))
    return row


def folder_ground_truth(folder_label):
    if folder_label == "rtsp_real":
        return None
    return POSITIVE_LABEL if folder_label == "faint" else NEGATIVE_LABEL


def labeled_rows(rows):
    return [row for row in rows if normalize_ground_truth(row.get("ground_truth")) is not None]


def metrics(rows):
    tp = fp = fn = tn = 0
    for row in labeled_rows(rows):
        truth = normalize_ground_truth(row.get("ground_truth"))
        predicted = normalize_prediction(row.get("prediction"), row.get("event_emitted"))
        if predicted == POSITIVE_LABEL and truth == POSITIVE_LABEL:
            tp += 1
        elif predicted == POSITIVE_LABEL and truth == NEGATIVE_LABEL:
            fp += 1
        elif predicted == NEGATIVE_LABEL and truth == POSITIVE_LABEL:
            fn += 1
        else:
            tn += 1
    return metric_payload(tp, fp, fn, tn)


def metric_payload(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "false_positive_count": fp,
        "false_negative_count": fn,
        "total_labeled": tp + fp + fn + tn,
    }


def normalize_prediction(prediction, event_emitted=None):
    if event_emitted is not None:
        return POSITIVE_LABEL if bool(event_emitted) else NEGATIVE_LABEL
    return POSITIVE_LABEL if str(prediction) == POSITIVE_LABEL else NEGATIVE_LABEL


def threshold_sweep(rows, threshold_grid):
    results = []
    for combo in itertools.product(
        threshold_grid["faint_confidence_threshold"],
        threshold_grid["consecutive_faint_count"],
        threshold_grid["event_cooldown_seconds"],
        threshold_grid["max_keypoint_missing_rate"],
        threshold_grid["min_avg_keypoint_conf"],
    ):
        params = {
            "faint_confidence_threshold": combo[0],
            "consecutive_faint_count": combo[1],
            "event_cooldown_seconds": combo[2],
            "max_keypoint_missing_rate": combo[3],
            "min_avg_keypoint_conf": combo[4],
        }
        swept_rows = apply_thresholds(rows, params)
        result = metrics(swept_rows)
        result["thresholds"] = params
        results.append(result)
    return sorted(results, key=sweep_rank, reverse=True)


def apply_thresholds(rows, params):
    sorted_rows = sorted(rows, key=row_sort_key)
    last_event_time_by_key = {}
    output = []
    for row in sorted_rows:
        candidate = threshold_candidate(row, params)
        key = event_key(row)
        timestamp = float(row.get("timestamp") or 0.0)
        in_cooldown = False
        last_event_time = last_event_time_by_key.get(key)
        if last_event_time is not None:
            in_cooldown = timestamp - last_event_time < float(params["event_cooldown_seconds"])
        swept = dict(row)
        swept["event_emitted"] = bool(candidate and not in_cooldown)
        if swept["event_emitted"]:
            last_event_time_by_key[key] = timestamp
        output.append(swept)
    return output


def threshold_candidate(row, params):
    return (
        str(row.get("prediction")) == POSITIVE_LABEL
        and float(row.get("confidence") or 0.0) >= float(params["faint_confidence_threshold"])
        and int(row.get("consecutive_faint_count") or 0) >= int(params["consecutive_faint_count"])
        and float(row.get("keypoint_missing_rate") or 0.0) <= float(params["max_keypoint_missing_rate"])
        and float(row.get("avg_keypoint_conf") or 0.0) >= float(params["min_avg_keypoint_conf"])
    )


def row_sort_key(row):
    return (
        str(row.get("source_id") or row.get("video_id") or ""),
        str(row.get("camera_id") or ""),
        int(row.get("track_id") or -1),
        float(row.get("timestamp") or 0.0),
        int(row.get("frame_idx") or 0),
    )


def event_key(row):
    return (row.get("source_id") or row.get("video_id"), row.get("camera_id"), row.get("track_id"))


def sweep_rank(row):
    return (float(row["f1_score"]), float(row["recall"]), -int(row["false_positive_count"]))
