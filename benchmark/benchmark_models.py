import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        class _NoopProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def update(self, _count):
                return None

        return _NoopProgress()

try:
    import torch
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import pandas as pd
except ImportError as exc:
    pd = None
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model_benchmark.yaml"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUILTIN_CONFIG = {
    "warmup_frames": 5,
    "keypoint_conf_threshold": 0.3,
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "models": [
        {"name": "YOLOv8n-pose", "candidates": ["yolo8n-pose.pt", "yolov8n-pose.pt"]},
        {"name": "YOLOv11n-pose", "candidates": ["yolo11n-pose.pt"]},
        {"name": "YOLO26n-pose", "candidates": ["yolo26n-pose.pt"]},
        {"name": "YOLOv8s-pose", "candidates": ["yolo8s-pose.pt", "yolov8s-pose.pt"]},
        {"name": "YOLOv11s-pose", "candidates": ["yolo11s-pose.pt"]},
        {"name": "YOLO26s-pose", "candidates": ["yolo26s-pose.pt"]},
    ],
}


COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO pose models on sample videos.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to benchmark YAML config.")
    parser.add_argument("--video-dir", default="sample_videos", help="Directory containing input videos.")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model labels or weight names to benchmark, e.g. YOLO26n-pose or yolo26n-pose.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="auto", help="CUDA device index, 'cpu', or 'auto'.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional maximum measured frames per video.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for CSV/Markdown outputs.")
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if yaml is None:
        print(f"PyYAML import failed, using built-in default config: {YAML_IMPORT_ERROR}")
        return BUILTIN_CONFIG
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def resolve_device(requested):
    if torch is None:
        if requested != "auto" and str(requested).lower() != "cpu":
            print(f"torch import failed, so CUDA device {requested} cannot be used: {TORCH_IMPORT_ERROR}")
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return "0" if cuda_available else "cpu"
    if str(requested).lower() == "cpu":
        return "cpu"
    if cuda_available:
        return str(requested)
    print("CUDA is not available. Falling back to CPU and recording device=cpu.")
    return "cpu"


def is_cuda_device(device):
    return torch is not None and str(device).lower() != "cpu" and torch.cuda.is_available()


def list_videos(video_dir, extensions):
    root = Path(video_dir)
    if not root.exists():
        return []
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in normalized)


def candidate_status_name(model_entry):
    return model_entry.get("name") or ", ".join(model_entry.get("candidates", [])) or "unknown"


def normalize_model_selector(value):
    text = str(value).strip().lower()
    return text[:-3] if text.endswith(".pt") else text


def model_entry_selectors(model_entry):
    selectors = {normalize_model_selector(candidate_status_name(model_entry))}
    for candidate in model_entry.get("candidates", []):
        selectors.add(normalize_model_selector(candidate))
    return selectors


def filter_model_entries(model_entries, requested_models):
    requested = {normalize_model_selector(item) for item in str(requested_models).split(",") if item.strip()}
    if not requested:
        return list(model_entries)

    selected = []
    matched = set()
    for model_entry in model_entries:
        selectors = model_entry_selectors(model_entry)
        if selectors & requested:
            selected.append(model_entry)
            matched.update(selectors & requested)

    missing = sorted(requested - matched)
    if missing:
        available = sorted({selector for model_entry in model_entries for selector in model_entry_selectors(model_entry)})
        raise ValueError(f"Unknown model selector(s): {', '.join(missing)}. Available: {', '.join(available)}")
    return selected


def load_first_available_model(model_entry):
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {ULTRALYTICS_IMPORT_ERROR}")
    if torch is None:
        raise RuntimeError(f"torch import failed: {TORCH_IMPORT_ERROR}")

    errors = []
    for candidate in model_entry.get("candidates", []):
        try:
            return candidate, YOLO(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    joined = " | ".join(errors) if errors else "no candidates configured"
    raise RuntimeError(f"No loadable model candidate for {candidate_status_name(model_entry)}. Tried: {joined}")


def extract_pose_metrics(result, keypoint_conf_threshold):
    person_confidences = []
    keypoint_confidences = []
    missing_keypoints = 0
    total_keypoints = 0
    fall_candidates = 0

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None:
        person_confidences = boxes.conf.detach().float().cpu().tolist()

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.conf is None:
        return {
            "person_confidences": person_confidences,
            "keypoint_confidences": keypoint_confidences,
            "missing_keypoints": missing_keypoints,
            "total_keypoints": total_keypoints,
            "fall_candidates": fall_candidates,
        }

    conf_tensor = keypoints.conf.detach().float().cpu()
    xy_tensor = keypoints.xy.detach().float().cpu() if keypoints.xy is not None else None

    for person_idx in range(conf_tensor.shape[0]):
        kp_conf = conf_tensor[person_idx]
        total_keypoints += int(kp_conf.numel())
        missing_keypoints += int((kp_conf < keypoint_conf_threshold).sum().item())
        visible = kp_conf[kp_conf >= keypoint_conf_threshold]
        if visible.numel() > 0:
            keypoint_confidences.extend(visible.tolist())

        if xy_tensor is not None and evaluate_fall_candidate(xy_tensor[person_idx], kp_conf, keypoint_conf_threshold)["candidate"]:
            fall_candidates += 1

    return {
        "person_confidences": person_confidences,
        "keypoint_confidences": keypoint_confidences,
        "missing_keypoints": missing_keypoints,
        "total_keypoints": total_keypoints,
        "fall_candidates": fall_candidates,
    }


def is_fall_candidate(keypoints_xy, keypoints_conf, threshold):
    return evaluate_fall_candidate(keypoints_xy, keypoints_conf, threshold)["candidate"]


def evaluate_fall_candidate(keypoints_xy, keypoints_conf, threshold):
    required = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
    details = {
        "required_keypoints": required,
        "threshold": float(threshold),
        "required_confidences": {},
        "missing_required": [],
        "shoulder_center": None,
        "hip_center": None,
        "torso_dx": None,
        "torso_dy": None,
        "torso_ratio": None,
        "torso_angle_degrees": None,
        "torso_center": None,
        "torso_center_y": None,
        "uses_bbox_ratio": False,
        "uses_center_height": False,
        "uses_torso_angle": False,
        "uses_torso_ratio": True,
        "uses_confidence_threshold": True,
        "candidate": False,
        "reason": "",
    }
    for idx in required:
        if idx >= tensor_len(keypoints_conf):
            details["missing_required"].append(idx)
            continue
        conf = float(keypoints_conf[idx])
        details["required_confidences"][str(idx)] = conf
        if conf < threshold:
            details["missing_required"].append(idx)
    if details["missing_required"]:
        details["reason"] = "required_keypoint_below_threshold_or_missing"
        return details

    left_shoulder = keypoints_xy[COCO_LEFT_SHOULDER]
    right_shoulder = keypoints_xy[COCO_RIGHT_SHOULDER]
    left_hip = keypoints_xy[COCO_LEFT_HIP]
    right_hip = keypoints_xy[COCO_RIGHT_HIP]

    shoulder_center = midpoint_xy(left_shoulder, right_shoulder)
    hip_center = midpoint_xy(left_hip, right_hip)
    torso_dx = abs(float(shoulder_center[0] - hip_center[0]))
    torso_dy = abs(float(shoulder_center[1] - hip_center[1]))
    torso_ratio = torso_dx / max(torso_dy, 1.0) if torso_dx > 0 else 0.0
    torso_angle_degrees = math.degrees(math.atan2(torso_dy, torso_dx)) if torso_dx > 0 else 90.0
    torso_center = ((shoulder_center[0] + hip_center[0]) / 2.0, (shoulder_center[1] + hip_center[1]) / 2.0)

    details.update(
        {
            "shoulder_center": [float(shoulder_center[0]), float(shoulder_center[1])],
            "hip_center": [float(hip_center[0]), float(hip_center[1])],
            "torso_dx": torso_dx,
            "torso_dy": torso_dy,
            "torso_ratio": torso_ratio,
            "torso_angle_degrees": torso_angle_degrees,
            "torso_center": [float(torso_center[0]), float(torso_center[1])],
            "torso_center_y": float(torso_center[1]),
        }
    )

    if torso_dx <= 0:
        details["reason"] = "zero_horizontal_torso_delta"
        return details
    details["candidate"] = torso_ratio >= 1.3
    details["reason"] = "torso_ratio_pass" if details["candidate"] else "torso_ratio_below_1.3"
    return details


def tensor_len(value):
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(value)


def midpoint_xy(first, second):
    first_x, first_y = float(first[0]), float(first[1])
    second_x, second_y = float(second[0]), float(second[1])
    return ((first_x + second_x) / 2.0, (first_y + second_y) / 2.0)


def mean_or_zero(values):
    return float(statistics.fmean(values)) if values else 0.0


def percentile(values, percentile_value):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((percentile_value / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def run_warmup(model, video_path, warmup_frames, imgsz, device):
    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")
    if warmup_frames <= 0:
        return

    cap = cv2.VideoCapture(str(video_path))
    try:
        count = 0
        while count < warmup_frames:
            ok, frame = cap.read()
            if not ok:
                break
            model.predict(frame, imgsz=imgsz, device=device, verbose=False)
            count += 1
    finally:
        cap.release()


def benchmark_video(model, model_name, video_path, imgsz, device, config, max_frames):
    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    keypoint_threshold = float(config.get("keypoint_conf_threshold", 0.3))
    latencies = []
    person_confidences = []
    keypoint_confidences = []
    missing_keypoints = 0
    total_keypoints = 0
    fall_candidate_count = 0
    processed_frames = 0
    cuda = is_cuda_device(device)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    progress_total = total_frames if total_frames > 0 else None
    if max_frames > 0 and progress_total:
        progress_total = min(progress_total, max_frames)

    try:
        with tqdm(total=progress_total, desc=f"{model_name} | {video_path.name}", unit="frame") as progress:
            while True:
                if max_frames > 0 and processed_frames >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                if cuda:
                    torch.cuda.synchronize()
                frame_started = time.perf_counter()
                results = model.predict(frame, imgsz=imgsz, device=device, verbose=False)
                if cuda:
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - frame_started) * 1000.0
                latencies.append(latency_ms)

                for result in results:
                    metrics = extract_pose_metrics(result, keypoint_threshold)
                    person_confidences.extend(metrics["person_confidences"])
                    keypoint_confidences.extend(metrics["keypoint_confidences"])
                    missing_keypoints += metrics["missing_keypoints"]
                    total_keypoints += metrics["total_keypoints"]
                    fall_candidate_count += metrics["fall_candidates"]

                processed_frames += 1
                progress.update(1)
    finally:
        cap.release()

    elapsed = time.perf_counter() - started
    gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if cuda else 0.0
    missing_rate = missing_keypoints / total_keypoints if total_keypoints else 0.0

    return {
        "model_name": model_name,
        "video_name": video_path.name,
        "device": "cuda" if cuda else "cpu",
        "imgsz": imgsz,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "avg_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "avg_latency_ms": mean_or_zero(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "gpu_memory_mb": gpu_memory_mb,
        "avg_person_confidence": mean_or_zero(person_confidences),
        "avg_keypoint_confidence": mean_or_zero(keypoint_confidences),
        "keypoint_missing_rate": missing_rate,
        "fall_candidate_count": fall_candidate_count,
        "error_or_status": "OK",
    }


def failure_row(model_name, video_name, device, imgsz, status):
    return {
        "model_name": model_name,
        "video_name": video_name,
        "device": "cpu" if str(device).lower() == "cpu" else "cuda",
        "imgsz": imgsz,
        "total_frames": 0,
        "processed_frames": 0,
        "avg_fps": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "gpu_memory_mb": 0.0,
        "avg_person_confidence": 0.0,
        "avg_keypoint_confidence": 0.0,
        "keypoint_missing_rate": 0.0,
        "fall_candidate_count": 0,
        "error_or_status": status,
    }


def round_numeric_columns(df):
    numeric_columns = [
        "avg_fps",
        "avg_latency_ms",
        "p95_latency_ms",
        "gpu_memory_mb",
        "avg_person_confidence",
        "avg_keypoint_confidence",
        "keypoint_missing_rate",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = df[column].astype(float).round(4)
    return df


def round_rows(rows):
    numeric_columns = [
        "avg_fps",
        "avg_latency_ms",
        "p95_latency_ms",
        "gpu_memory_mb",
        "avg_person_confidence",
        "avg_keypoint_confidence",
        "keypoint_missing_rate",
    ]
    rounded = []
    for row in rows:
        next_row = dict(row)
        for column in numeric_columns:
            next_row[column] = round(float(next_row.get(column, 0.0)), 4)
        rounded.append(next_row)
    return rounded


def rows_to_markdown(rows, columns):
    rounded_rows = round_rows(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rounded_rows]
    return "\n".join([header, separator, *body])


def build_model_summary(df):
    ok = df[df["error_or_status"] == "OK"].copy()
    if ok.empty:
        return "\n\n## Model Summary\n\nNo successful benchmark rows were produced.\n"

    grouped = ok.groupby("model_name", as_index=False).agg(
        avg_fps=("avg_fps", "mean"),
        avg_latency_ms=("avg_latency_ms", "mean"),
        p95_latency_ms=("p95_latency_ms", "mean"),
        gpu_memory_mb=("gpu_memory_mb", "max"),
        avg_keypoint_confidence=("avg_keypoint_confidence", "mean"),
        keypoint_missing_rate=("keypoint_missing_rate", "mean"),
        fall_candidate_count=("fall_candidate_count", "sum"),
    )
    grouped = round_numeric_columns(grouped)

    fastest = grouped.sort_values("avg_fps", ascending=False).iloc[0]["model_name"]
    lowest_latency = grouped.sort_values("avg_latency_ms", ascending=True).iloc[0]["model_name"]
    best_keypoints = grouped.sort_values("avg_keypoint_confidence", ascending=False).iloc[0]["model_name"]
    lowest_missing = grouped.sort_values("keypoint_missing_rate", ascending=True).iloc[0]["model_name"]

    lines = [
        "\n\n## Model Summary",
        "",
        rows_to_markdown(grouped.to_dict(orient="records"), list(grouped.columns)),
        "",
        "## Notes",
        "",
        f"- Highest average FPS: {fastest}",
        f"- Lowest average latency: {lowest_latency}",
        f"- Highest average keypoint confidence: {best_keypoints}",
        f"- Lowest keypoint missing rate: {lowest_missing}",
        "- Lower latency/FPS models are usually better for real-time control rooms; higher confidence and lower missing rate are usually better for fall-rule stability.",
    ]
    return "\n".join(lines) + "\n"


def save_results(rows, results_dir):
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_benchmark.csv"
    md_path = output_dir / "model_benchmark.md"

    fieldnames = [
        "model_name",
        "video_name",
        "device",
        "imgsz",
        "total_frames",
        "processed_frames",
        "avg_fps",
        "avg_latency_ms",
        "p95_latency_ms",
        "gpu_memory_mb",
        "avg_person_confidence",
        "avg_keypoint_confidence",
        "keypoint_missing_rate",
        "fall_candidate_count",
        "error_or_status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_content = "# YOLO Pose Model Benchmark\n\n"
    if pd is None:
        print(f"pandas import failed, using built-in Markdown writer: {PANDAS_IMPORT_ERROR}")
        md_content += rows_to_markdown(rows, fieldnames)
        md_content += "\n\n## Model Summary\n\nInstall pandas and rerun to generate grouped model summary statistics.\n"
    else:
        df = pd.DataFrame(rows, columns=fieldnames)
        df = round_numeric_columns(df)
        md_content += rows_to_markdown(df.to_dict(orient="records"), fieldnames)
        md_content += build_model_summary(df)
    md_path.write_text(md_content, encoding="utf-8")
    return csv_path, md_path


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(args.device))
    model_entries = filter_model_entries(config.get("models", []), args.models)
    videos = list_videos(args.video_dir, config.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv"]))

    if not videos:
        print(f"No videos found in {args.video_dir}. Add videos to this folder and run the benchmark again.")
        rows = [
            failure_row(
                candidate_status_name(model_entry),
                "(no video)",
                device,
                args.imgsz,
                "SKIPPED: sample_videos folder is empty or has no supported video files",
            )
            for model_entry in model_entries
        ]
        csv_path, md_path = save_results(rows, args.results_dir)
        print(f"Saved empty benchmark report: {csv_path}")
        print(f"Saved empty benchmark report: {md_path}")
        return 0

    rows = []
    for model_entry in model_entries:
        configured_name = candidate_status_name(model_entry)
        try:
            loaded_name, model = load_first_available_model(model_entry)
        except Exception as exc:
            status_prefix = "SKIPPED" if "yolo26" in configured_name.lower() else "FAILED"
            status = f"{status_prefix}: {exc}"
            print(status)
            for video_path in videos:
                rows.append(failure_row(configured_name, video_path.name, device, args.imgsz, status))
            continue

        model_name = model_entry.get("name", loaded_name)
        for video_path in videos:
            try:
                run_warmup(model, video_path, int(config.get("warmup_frames", 5)), args.imgsz, device)
                rows.append(benchmark_video(model, model_name, video_path, args.imgsz, device, config, args.max_frames))
            except Exception as exc:
                status = f"FAILED: {exc}"
                print(f"{model_name} | {video_path.name}: {status}")
                rows.append(failure_row(model_name, video_path.name, device, args.imgsz, status))

    csv_path, md_path = save_results(rows, args.results_dir)
    print(f"Saved CSV results: {csv_path}")
    print(f"Saved Markdown results: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
