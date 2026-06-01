import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark.benchmark_models import evaluate_fall_candidate, resolve_device

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


MODEL_CANDIDATES = {
    "YOLOv8s-pose": ["yolo8s-pose.pt", "yolov8s-pose.pt"],
    "YOLOv11n-pose": ["yolo11n-pose.pt"],
    "YOLO26n-pose": ["yolo26n-pose.pt"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose fall-candidate rule behavior across YOLO pose models.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default="benchmark/results/fall_candidate_diagnostics")
    parser.add_argument("--models", default="YOLOv8s-pose,YOLOv11n-pose,YOLO26n-pose")
    parser.add_argument("--frame-indices", default="", help="Comma-separated exact frame indices.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keypoint-conf-threshold", type=float, default=0.3)
    return parser.parse_args()


def load_model(model_label):
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {ULTRALYTICS_IMPORT_ERROR}")
    errors = []
    for candidate in MODEL_CANDIDATES.get(model_label, [model_label]):
        try:
            return candidate, YOLO(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(" | ".join(errors))


def sample_frame_indices(video_path, args):
    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")
    if args.frame_indices:
        return [int(value.strip()) for value in args.frame_indices.split(",") if value.strip()]
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    start = max(0, int(args.start_frame))
    end = int(args.end_frame) if args.end_frame > 0 else max(start, total - 1)
    if end < start:
        end = start
    if args.samples <= 1:
        return [start]
    step = max((end - start) / float(args.samples - 1), 1.0)
    return sorted({int(round(start + idx * step)) for idx in range(args.samples)})


def read_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                frames[frame_idx] = frame
    finally:
        cap.release()
    return frames


def tensor_to_list(tensor):
    if tensor is None:
        return []
    return tensor.detach().float().cpu().tolist()


def result_format_report(result):
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    report = {
        "boxes_xyxy_shape": None,
        "boxes_xywh_shape": None,
        "boxes_xyxyn_shape": None,
        "boxes_xyxy_minmax": None,
        "boxes_xyxyn_minmax": None,
        "keypoints_xy_shape": None,
        "keypoints_xyn_shape": None,
        "keypoints_conf_shape": None,
        "keypoints_xy_minmax": None,
        "keypoints_xyn_minmax": None,
    }
    if boxes is not None:
        for name in ("xyxy", "xywh", "xyxyn"):
            value = getattr(boxes, name, None)
            if value is not None:
                report[f"boxes_{name}_shape"] = list(value.shape)
                if name in {"xyxy", "xyxyn"} and value.numel() > 0:
                    cpu_value = value.detach().float().cpu()
                    report[f"boxes_{name}_minmax"] = [float(cpu_value.min()), float(cpu_value.max())]
    if keypoints is not None:
        for name in ("xy", "xyn", "conf"):
            value = getattr(keypoints, name, None)
            if value is not None:
                report[f"keypoints_{name}_shape"] = list(value.shape)
                if name in {"xy", "xyn"} and value.numel() > 0:
                    cpu_value = value.detach().float().cpu()
                    report[f"keypoints_{name}_minmax"] = [float(cpu_value.min()), float(cpu_value.max())]
    return report


def analyze_result(result, frame_idx, model_label, loaded_name, threshold):
    rows = []
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    xyxy = tensor_to_list(getattr(boxes, "xyxy", None)) if boxes is not None else []
    confs = tensor_to_list(getattr(boxes, "conf", None)) if boxes is not None else []
    kxy = keypoints.xy.detach().float().cpu() if keypoints is not None and keypoints.xy is not None else None
    kconf = keypoints.conf.detach().float().cpu() if keypoints is not None and keypoints.conf is not None else None
    format_report = result_format_report(result)

    person_count = max(len(xyxy), int(kconf.shape[0]) if kconf is not None else 0)
    for person_idx in range(person_count):
        bbox = xyxy[person_idx] if person_idx < len(xyxy) else None
        bbox_width = max(float(bbox[2]) - float(bbox[0]), 0.0) if bbox else None
        bbox_height = max(float(bbox[3]) - float(bbox[1]), 0.0) if bbox else None
        bbox_wh_ratio = bbox_width / max(bbox_height, 1.0) if bbox_width is not None else None
        if kxy is not None and kconf is not None and person_idx < kconf.shape[0]:
            fall_rule = evaluate_fall_candidate(kxy[person_idx], kconf[person_idx], threshold)
            keypoint_coordinates = kxy[person_idx].tolist()
            keypoint_confidence = kconf[person_idx].tolist()
        else:
            fall_rule = {"candidate": False, "reason": "missing_keypoint_tensor"}
            keypoint_coordinates = []
            keypoint_confidence = []

        rows.append(
            {
                "frame_idx": frame_idx,
                "model_label": model_label,
                "loaded_model": loaded_name,
                "person_idx": person_idx,
                "bbox_xyxy": bbox,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_width_height_ratio": bbox_wh_ratio,
                "person_confidence": confs[person_idx] if person_idx < len(confs) else None,
                "keypoint_coordinates": keypoint_coordinates,
                "keypoint_confidence": keypoint_confidence,
                "fall_rule": fall_rule,
                "format_report": format_report,
                "candidate": bool(fall_rule.get("candidate")),
                "reason": fall_rule.get("reason", ""),
            }
        )
    if not rows:
        rows.append(
            {
                "frame_idx": frame_idx,
                "model_label": model_label,
                "loaded_model": loaded_name,
                "person_idx": None,
                "bbox_xyxy": None,
                "bbox_width": None,
                "bbox_height": None,
                "bbox_width_height_ratio": None,
                "person_confidence": None,
                "keypoint_coordinates": [],
                "keypoint_confidence": [],
                "fall_rule": {"candidate": False, "reason": "no_person_result"},
                "format_report": format_report,
                "candidate": False,
                "reason": "no_person_result",
            }
        )
    return rows


def draw_overlay(frame, rows, model_label, frame_idx, output_path):
    image = frame.copy()
    for row in rows:
        bbox = row.get("bbox_xyxy")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if row["candidate"] else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{model_label} cand={row['candidate']} ratio={_fmt(row['fall_rule'].get('torso_ratio'))}"
            cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        keypoints = row.get("keypoint_coordinates") or []
        confs = row.get("keypoint_confidence") or []
        for idx, point in enumerate(keypoints):
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            if conf < 0.3:
                continue
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 191, 0), -1)
    cv2.putText(image, f"frame={frame_idx}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fall_candidate_diagnostics.json"
    csv_path = output_dir / "fall_candidate_diagnostics.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "frame_idx",
        "model_label",
        "loaded_model",
        "person_idx",
        "bbox_xyxy",
        "bbox_width",
        "bbox_height",
        "bbox_width_height_ratio",
        "person_confidence",
        "keypoint_coordinates",
        "keypoint_confidence",
        "fall_rule",
        "format_report",
        "candidate",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fieldnames})
    return csv_path, json_path


def print_diagnosis(rows):
    by_model = {}
    for row in rows:
        stats = by_model.setdefault(row["model_label"], {"people": 0, "candidates": 0, "reasons": {}})
        if row["person_idx"] is not None:
            stats["people"] += 1
        if row["candidate"]:
            stats["candidates"] += 1
        stats["reasons"][row["reason"]] = stats["reasons"].get(row["reason"], 0) + 1
    comparison = compare_yolo26_against_other_models(rows)
    parsing = summarize_parsing_formats(rows)
    print(json.dumps({"diagnosis_summary": by_model, "cross_model_comparison": comparison, "parsing_format_summary": parsing}, indent=2, ensure_ascii=False))
    yolo26 = by_model.get("YOLO26n-pose")
    if comparison["other_models_candidate_frames"] and comparison["yolo26_people_frames"] and not comparison["yolo26_candidate_frames"]:
        print(
            "[diagnosis] Other models produced fall candidates on sampled frames, while YOLO26n detected people but never passed the current torso-ratio rule. "
            "This points to rule/model-output incompatibility or different keypoint geometry rather than a simple missing-detection issue.",
            flush=True,
        )
    elif yolo26 and yolo26["people"] > 0 and yolo26["candidates"] == 0:
        print(
            "[diagnosis] YOLO26n detected people/keypoints in sampled frames, but the torso-ratio fall rule never passed. "
            "If other models also have zero candidates, the sampled section may not satisfy the current rule.",
            flush=True,
        )
    elif yolo26 and yolo26["people"] == 0:
        print("[diagnosis] YOLO26n produced no person rows in sampled frames; this points to detection/model loading rather than fall-rule incompatibility.", flush=True)
    yolo26_parsing = parsing.get("YOLO26n-pose", {})
    if yolo26_parsing.get("warnings"):
        print(f"[diagnosis] YOLO26n parsing warnings: {', '.join(yolo26_parsing['warnings'])}", flush=True)


def summarize_parsing_formats(rows):
    summary = {}
    for row in rows:
        model = row["model_label"]
        stats = summary.setdefault(
            model,
            {
                "rows": 0,
                "xyxy_pixel_like_rows": 0,
                "xyxy_normalized_like_rows": 0,
                "xyxyn_normalized_like_rows": 0,
                "keypoints_xy_pixel_like_rows": 0,
                "keypoints_xyn_normalized_like_rows": 0,
                "keypoints_conf_shape_examples": [],
                "warnings": [],
            },
        )
        stats["rows"] += 1
        report = row.get("format_report") or {}
        xyxy = report.get("boxes_xyxy_minmax")
        xyxyn = report.get("boxes_xyxyn_minmax")
        kxy = report.get("keypoints_xy_minmax")
        kxyn = report.get("keypoints_xyn_minmax")
        kconf_shape = report.get("keypoints_conf_shape")
        if xyxy:
            if float(xyxy[1]) <= 1.5:
                stats["xyxy_normalized_like_rows"] += 1
            else:
                stats["xyxy_pixel_like_rows"] += 1
        if xyxyn and 0.0 <= float(xyxyn[0]) and float(xyxyn[1]) <= 1.5:
            stats["xyxyn_normalized_like_rows"] += 1
        if kxy:
            if float(kxy[1]) > 1.5:
                stats["keypoints_xy_pixel_like_rows"] += 1
        if kxyn and 0.0 <= float(kxyn[0]) and float(kxyn[1]) <= 1.5:
            stats["keypoints_xyn_normalized_like_rows"] += 1
        if kconf_shape and kconf_shape not in stats["keypoints_conf_shape_examples"]:
            stats["keypoints_conf_shape_examples"].append(kconf_shape)

    for model, stats in summary.items():
        if stats["xyxy_normalized_like_rows"] > 0:
            stats["warnings"].append("boxes.xyxy looks normalized; benchmark expects pixel xyxy")
        if stats["rows"] > 0 and stats["xyxy_pixel_like_rows"] == 0 and stats["xyxy_normalized_like_rows"] == 0:
            stats["warnings"].append("boxes.xyxy missing in sampled rows")
        if stats["rows"] > 0 and stats["keypoints_xy_pixel_like_rows"] == 0:
            stats["warnings"].append("keypoints.xy missing or not pixel-like in sampled rows")
        if not any(shape and len(shape) == 2 and shape[-1] == 17 for shape in stats["keypoints_conf_shape_examples"]):
            stats["warnings"].append("keypoints.conf shape does not look like [persons, 17]")
    return summary


def compare_yolo26_against_other_models(rows):
    by_frame = {}
    for row in rows:
        frame = by_frame.setdefault(row["frame_idx"], {})
        model = frame.setdefault(row["model_label"], {"people": 0, "candidate": False, "reasons": set(), "torso_ratios": []})
        if row["person_idx"] is not None:
            model["people"] += 1
        if row["candidate"]:
            model["candidate"] = True
        if row.get("reason"):
            model["reasons"].add(row["reason"])
        torso_ratio = row.get("fall_rule", {}).get("torso_ratio")
        if torso_ratio is not None:
            model["torso_ratios"].append(float(torso_ratio))

    other_candidate_frames = []
    yolo26_people_frames = []
    yolo26_candidate_frames = []
    yolo26_false_while_other_true = []
    yolo26_missing_while_other_true = []
    frame_summaries = []
    for frame_idx, models in sorted(by_frame.items()):
        yolo26 = models.get("YOLO26n-pose", {"people": 0, "candidate": False, "reasons": set(), "torso_ratios": []})
        other_candidate = any(label != "YOLO26n-pose" and item["candidate"] for label, item in models.items())
        if other_candidate:
            other_candidate_frames.append(frame_idx)
        if yolo26["people"] > 0:
            yolo26_people_frames.append(frame_idx)
        if yolo26["candidate"]:
            yolo26_candidate_frames.append(frame_idx)
        if other_candidate and yolo26["people"] > 0 and not yolo26["candidate"]:
            yolo26_false_while_other_true.append(frame_idx)
        if other_candidate and yolo26["people"] == 0:
            yolo26_missing_while_other_true.append(frame_idx)
        frame_summaries.append(
            {
                "frame_idx": frame_idx,
                "other_model_candidate": other_candidate,
                "yolo26_people": yolo26["people"],
                "yolo26_candidate": yolo26["candidate"],
                "yolo26_reasons": sorted(yolo26["reasons"]),
                "yolo26_torso_ratios": yolo26["torso_ratios"],
            }
        )

    return {
        "other_models_candidate_frames": other_candidate_frames,
        "yolo26_people_frames": yolo26_people_frames,
        "yolo26_candidate_frames": yolo26_candidate_frames,
        "yolo26_false_while_other_model_true_frames": yolo26_false_while_other_true,
        "yolo26_missing_while_other_model_true_frames": yolo26_missing_while_other_true,
        "per_frame": frame_summaries,
    }


def main():
    args = parse_args()
    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    device = resolve_device(str(args.device))
    frame_indices = sample_frame_indices(video_path, args)
    frames = read_frames(video_path, frame_indices)
    if not frames:
        raise RuntimeError(f"No sampled frames could be read from {video_path}")

    all_rows = []
    model_labels = [item.strip() for item in args.models.split(",") if item.strip()]
    for model_label in model_labels:
        loaded_name, model = load_model(model_label)
        model_overlay_dir = output_dir / "overlays" / model_label
        for frame_idx in frame_indices:
            frame = frames.get(frame_idx)
            if frame is None:
                continue
            results = model.predict(frame, imgsz=args.imgsz, device=device, verbose=False)
            frame_rows = []
            for result in results:
                frame_rows.extend(analyze_result(result, frame_idx, model_label, loaded_name, args.keypoint_conf_threshold))
            all_rows.extend(frame_rows)
            draw_overlay(frame, frame_rows, model_label, frame_idx, model_overlay_dir / f"frame_{frame_idx:06d}.jpg")

    csv_path, json_path = write_outputs(all_rows, output_dir)
    print_diagnosis(all_rows)
    print(f"Saved diagnostics CSV: {csv_path}")
    print(f"Saved diagnostics JSON: {json_path}")
    print(f"Saved overlays under: {output_dir / 'overlays'}")


if __name__ == "__main__":
    main()
