import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import MockActionClassifier
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.detection.yolo_person_detector import MockPersonDetector, YoloPersonDetector
from ai.publishers.event_publisher import build_event_payload
from ai.streams.video_reader import VideoReader


def read_dataset_videos(path, limit):
    if not path or not Path(path).exists():
        return []
    videos = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            video = row.get("video_path") or row.get("clip_path")
            if video and Path(video).exists():
                videos.append(video)
            if len(videos) >= limit:
                break
    return videos


def load_config(path):
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    except ModuleNotFoundError:
        return load_demo_camera_yaml_without_pyyaml(path)


def load_demo_camera_yaml_without_pyyaml(path):
    cameras = []
    current = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "cameras:":
            continue
        if line.startswith("- "):
            if current:
                cameras.append(current)
            current = {}
            line = line[2:].strip()
        if ":" not in line or current is None:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value.lower() == "true":
            parsed = True
        elif value.lower() == "false":
            parsed = False
        else:
            parsed = value
        current[key.strip()] = parsed
    if current:
        cameras.append(current)
    return {"cameras": cameras}


def create_detector(mode, yolo_model, conf, iou, imgsz):
    if mode == "mock":
        return MockPersonDetector(conf=conf)
    return YoloPersonDetector(yolo_model, conf=conf, iou=iou, imgsz=imgsz)


def process_camera(camera, video_path, args):
    detector = create_detector(args.detector_mode, args.yolo_model, args.yolo_conf, args.yolo_iou, args.imgsz)
    classifier = MockActionClassifier(default_label="Faint", score=0.80)
    sequence_buffer = CropSequenceBuffer(args.sequence_length, args.sequence_stride, args.resize_size)
    summary = {
        "camera_id": camera["camera_id"],
        "name": camera.get("name", ""),
        "rtsp_url": camera["rtsp_url"],
        "local_video": video_path,
        "frames_processed": 0,
        "bbox_detections": 0,
        "generated_sequences": 0,
        "events_generated": 0,
        "decoding_errors": 0,
        "sample_event": None,
        "alert_delivery_result": "local_log_dry_run" if args.dry_run else "local_log",
    }
    try:
        with VideoReader(video_path) as reader:
            while True:
                packet = reader.read()
                if packet is None:
                    break
                if args.max_frames > 0 and summary["frames_processed"] >= args.max_frames:
                    break
                detection = detector.detect(packet.frame, packet.frame_idx)
                boxes = detection["boxes"]
                summary["frames_processed"] += 1
                summary["bbox_detections"] += len(boxes)
                sequence = sequence_buffer.add(packet.frame_idx, packet.frame, boxes)
                if sequence is None:
                    continue
                summary["generated_sequences"] += 1
                prediction = classifier.predict(sequence)
                if not prediction:
                    continue
                payload = build_event_payload(
                    camera_id=camera["camera_id"],
                    frame_idx=packet.frame_idx,
                    timestamp=packet.timestamp,
                    event_type=prediction["label"],
                    score=prediction["score"],
                    boxes=boxes,
                    snapshot_path=None,
                )
                payload["rtsp_url"] = camera["rtsp_url"]
                payload["sequence_window"] = {"start": sequence["start_frame"], "end": sequence["end_frame"]}
                summary["events_generated"] += 1
                if summary["sample_event"] is None:
                    summary["sample_event"] = payload
    except Exception as exc:
        summary["decoding_errors"] += 1
        summary["error"] = str(exc)
    return summary


def rtsp_path(rtsp_url):
    return rtsp_url.rstrip("/").split("/")[-1]


def publisher_command(video_path, camera):
    return ["bash", "scripts/publish_sample_video.sh", video_path, rtsp_path(camera["rtsp_url"])]


def start_publishers(cameras, videos):
    processes = []
    for camera, video in zip(cameras, videos):
        processes.append(subprocess.Popen(publisher_command(video, camera)))
    return processes


def stop_publishers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    parser = argparse.ArgumentParser(description="Run a safe local 4-camera RTSP demo dry-run.")
    parser.add_argument("--config", default="configs/demo_4cams.yaml")
    parser.add_argument("--dataset-csv", default="../ai_fall_experiments/data/metadata/metadata.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--detector-mode", choices=["mock", "yolo"], default="mock")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--start-rtsp-publishers", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    cameras = [cam for cam in config.get("cameras", []) if cam.get("enabled", True)]
    videos = read_dataset_videos(args.dataset_csv, len(cameras))
    if len(videos) < len(cameras):
        raise RuntimeError(f"Need {len(cameras)} local videos, found {len(videos)} in {args.dataset_csv}")

    publish_plan = [
        {
            "camera_id": camera["camera_id"],
            "rtsp_url": camera["rtsp_url"],
            "local_video": video,
            "command": " ".join(publisher_command(video, camera)),
        }
        for camera, video in zip(cameras, videos)
    ]
    publisher_processes = []
    if args.start_rtsp_publishers:
        publisher_processes = start_publishers(cameras, videos)

    results = []
    try:
        for camera, video in zip(cameras, videos):
            results.append(process_camera(camera, video, args))
    finally:
        stop_publishers(publisher_processes)

    final = {
        "config": args.config,
        "dry_run": args.dry_run,
        "rtsp_publish_plan": publish_plan,
        "rtsp_streams_configured": len(cameras),
        "rtsp_streams_started": len(publisher_processes),
        "messaging_architecture": "local_log_dry_run; MQTT/EQMS path not contacted",
        "cameras": results,
        "totals": {
            "frames_processed": sum(item["frames_processed"] for item in results),
            "bbox_detections": sum(item["bbox_detections"] for item in results),
            "generated_sequences": sum(item["generated_sequences"] for item in results),
            "events_generated": sum(item["events_generated"] for item in results),
            "failed_cameras": sum(1 for item in results if item["decoding_errors"]),
        },
    }
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
