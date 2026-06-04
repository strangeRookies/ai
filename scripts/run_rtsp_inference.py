import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import LSTMActionClassifier, MockActionClassifier
from ai.action.faint_post_processing import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    FaintEventPostProcessor,
    faint_probability,
    is_alert_prediction,
)
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.publishers.event_publisher import ConsoleEventPublisher, build_event_payload
from ai.runtime_metrics import RuntimeMetrics
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector


def create_detector(mode, model, device, imgsz=640, conf=0.25):
    if mode == "mock":
        return MockDetector(model_name="mock-pose-detector")
    return YoloPoseDetector(model, device=device, imgsz=imgsz, conf=conf)


def create_classifier(action_model, device, action_threshold=DEFAULT_FAINT_THRESHOLD):
    if action_model:
        return LSTMActionClassifier(action_model, device=device, faint_threshold=action_threshold), "lstm_checkpoint"
    return MockActionClassifier(default_label="Faint", score=0.80), "mock_lstm"


def normalize_detections(detections):
    boxes = []
    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        boxes.append(
            {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "score": float(detection.get("confidence", 0.0)),
                "class_name": "person",
                "keypoints": detection.get("keypoints"),
            }
        )
    return boxes


def mock_keypoints_for_bbox(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    points = []
    for idx in range(17):
        col = idx % 5
        row = idx // 5
        points.append(
            {
                "x": round(x1 + width * (0.2 + col * 0.15), 2),
                "y": round(y1 + height * (0.1 + row * 0.2), 2),
                "confidence": 0.9,
            }
        )
    return points


def ensure_mock_keypoints(detections):
    for detection in detections:
        if detection.get("keypoints"):
            continue
        bbox = detection.get("bbox")
        if bbox:
            detection["keypoints"] = mock_keypoints_for_bbox(bbox)
    return detections


def maybe_log_debug(packet, boxes, summary, prediction, args):
    every_n = max(0, int(getattr(args, "debug_every_n", 30)))
    missing_detection = len(boxes) == 0
    should_log = missing_detection or (every_n > 0 and summary["frames_processed"] % every_n == 0)
    if not should_log:
        return
    faint_prob = faint_probability(prediction)
    faint_text = "None" if faint_prob is None else f"{faint_prob:.4f}"
    print(
        "[rtsp-inference-debug] "
        f"frame={packet.frame_idx} "
        f"bbox={len(boxes)} "
        f"keypoints={summary.get('latest_frame_keypoints', 0)} "
        f"seq={summary['generated_sequences']} "
        f"pred={summary['lstm_predictions']} "
        f"latest_faint_prob={faint_text}",
        flush=True,
    )


def build_inference_event_payload(args, packet, prediction, boxes, sequence):
    payload = build_event_payload(
        camera_id=args.camera_id,
        frame_idx=packet.frame_idx,
        timestamp=packet.timestamp,
        event_type=prediction["label"],
        score=prediction["score"],
        boxes=boxes,
        snapshot_path=None,
    )
    bbox = sequence.get("bbox") if sequence else None
    track_id = sequence.get("track_id") if sequence else None
    payload["bbox"] = bbox
    payload["confidence"] = prediction["score"]
    payload["threshold"] = getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD)
    payload["track_id"] = track_id
    payload["severity"] = getattr(args, "event_severity", "HIGH")
    payload["sequence_window"] = {"start": sequence["start_frame"], "end": sequence["end_frame"]} if sequence else None
    payload["probabilities"] = prediction.get("probabilities", {})
    payload["post_processing"] = {
        "min_consecutive_faint": getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        "camera_cooldown_seconds": getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
    }
    return payload


def run(args):
    detector = create_detector(args.detector_mode, args.yolo_model, args.device, getattr(args, "imgsz", 640), conf=getattr(args, "detector_conf", 0.25))
    classifier, classifier_mode = create_classifier(args.action_model, args.action_device, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD))
    keypoint_buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
    classifier_input = getattr(args, "classifier_input", "keypoints")
    crop_buffer = CropSequenceBuffer(args.sequence_length, args.sequence_stride, args.resize_size) if args.action_model and classifier_input == "crops" else None
    post_processor = FaintEventPostProcessor(
        min_consecutive_faint=getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        cooldown_seconds=getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
    )
    publisher = ConsoleEventPublisher()
    metrics = RuntimeMetrics()
    writer = None
    summary = {
        "rtsp_url": args.rtsp_url,
        "camera_id": args.camera_id,
        "dry_run": args.dry_run,
        "detector_mode": args.detector_mode,
        "yolo_model": args.yolo_model if args.detector_mode == "real" else None,
        "classifier_mode": classifier_mode,
        "classifier_input": classifier_input,
        "action_threshold": getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD),
        "min_consecutive_faint": getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        "camera_cooldown_seconds": getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
        "latest_faint_probability": None,
        "latest_prediction_label": None,
        "latest_frame_keypoints": 0,
        "frames_processed": 0,
        "bbox_detections": 0,
        "keypoints_extracted": 0,
        "generated_sequences": 0,
        "lstm_predictions": 0,
        "events_generated": 0,
        "sample_event": None,
        "alert_delivery_result": "local_log_dry_run" if args.dry_run else "local_log",
    }

    with VideoReader(args.rtsp_url) as reader:
        while True:
            if args.max_frames > 0 and summary["frames_processed"] >= args.max_frames:
                break
            frame_started_at = time.perf_counter()
            read_started_at = time.perf_counter()
            packet = reader.read()
            metrics.add_read_ms((time.perf_counter() - read_started_at) * 1000.0)
            if packet is None:
                break

            yolo_started_at = time.perf_counter()
            detections = detector.detect(packet.frame)
            metrics.add_yolo_ms((time.perf_counter() - yolo_started_at) * 1000.0)
            if args.detector_mode == "mock":
                detections = ensure_mock_keypoints(detections)
            boxes = normalize_detections(detections)
            frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
            summary["frames_processed"] += 1
            summary["bbox_detections"] += len(boxes)
            summary["keypoints_extracted"] += frame_keypoint_count
            summary["latest_frame_keypoints"] = frame_keypoint_count

            keypoint_sequence = keypoint_buffer.add(packet.frame_idx, detections, packet.frame.shape)
            crop_sequence = crop_buffer.add(packet.frame_idx, packet.frame, boxes) if crop_buffer else None
            classifier_sequence = crop_sequence if crop_sequence else keypoint_sequence
            if classifier_sequence:
                lstm_started_at = time.perf_counter()
                prediction = classifier.predict(classifier_sequence)
                metrics.add_lstm_ms((time.perf_counter() - lstm_started_at) * 1000.0)
            else:
                prediction = None
            if keypoint_sequence:
                summary["generated_sequences"] += 1
            if prediction:
                summary["lstm_predictions"] += 1
                summary["latest_prediction_label"] = prediction.get("label")
                summary["latest_faint_probability"] = faint_probability(prediction)
            if post_processor.should_trigger(args.camera_id, prediction, packet.timestamp):
                sequence_for_event = keypoint_sequence or crop_sequence
                payload = build_inference_event_payload(args, packet, prediction, boxes, sequence_for_event)
                if args.dry_run:
                    publisher.publish(payload)
                summary["events_generated"] += 1
                if summary["sample_event"] is None:
                    summary["sample_event"] = payload
            maybe_log_debug(packet, boxes, summary, prediction, args)

            if args.overlay_output:
                import cv2

                overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
                if writer is None:
                    h, w = overlay.shape[:2]
                    writer = cv2.VideoWriter(args.overlay_output, cv2.VideoWriter_fourcc(*"mp4v"), packet.fps, (w, h))
                writer.write(overlay)
            metrics.add_total_frame_ms((time.perf_counter() - frame_started_at) * 1000.0)

    if writer is not None:
        writer.release()
    summary.update(
        metrics.summary(
            summary["frames_processed"],
            summary["bbox_detections"],
            summary["keypoints_extracted"],
            summary["generated_sequences"],
            summary["lstm_predictions"],
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run safe local RTSP YOLO Pose + LSTM inference dry-run.")
    parser.add_argument("--rtsp-url", default="rtsp://localhost:8554/cam1")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overlay-output", default=None)
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.10)
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_FAINT_THRESHOLD, help="Faint probability threshold for LSTM checkpoints with Normal/Faint classes.")
    parser.add_argument("--min-consecutive-faint", type=int, default=DEFAULT_MIN_CONSECUTIVE_FAINT, help="Consecutive Faint sequences required before emitting an event.")
    parser.add_argument("--camera-cooldown-seconds", type=float, default=DEFAULT_CAMERA_COOLDOWN_SECONDS, help="Per-camera event cooldown after a Faint event.")
    parser.add_argument("--event-severity", default="HIGH")
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--resize-size", type=int, default=224)
    args = parser.parse_args()

    summary = run(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
