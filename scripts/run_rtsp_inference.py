import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.classifier import LSTMActionClassifier, MockActionClassifier
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.publishers.event_publisher import ConsoleEventPublisher, build_event_payload
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector


def create_detector(mode, model, device, imgsz=640):
    if mode == "mock":
        return MockDetector(model_name="mock-pose-detector")
    return YoloPoseDetector(model, device=device, imgsz=imgsz)


def create_classifier(action_model, device, action_threshold=0.5):
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


def is_alert_prediction(prediction):
    return bool(prediction) and prediction.get("label") != "Normal"


def run(args):
    detector = create_detector(args.detector_mode, args.yolo_model, args.device, getattr(args, "imgsz", 640))
    classifier, classifier_mode = create_classifier(args.action_model, args.action_device, getattr(args, "action_threshold", 0.5))
    keypoint_buffer = KeypointSequenceBuffer(args.sequence_length, args.sequence_stride)
    classifier_input = getattr(args, "classifier_input", "keypoints")
    crop_buffer = CropSequenceBuffer(args.sequence_length, args.sequence_stride, args.resize_size) if args.action_model and classifier_input == "crops" else None
    publisher = ConsoleEventPublisher()
    writer = None
    summary = {
        "rtsp_url": args.rtsp_url,
        "camera_id": args.camera_id,
        "dry_run": args.dry_run,
        "detector_mode": args.detector_mode,
        "yolo_model": args.yolo_model if args.detector_mode == "real" else None,
        "classifier_mode": classifier_mode,
        "classifier_input": classifier_input,
        "action_threshold": getattr(args, "action_threshold", 0.5),
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
            packet = reader.read()
            if packet is None:
                break
            if args.max_frames > 0 and summary["frames_processed"] >= args.max_frames:
                break

            detections = detector.detect(packet.frame)
            if args.detector_mode == "mock":
                detections = ensure_mock_keypoints(detections)
            boxes = normalize_detections(detections)
            summary["frames_processed"] += 1
            summary["bbox_detections"] += len(boxes)
            summary["keypoints_extracted"] += sum(1 for item in detections if item.get("keypoints"))

            keypoint_sequence = keypoint_buffer.add(packet.frame_idx, detections, packet.frame.shape)
            crop_sequence = crop_buffer.add(packet.frame_idx, packet.frame, boxes) if crop_buffer else None
            classifier_sequence = crop_sequence if crop_sequence else keypoint_sequence
            prediction = classifier.predict(classifier_sequence) if classifier_sequence else None
            if keypoint_sequence:
                summary["generated_sequences"] += 1
            if prediction:
                summary["lstm_predictions"] += 1
            if is_alert_prediction(prediction):
                payload = build_event_payload(
                    camera_id=args.camera_id,
                    frame_idx=packet.frame_idx,
                    timestamp=packet.timestamp,
                    event_type=prediction["label"],
                    score=prediction["score"],
                    boxes=boxes,
                    snapshot_path=None,
                )
                sequence_for_event = keypoint_sequence or crop_sequence
                payload["bbox"] = sequence_for_event.get("bbox") if sequence_for_event else None
                payload["confidence"] = prediction["score"]
                payload["sequence_window"] = {"start": sequence_for_event["start_frame"], "end": sequence_for_event["end_frame"]}
                if args.dry_run:
                    publisher.publish(payload)
                summary["events_generated"] += 1
                if summary["sample_event"] is None:
                    summary["sample_event"] = payload

            if args.overlay_output:
                import cv2

                overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
                if writer is None:
                    h, w = overlay.shape[:2]
                    writer = cv2.VideoWriter(args.overlay_output, cv2.VideoWriter_fourcc(*"mp4v"), packet.fps, (w, h))
                writer.write(overlay)

    if writer is not None:
        writer.release()
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
    parser.add_argument("--action-model", default=None)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=0.5, help="Faint probability threshold for LSTM checkpoints with Normal/Faint classes.")
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
