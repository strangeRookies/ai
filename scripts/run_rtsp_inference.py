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
from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.inference.rtsp_runtime import (
    build_inference_event_payload,
    ensure_mock_keypoints,
    maybe_log_debug,
    normalize_detections,
    update_prediction_counts,
)
from ai.publishers.event_publisher import ConsoleEventPublisher
from ai.runtime_metrics import RuntimeMetrics
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from detector.mock_detector import MockDetector
from detector.yolo_pose_detector import YoloPoseDetector
from tracking.simple_tracker import SimpleTrackAssigner


def create_detector(mode, model, device, imgsz=640, conf=0.25):
    if mode == "mock":
        return MockDetector(model_name="mock-pose-detector")
    return YoloPoseDetector(model, device=device, imgsz=imgsz, conf=conf)


def create_classifier(action_model, device, action_threshold=DEFAULT_FAINT_THRESHOLD):
    if action_model:
        return LSTMActionClassifier(action_model, device=device, faint_threshold=action_threshold), "lstm_checkpoint"
    return MockActionClassifier(default_label="Faint", score=0.80), "mock_lstm"


def run(args):
    detector = create_detector(args.detector_mode, args.yolo_model, args.device, getattr(args, "imgsz", 640), conf=getattr(args, "detector_conf", 0.25))
    classifier, classifier_mode = create_classifier(args.action_model, args.action_device, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD))
    classifier_input = getattr(args, "classifier_input", "keypoints")
    tracker = SimpleTrackAssigner(
        iou_threshold=getattr(args, "tracker_iou_threshold", 0.3),
        max_missing_seconds=getattr(args, "track_max_missing_seconds", 2.0),
    )
    keypoint_buffers = PerTrackKeypointSequenceBuffers(
        args.sequence_length,
        args.sequence_stride,
        max_track_age_seconds=getattr(args, "track_max_missing_seconds", 2.0),
    )
    crop_buffers = (
        PerTrackCropSequenceBuffers(
            args.sequence_length,
            args.sequence_stride,
            args.resize_size,
            max_track_age_seconds=getattr(args, "track_max_missing_seconds", 2.0),
        )
        if args.action_model and classifier_input == "crops"
        else None
    )
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
        "active_tracks": 0,
        "max_active_tracks": 0,
        "per_track_sequences_generated": {},
        "faint_predictions": 0,
        "normal_predictions": 0,
        "events_generated_by_track": {},
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
            detections = tracker.update(detections, now=packet.timestamp)
            boxes = normalize_detections(detections)
            frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
            active_tracks = len({int(item["track_id"]) for item in detections if item.get("track_id") is not None})
            metrics.observe_active_tracks(active_tracks)
            summary["frames_processed"] += 1
            summary["bbox_detections"] += len(boxes)
            summary["keypoints_extracted"] += frame_keypoint_count
            summary["latest_frame_keypoints"] = frame_keypoint_count
            summary["active_tracks"] = active_tracks
            summary["max_active_tracks"] = max(summary["max_active_tracks"], active_tracks)

            keypoint_sequences = keypoint_buffers.add(packet.frame_idx, detections, packet.frame.shape, now=packet.timestamp)
            crop_sequences = crop_buffers.add(packet.frame_idx, packet.frame, boxes, now=packet.timestamp) if crop_buffers else []
            classifier_sequences = crop_sequences if crop_sequences else keypoint_sequences
            prediction = None
            predictions_by_track = {}
            sequences_by_track = {}
            for classifier_sequence in classifier_sequences:
                lstm_started_at = time.perf_counter()
                prediction = classifier.predict(classifier_sequence)
                metrics.add_lstm_ms((time.perf_counter() - lstm_started_at) * 1000.0)
                track_id = classifier_sequence.get("track_id")
                if track_id is not None:
                    predictions_by_track[int(track_id)] = prediction
                    sequences_by_track[int(track_id)] = classifier_sequence
                summary["generated_sequences"] += 1
                summary["lstm_predictions"] += 1
                summary["latest_prediction_label"] = prediction.get("label")
                summary["latest_faint_probability"] = faint_probability(prediction)
                update_prediction_counts(summary, prediction)
            summary["per_track_sequences_generated"] = {
                str(track_id): count for track_id, count in keypoint_buffers.sequences_generated_by_track.items()
            }
            if crop_buffers:
                summary["per_track_sequences_generated"].update(
                    {str(track_id): count for track_id, count in crop_buffers.sequences_generated_by_track.items()}
                )
            for track_id, track_prediction in predictions_by_track.items():
                if post_processor.should_trigger(args.camera_id, track_prediction, packet.timestamp, track_id=track_id):
                    payload = build_inference_event_payload(args, packet, track_prediction, boxes, sequences_by_track.get(track_id))
                    if args.dry_run:
                        publisher.publish(payload)
                    summary["events_generated"] += 1
                    track_key = str(track_id)
                    summary["events_generated_by_track"][track_key] = summary["events_generated_by_track"].get(track_key, 0) + 1
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
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.3)
    parser.add_argument("--track-max-missing-seconds", type=float, default=2.0)
    args = parser.parse_args()

    summary = run(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
