import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    build_inference_event_log,
    build_inference_event_payload,
    create_classifier,
    create_detection_postprocessor,
    create_detector,
    ensure_mock_keypoints,
    maybe_log_debug,
    normalize_detections,
    save_inference_event_log,
    update_detections_with_postprocessor,
    update_prediction_counts,
    update_tracking_summary,
)
from ai.evaluation.prediction_log import append_prediction_jsonl, build_prediction_log_row
from ai.publishers.event_publisher import create_event_publisher
from ai.runtime_metrics import RuntimeMetrics
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from scripts.rtsp_inference_args import parse_args


def run(args):
    detector = create_detector(args.detector_mode, args.yolo_model, args.device, getattr(args, "imgsz", 640), conf=getattr(args, "detector_conf", 0.25))
    classifier, classifier_mode = create_classifier(args.action_model, args.action_device, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD))
    classifier_input = getattr(args, "classifier_input", "keypoints")
    detection_postprocessor, postprocessing_mode = create_detection_postprocessor(args)
    keypoint_buffers = PerTrackKeypointSequenceBuffers(
        args.sequence_length,
        args.sequence_stride,
        max_track_age_seconds=getattr(args, "track_max_missing_seconds", 4.0),
    )
    crop_buffers = (
        PerTrackCropSequenceBuffers(
            args.sequence_length,
            args.sequence_stride,
            args.resize_size,
            max_track_age_seconds=getattr(args, "track_max_missing_seconds", 4.0),
        )
        if args.action_model and classifier_input == "crops"
        else None
    )
    post_processor = FaintEventPostProcessor(
        min_consecutive_faint=getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        cooldown_seconds=getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
    )
    publisher = None
    publisher_mode = "preflight" if getattr(args, "preflight_only", False) else None
    metrics = RuntimeMetrics()
    writer = None
    summary = {
        "rtsp_url": args.rtsp_url,
        "source_id": getattr(args, "source_id", None),
        "video_id": getattr(args, "video_id", None),
        "camera_id": args.camera_id,
        "dry_run": args.dry_run,
        "detector_mode": args.detector_mode,
        "yolo_model": args.yolo_model,
        "yolo_model_exists": Path(args.yolo_model).exists(),
        "action_model": args.action_model,
        "action_model_exists": Path(args.action_model).exists() if args.action_model else False,
        "classifier_mode": classifier_mode,
        "classifier_input": classifier_input,
        "postprocessing_mode": postprocessing_mode,
        "action_threshold": getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD),
        "min_consecutive_faint": getattr(args, "min_consecutive_faint", DEFAULT_MIN_CONSECUTIVE_FAINT),
        "camera_cooldown_seconds": getattr(args, "camera_cooldown_seconds", DEFAULT_CAMERA_COOLDOWN_SECONDS),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "latest_faint_probability": None,
        "latest_prediction_label": None,
        "latest_frame_keypoints": 0,
        "active_tracks": 0,
        "max_active_tracks": 0,
        "new_tracks": 0,
        "lost_tracks": 0,
        "id_switch_like_events": 0,
        "track_diagnostics": {},
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
        "alert_delivery_result": publisher_mode,
    }

    if getattr(args, "preflight_only", False):
        return summary

    publisher, publisher_mode = create_event_publisher(args)
    summary["alert_delivery_result"] = publisher_mode

    try:
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
                detections = update_detections_with_postprocessor(
                    detection_postprocessor,
                    detections,
                    packet.frame,
                    packet.timestamp,
                )
                boxes = normalize_detections(detections)
                frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
                update_tracking_summary(summary, detection_postprocessor.diagnostics())
                metrics.observe_active_tracks(summary["active_tracks"])
                summary["frames_processed"] += 1
                summary["bbox_detections"] += len(boxes)
                summary["keypoints_extracted"] += frame_keypoint_count
                summary["latest_frame_keypoints"] = frame_keypoint_count

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
                    cooldown_was_active = post_processor.cooldown_active(args.camera_id, packet.timestamp, track_id=track_id)
                    event_emitted = post_processor.should_trigger(args.camera_id, track_prediction, packet.timestamp, track_id=track_id)
                    sequence = sequences_by_track.get(track_id)
                    if getattr(args, "evaluation_log", None):
                        append_prediction_jsonl(
                            args.evaluation_log,
                            build_prediction_log_row(
                                args,
                                packet,
                                sequence,
                                track_prediction,
                                post_processor.consecutive_count(args.camera_id, track_id=track_id),
                                cooldown_was_active,
                                event_emitted,
                                ground_truth=getattr(args, "ground_truth", None),
                            ),
                        )
                    if event_emitted:
                        sequence = sequences_by_track.get(track_id)
                        payload = build_inference_event_payload(
                            args,
                            packet,
                            track_prediction,
                            boxes,
                            sequence,
                        )
                        event_log = build_inference_event_log(args, packet, track_prediction, boxes, sequence)
                        if getattr(args, "event_log_dir", None):
                            save_inference_event_log(args.event_log_dir, event_log)
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
    finally:
        close = getattr(publisher, "close", None) if publisher is not None else None
        if close:
            close()

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


def write_run_summary(output_path, summary):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()

    summary = run(args)
    if args.output:
        write_run_summary(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
