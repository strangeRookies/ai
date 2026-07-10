import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ai.action.faint_post_processing import (
    DEFAULT_ACTION_MODEL,
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    FaintEventPostProcessor,
    faint_probability,
    is_alert_prediction,
)
from ai.action.fall_lifecycle_config import (
    build_faint_post_processor_from_args,
    resolved_faint_threshold,
)
from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.action.lstm_contract import DEFAULT_KEYPOINT_INPUT_SIZE, log_lstm_config
from ai.evidence import evidence_id
from ai.inference.rtsp_runtime import (
    build_inference_event_log,
    build_inference_event_payload,
    lifecycle_payload_kwargs,
    cheap_filter_config_from_args,
    create_classifier,
    create_detection_postprocessor,
    create_detector,
    create_pose_diagnostics_reporter,
    ensure_mock_keypoints,
    log_classifier_contract,
    log_pose_tracking_config,
    maybe_log_debug,
    normalize_detections,
    save_inference_event_log,
    update_detections_with_postprocessor,
    update_prediction_counts,
    update_tracking_summary,
)
from ai.inference.tensorrt_runtime import (
    attach_runtime_summary_fields,
    finalize_runtime_summary_fields,
    log_periodic_inference_metrics,
    log_worker_backend_startup,
)
from ai.inference.tracking_debug import (
    build_frame_tracking_record,
    log_frame_tracking_debug,
    log_track_lifecycle_events,
    log_tracker_startup,
)
from ai.evaluation.prediction_log import append_prediction_jsonl, build_prediction_log_row
import threading
from ai.frame_sync import FrameMetadataBuffer, FramePacket, CameraFrameQueue
from ai.publishers.event_publisher import create_event_publisher, mqtt_topic_settings_from_args
from ai.runtime_metrics import RuntimeMetrics
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from scripts.rtsp_inference_args import parse_args


def run(args):
    # Prefer explicit faint_threshold over action_threshold for classifier gating.
    action_threshold = resolved_faint_threshold(args)
    detector = create_detector(args.detector_mode, args.yolo_model, args.device, getattr(args, "imgsz", 640), conf=getattr(args, "detector_conf", 0.25))
    classifier, classifier_mode = create_classifier(args.action_model, args.action_device, action_threshold)
    camera_login_id = getattr(args, "camera_login_id", None) or args.camera_id
    classifier_input = getattr(args, "classifier_input", "keypoints")
    detection_postprocessor, postprocessing_mode = create_detection_postprocessor(args)
    pose_reporter = create_pose_diagnostics_reporter(args)
    log_pose_tracking_config(args, detection_postprocessor)
    log_tracker_startup(camera_login_id, postprocessing_mode, detection_postprocessor, args)
    cheap_filter_config = cheap_filter_config_from_args(args)
    keypoint_buffers = PerTrackKeypointSequenceBuffers(
        args.sequence_length,
        args.sequence_stride,
        max_track_age_seconds=getattr(args, "track_max_missing_seconds", 4.0),
        cheap_filter_config=cheap_filter_config,
        missing_track_grace_seconds=getattr(args, "tracking_grace_period_seconds", getattr(args, "track_max_missing_seconds", 4.0)),
        relink_iou_threshold=getattr(args, "tracking_relink_iou_threshold", 0.30),
        relink_center_distance_ratio=getattr(args, "tracking_relink_center_ratio", 0.70),
        relink_max_time_gap_seconds=getattr(args, "tracking_relink_max_time_gap_seconds", 2.0),
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
    post_processor = build_faint_post_processor_from_args(args)
    log_worker_backend_startup(
        camera_login_id=getattr(args, "camera_login_id", None) or args.camera_id,
        requested_model=args.yolo_model,
        detector=detector,
        device=getattr(args, "device", None),
    )
    publisher = None
    publisher_mode = "preflight" if getattr(args, "preflight_only", False) else None
    topic_settings = mqtt_topic_settings_from_args(args)
    metrics = RuntimeMetrics()
    metrics.set_warmup_skip(int(getattr(args, "infer_warmup_frames", 20)))
    frame_buffer = FrameMetadataBuffer(maxlen=int(getattr(args, "frame_sync_buffer_size", 60)))
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
        "lstm_input_size": getattr(classifier, "input_size", DEFAULT_KEYPOINT_INPUT_SIZE),
        "sequence_config_note": "runtime CLI/env values override buffer class defaults; stride is sequence start interval, not FPS sampling",
        "cheap_filter_enabled": cheap_filter_config.enabled,
        "cheap_filter_sequences_kept": 0,
        "cheap_filter_sequences_skipped": 0,
        "cheap_filter_reasons": {},
        "latest_faint_probability": None,
        "latest_prediction_label": None,
        "latest_frame_keypoints": 0,
        "active_tracks": 0,
        "max_active_tracks": 0,
        "new_tracks": 0,
        "lost_tracks": 0,
        "id_switch_like_events": 0,
        "track_diagnostics": {},
        "pose_diagnostics": {},
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
    attach_runtime_summary_fields(summary, detector, requested_model=args.yolo_model)

    if getattr(args, "preflight_only", False):
        print(
            "[rtsp-inference] sequence config: "
            f"sequence_length={args.sequence_length} "
            f"sequence_stride={args.sequence_stride} "
            "frame_sampling=disabled",
            flush=True,
        )
        log_lstm_config(
            "[lstm-config]",
            args.sequence_length,
            args.sequence_stride,
            getattr(classifier, "input_size", DEFAULT_KEYPOINT_INPUT_SIZE),
            f"checkpoint/config/cli:{classifier_mode}",
            getattr(classifier, "checkpoint_sequence_length", None),
            getattr(classifier, "checkpoint_sequence_stride", None),
        )
        log_classifier_contract("[lstm-checkpoint]", camera_login_id, classifier)
        return summary

    publisher, publisher_mode = create_event_publisher(args)
    summary["alert_delivery_result"] = publisher_mode
    print(
        "[rtsp-inference] sequence config: "
        f"sequence_length={args.sequence_length} "
        f"sequence_stride={args.sequence_stride} "
        "frame_sampling=disabled",
        flush=True,
    )
    log_lstm_config(
        "[lstm-config]",
        args.sequence_length,
        args.sequence_stride,
        getattr(classifier, "input_size", DEFAULT_KEYPOINT_INPUT_SIZE),
        f"checkpoint/config/cli:{classifier_mode}",
        getattr(classifier, "checkpoint_sequence_length", None),
        getattr(classifier, "checkpoint_sequence_stride", None),
    )
    log_classifier_contract("[lstm-checkpoint]", camera_login_id, classifier)

    writer = None
    rtsp_url_str = str(getattr(args, "rtsp_url", "") or "")
    is_offline_video = rtsp_url_str.endswith((".mp4", ".avi", ".mkv", ".mov")) or args.detector_mode == "mock" or getattr(args, "max_frames", 0) > 0
    max_q_size = 10000 if is_offline_video else getattr(args, "frame_queue_maxsize", 3)
    queue = CameraFrameQueue(camera_login_id, maxsize=max_q_size)
    stop_event = threading.Event()
    reader_exited = threading.Event()

    def reader_thread_fn():
        try:
            with VideoReader(args.rtsp_url) as reader:
                while not stop_event.is_set():
                    read_started_at = time.perf_counter()
                    packet = reader.read()
                    metrics.add_read_ms((time.perf_counter() - read_started_at) * 1000.0)
                    if packet is None:
                        break
                    
                    frame_metadata = frame_buffer.record_capture(camera_login_id, packet, packet.frame.shape)
                    packet_wrapped = FramePacket(
                        camera_login_id=camera_login_id,
                        frame_id=frame_metadata.frame_id,
                        captured_at_ms=frame_metadata.captured_at_ms,
                        frame=packet.frame,
                        width=frame_metadata.width,
                        height=frame_metadata.height,
                        frame_idx=packet.frame_idx,
                        timestamp=packet.timestamp,
                        fps=getattr(packet, "fps", 0.0)
                    )
                    queue.put_latest(packet_wrapped)
                    
                    every_n = max(0, int(getattr(args, "debug_every_n", 30)))
                    if every_n > 0 and frame_metadata.frame_id % every_n == 0:
                        now_ms = time.time_ns() // 1_000_000
                        lag = now_ms - frame_metadata.captured_at_ms
                        print(
                            f"[rtsp-buffer] {camera_login_id} "
                            f"frame_id={frame_metadata.frame_id} "
                            f"queue_lag_ms={lag} "
                            f"dropped={queue.dropped_frame_count}",
                            flush=True
                        )
        except Exception as exc:
            print(f"[rtsp-inference-reader] error: {exc}", file=sys.stderr, flush=True)
        finally:
            reader_exited.set()

    reader_thread = threading.Thread(target=reader_thread_fn, name="rtsp-reader", daemon=True)
    reader_thread.start()

    try:
        previous_processed_frame_id = None
        while True:
            if args.max_frames > 0 and summary["frames_processed"] >= args.max_frames:
                break
            
            drop_stale = not is_offline_video
            frame_packet = queue.get_latest(drop_stale=drop_stale)
            if frame_packet is None:
                if reader_exited.is_set() and queue.size() == 0:
                    break
                time.sleep(0.005)
                continue
                
            frame_started_at = time.perf_counter()
            frame_metadata = frame_buffer.get_by_frame_id(camera_login_id, frame_packet.frame_id)
            if frame_metadata is not None:
                summary["latest_frame_id"] = frame_metadata.frame_id
                summary["latest_captured_at_ms"] = frame_metadata.captured_at_ms
                summary["frame_sync_buffer_size"] = frame_buffer.size(camera_login_id)

            yolo_started_at = time.perf_counter()
            detections = detector.detect(frame_packet.frame)
            metrics.add_yolo_ms((time.perf_counter() - yolo_started_at) * 1000.0)
            if args.detector_mode == "mock":
                detections = ensure_mock_keypoints(detections)
            raw_detections = [dict(item) for item in detections]
            detections = update_detections_with_postprocessor(
                detection_postprocessor,
                detections,
                frame_packet.frame,
                frame_packet.timestamp,
            )
            boxes = normalize_detections(detections)
            frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
            tracker_diagnostics = detection_postprocessor.diagnostics()
            update_tracking_summary(summary, tracker_diagnostics)
            log_track_lifecycle_events(
                camera_login_id,
                frame_packet.frame_id if frame_metadata is None else frame_metadata.frame_id,
                tracker_diagnostics,
            )
            metrics.observe_active_tracks(summary["active_tracks"])
            summary["frames_processed"] += 1
            summary["bbox_detections"] += len(boxes)
            summary["keypoints_extracted"] += frame_keypoint_count
            summary["latest_frame_keypoints"] = frame_keypoint_count
            metrics_every = max(0, int(getattr(args, "infer_metrics_every_n", 30)))
            if metrics_every > 0 and summary["frames_processed"] % metrics_every == 0:
                log_periodic_inference_metrics(
                    camera_login_id=camera_login_id,
                    backend=getattr(detector, "runtime", summary.get("backend")),
                    metrics=metrics,
                    frames_processed=summary["frames_processed"],
                )

            keypoint_sequences = keypoint_buffers.add(
                frame_packet.frame_idx,
                detections,
                frame_packet.frame.shape,
                now=frame_packet.timestamp,
                frame_id=frame_metadata.frame_id if frame_metadata else None,
                captured_at_ms=frame_metadata.captured_at_ms if frame_metadata else None,
            )
            summary["cheap_filter_sequences_kept"] = keypoint_buffers.sequences_kept_by_filter
            summary["cheap_filter_sequences_skipped"] = keypoint_buffers.sequences_skipped_by_filter
            summary["cheap_filter_reasons"] = dict(keypoint_buffers.cheap_filter_reasons)
            crop_sequences = (
                crop_buffers.add(
                    frame_packet.frame_idx,
                    frame_packet.frame,
                    boxes,
                    now=frame_packet.timestamp,
                    frame_id=frame_metadata.frame_id if frame_metadata else None,
                    captured_at_ms=frame_metadata.captured_at_ms if frame_metadata else None,
                )
                if crop_buffers
                else []
            )
            classifier_sequences = crop_sequences if crop_sequences else keypoint_sequences
            pose_record = pose_reporter.observe(
                camera_login_id=camera_login_id,
                source_url=str(args.rtsp_url),
                assigned_video_path=str(getattr(args, "assigned_video_path", "") or getattr(args, "video_id", "") or ""),
                frame_id=frame_metadata.frame_id if frame_metadata else getattr(frame_packet, "frame_id", None),
                timestamp_ms=frame_metadata.captured_at_ms if frame_metadata else getattr(frame_packet, "captured_at_ms", None),
                raw_detections=raw_detections,
                tracker_diagnostics=detection_postprocessor.diagnostics(),
                sequence_ready_count=len(classifier_sequences),
                sequence_diagnostics={
                    "relink_success_count": keypoint_buffers.relink_success_count,
                    "relink_fail_count": keypoint_buffers.relink_failure_count,
                },
                frame=frame_packet.frame,
            )
            summary["pose_diagnostics"] = pose_record
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
            if frame_metadata is not None:
                frame_metadata = frame_buffer.mark_processed(camera_login_id, frame_metadata.frame_id)
                summary["latest_processed_at_ms"] = frame_metadata.processed_at_ms
                summary["latest_ai_latency_ms"] = frame_metadata.ai_latency_ms
                frame_gap = (
                    None
                    if previous_processed_frame_id is None
                    else int(frame_metadata.frame_id) - int(previous_processed_frame_id)
                )
                previous_processed_frame_id = frame_metadata.frame_id
                log_frame_tracking_debug(
                    build_frame_tracking_record(
                        camera_login_id=camera_login_id,
                        frame_id=frame_metadata.frame_id,
                        captured_at_ms=frame_metadata.captured_at_ms,
                        processed_at_ms=frame_metadata.processed_at_ms,
                        frame_gap=frame_gap,
                        dropped_frame_count=queue.dropped_frame_count,
                        raw_detections=raw_detections,
                        tracked_detections=detections,
                        tracker_diagnostics=detection_postprocessor.diagnostics(),
                        tracker_object_id=id(detection_postprocessor),
                    )
                )
            from ai.action.posture_estimator import detection_for_track

            post_processor.prune_lost_tracks(
                args.camera_id,
                list(predictions_by_track.keys()),
                frame_packet.timestamp,
            )
            for track_id, track_prediction in predictions_by_track.items():
                cooldown_was_active = post_processor.cooldown_active(args.camera_id, frame_packet.timestamp, track_id=track_id)
                track_detection = detection_for_track(detections, track_id) or detection_for_track(boxes, track_id)
                emit_decision = post_processor.evaluate(
                    args.camera_id,
                    track_prediction,
                    frame_packet.timestamp,
                    track_id=track_id,
                    detection=track_detection,
                )
                event_emitted = bool(emit_decision.emit)
                sequence = sequences_by_track.get(track_id)
                if getattr(args, "evaluation_log", None):
                    append_prediction_jsonl(
                        args.evaluation_log,
                        build_prediction_log_row(
                            args,
                            frame_packet,
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
                    if frame_metadata is not None:
                        frame_metadata = frame_buffer.mark_published(camera_login_id, frame_metadata.frame_id)
                        summary["latest_published_at_ms"] = frame_metadata.published_at_ms
                        summary["latest_publish_latency_ms"] = frame_metadata.publish_latency_ms
                        evidence_key = evidence_id(
                            camera_login_id,
                            frame_metadata.frame_id,
                            frame_metadata.captured_at_ms,
                        )
                        summary["latest_evidence_id"] = evidence_key
                        summary["latest_trace_id"] = evidence_key
                        summary["latest_dropped_frame_count"] = queue.dropped_frame_count
                        summary["latest_latency_order_valid"] = frame_metadata.latency_order_valid
                        if not frame_metadata.latency_order_valid:
                            print(
                                "[frame-sync] warning "
                                f"{camera_login_id} "
                                f"latency_order_invalid=true "
                                f"frame_id={frame_metadata.frame_id} "
                                f"captured_at_ms={frame_metadata.captured_at_ms} "
                                f"processed_at_ms={frame_metadata.processed_at_ms} "
                                f"published_at_ms={frame_metadata.published_at_ms}",
                                flush=True,
                            )
                    payload = build_inference_event_payload(
                        args,
                        frame_packet,
                        track_prediction,
                        boxes,
                        sequence,
                        frame_metadata=frame_metadata,
                        published_at_ms=frame_metadata.published_at_ms if frame_metadata else None,
                        dropped_frame_count=queue.dropped_frame_count,
                        **lifecycle_payload_kwargs(emit_decision),
                    )
                    event_log = build_inference_event_log(args, frame_packet, track_prediction, boxes, sequence)
                    if getattr(args, "event_log_dir", None):
                        save_inference_event_log(args.event_log_dir, event_log)
                    publisher.publish(payload, topic=topic_settings["event_topic"])
                    summary["events_generated"] += 1
                    if emit_decision.is_unrecovered:
                        summary["unrecovered_events_generated"] = int(summary.get("unrecovered_events_generated", 0)) + 1
                    track_key = str(track_id)
                    summary["events_generated_by_track"][track_key] = summary["events_generated_by_track"].get(track_key, 0) + 1
                    if summary["sample_event"] is None:
                        summary["sample_event"] = payload
            maybe_log_debug(frame_packet, boxes, summary, prediction, args)

            if args.overlay_output:
                import cv2

                overlay = draw_overlay(frame_packet.frame, boxes, prediction, frame_packet.frame_idx)
                if writer is None:
                    h, w = overlay.shape[:2]
                    writer = cv2.VideoWriter(args.overlay_output, cv2.VideoWriter_fourcc(*"mp4v"), frame_packet.fps, (w, h))
                writer.write(overlay)
            metrics.add_total_frame_ms((time.perf_counter() - frame_started_at) * 1000.0)
            every_n = max(0, int(getattr(args, "debug_every_n", 30)))
            if every_n > 0 and summary["frames_processed"] % every_n == 0:
                partial = metrics.summary(
                    summary["frames_processed"],
                    summary["bbox_detections"],
                    summary["keypoints_extracted"],
                    summary["generated_sequences"],
                    summary["lstm_predictions"],
                )
                print(
                    "[rtsp-inference-status] "
                    f"frames_processed={summary['frames_processed']} "
                    f"effective_processing_fps={partial['effective_fps']} "
                    f"inference_latency_ms={partial['avg_lstm_inference_ms']} "
                    f"queue_drop_count={queue.dropped_frame_count}",
                    flush=True,
                )
    finally:
        stop_event.set()
        reader_thread.join(timeout=3)
        close = getattr(publisher, "close", None) if publisher is not None else None
        if close:
            close()
        if writer is not None:
            writer.release()
        pose_reporter.log_final_summary()
    summary.update(
        metrics.summary(
            summary["frames_processed"],
            summary["bbox_detections"],
            summary["keypoints_extracted"],
            summary["generated_sequences"],
            summary["lstm_predictions"],
        )
    )
    finalize_runtime_summary_fields(summary)
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
