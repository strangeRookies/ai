import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.per_track_sequence_buffer import PerTrackCropSequenceBuffers, PerTrackKeypointSequenceBuffers
from ai.inference.rtsp_runtime import (
    build_inference_event_payload,
    ensure_mock_keypoints,
    maybe_log_debug,
    normalize_detections,
    update_prediction_counts,
    update_tracking_summary,
)
from ai.overlay_http import OverlayState, create_overlay_server
from ai.streams.video_reader import VideoReader
from ai.visualization.action_overlay import (
    annotate_boxes_with_action,
    annotate_boxes_with_track_actions,
    draw_metrics_panel,
    faint_probability,
    format_action_overlay_text,
    initial_overlay_summary,
    update_overlay_runtime,
)
from ai.visualization.draw import draw_overlay
from scripts.run_rtsp_inference import (
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_ACTION_MODEL,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    FaintEventPostProcessor,
    create_classifier,
    create_detector,
)
from stream.rtsp_reader import redact_url
from tracking.simple_tracker import SimpleTrackAssigner


def initial_summary():
    return initial_overlay_summary()


def process_frame(packet, detector, classifier, sequence_buffer, summary, args, post_processor=None, tracker=None, state=None):
    detections = detector.detect(packet.frame)
    if args.detector_mode == "mock":
        detections = ensure_mock_keypoints(detections)
    if tracker is not None:
        detections = tracker.update(detections, now=packet.timestamp)
    boxes = normalize_detections(detections)
    frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
    summary["frames_processed"] += 1
    summary["bbox_detections"] += len(boxes)
    summary["keypoints_extracted"] += frame_keypoint_count
    summary["latest_frame_bbox"] = len(boxes)
    summary["latest_frame_keypoints"] = frame_keypoint_count
    if tracker is not None:
        update_tracking_summary(summary, tracker.diagnostics())

    classifier_input = getattr(args, "classifier_input", None)
    if classifier_input == "crops":
        sequences = sequence_buffer.add(packet.frame_idx, packet.frame, boxes, now=packet.timestamp)
    else:
        sequences = sequence_buffer.add(packet.frame_idx, detections, packet.frame.shape, now=packet.timestamp)
    prediction = None
    predictions_by_track = {}
    sequences_by_track = {}
    triggered_track_ids = set()
    consecutive_by_track = {}
    for sequence in sequences:
        prediction = classifier.predict(sequence)
        track_id = sequence.get("track_id")
        if track_id is not None:
            track_id = int(track_id)
            predictions_by_track[track_id] = prediction
            sequences_by_track[track_id] = sequence
        summary["generated_sequences"] += 1
        summary["lstm_predictions"] += 1
        summary["latest_prediction_label"] = prediction.get("label")
        summary["latest_faint_probability"] = faint_probability(prediction)
        update_prediction_counts(summary, prediction)
    summary["per_track_sequences_generated"] = {
        str(track_id): count for track_id, count in sequence_buffer.sequences_generated_by_track.items()
    }
    for track_id, track_prediction in predictions_by_track.items():
        event_triggered = False
        if post_processor is not None:
            event_triggered = post_processor.should_trigger(args.camera_id, track_prediction, packet.timestamp, track_id=track_id)
            consecutive_by_track[track_id] = post_processor.consecutive_count(args.camera_id, track_id=track_id)
        elif track_prediction and track_prediction.get("label") != "Normal":
            event_triggered = True
            consecutive_by_track[track_id] = 1
        if event_triggered:
            triggered_track_ids.add(track_id)
            track_key = str(track_id)
            summary["events_generated_by_track"][track_key] = summary["events_generated_by_track"].get(track_key, 0) + 1
    summary["latest_consecutive_faint"] = max(consecutive_by_track.values(), default=0)
    annotate_boxes_with_track_actions(boxes, predictions_by_track, consecutive_by_track, triggered_track_ids, args)
    for track_id in triggered_track_ids:
        track_prediction = predictions_by_track[track_id]
        sequence = sequences_by_track[track_id]
        payload = build_inference_event_payload(args, packet, track_prediction, boxes, sequence)
        summary["events_generated"] += 1
        if summary["sample_event"] is None:
            summary["sample_event"] = payload
        if args.print_events:
            print(f"[ai-overlay-event] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        if state is not None:
            state.push_event(payload)
    maybe_log_debug(packet, boxes, summary, prediction, args, prefix="[ai-overlay-debug]")

    update_overlay_runtime(summary)
    overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
    draw_metrics_panel(overlay, summary, args, prediction)
    return overlay


class OverlayWorker:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ai-overlay-worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _run(self):
        detector = create_detector(self.args.detector_mode, self.args.yolo_model, self.args.device, self.args.imgsz, conf=self.args.detector_conf)
        classifier, _classifier_mode = create_classifier(self.args.action_model, self.args.action_device, self.args.action_threshold)
        summary = initial_summary()
        post_processor = FaintEventPostProcessor(
            min_consecutive_faint=self.args.min_consecutive_faint,
            cooldown_seconds=self.args.camera_cooldown_seconds,
        )
        tracker = SimpleTrackAssigner(
            track_thresh=self.args.track_thresh,
            match_thresh=self.args.match_thresh,
            track_buffer=self.args.track_buffer,
            min_box_area=self.args.min_box_area,
            bbox_smoothing_alpha=self.args.bbox_smoothing_alpha,
            max_missing_seconds=self.args.track_max_missing_seconds,
        )
        while not self.stop_event.is_set():
            if self.args.classifier_input == "crops":
                sequence_buffer = PerTrackCropSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    self.args.resize_size,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                )
            else:
                sequence_buffer = PerTrackKeypointSequenceBuffers(
                    self.args.sequence_length,
                    self.args.sequence_stride,
                    max_track_age_seconds=self.args.track_max_missing_seconds,
                )
            try:
                with VideoReader(self.args.rtsp_url) as reader:
                    print(f"[ai-overlay] connected: {redact_url(self.args.rtsp_url)}", flush=True)
                    while not self.stop_event.is_set():
                        packet = reader.read()
                        if packet is None:
                            break
                        overlay = process_frame(packet, detector, classifier, sequence_buffer, summary, self.args, post_processor=post_processor, tracker=tracker, state=self.state)
                        self.state.update_frame(overlay, summary)
                        if self.args.max_frames > 0 and summary["frames_processed"] >= self.args.max_frames:
                            return
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"[ai-overlay] {message}", file=sys.stderr, flush=True)
                self.state.update_error(message)
            time.sleep(self.args.reconnect_delay)


def main():
    parser = argparse.ArgumentParser(description="Serve a local MJPEG stream with AI bbox/keypoint/action overlays.")
    parser.add_argument("--rtsp-url", default="rtsp://localhost:8554/cam1")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mjpeg-fps", type=float, default=8.0)
    parser.add_argument("--detector-mode", choices=["real", "mock"], default="mock")
    parser.add_argument("--yolo-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.10)
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-device", default="auto")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_FAINT_THRESHOLD)
    parser.add_argument("--min-consecutive-faint", type=int, default=DEFAULT_MIN_CONSECUTIVE_FAINT)
    parser.add_argument("--camera-cooldown-seconds", type=float, default=DEFAULT_CAMERA_COOLDOWN_SECONDS)
    parser.add_argument("--classifier-input", choices=["keypoints", "crops"], default="keypoints")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--track-thresh", type=float, default=0.10)
    parser.add_argument("--match-thresh", "--tracker-iou-threshold", dest="match_thresh", type=float, default=0.20)
    parser.add_argument("--track-buffer", type=int, default=45)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--track-max-missing-seconds", type=float, default=3.0)
    parser.add_argument("--overlay-debug-tracks", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--debug-every-n", type=int, default=30)
    parser.add_argument("--print-events", action="store_true")
    args = parser.parse_args()

    state = OverlayState()
    worker = OverlayWorker(args, state)
    server = create_overlay_server(args.host, args.port, state, args.camera_id, args.mjpeg_fps)

    worker.start()
    print(f"[ai-overlay] serving http://{args.host}:{args.port}/stream", flush=True)
    print(f"[ai-overlay] input={redact_url(args.rtsp_url)} detector={args.detector_mode}", flush=True)

    def shutdown(_signum, _frame):
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    finally:
        worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
