import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.overlay_http import OverlayState, create_overlay_server
from ai.publishers.event_publisher import build_event_payload
from ai.streams.video_reader import VideoReader
from ai.visualization.action_overlay import (
    annotate_boxes_with_action,
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
    ensure_mock_keypoints,
    normalize_detections,
)
from stream.rtsp_reader import redact_url


def initial_summary():
    return initial_overlay_summary()


def process_frame(packet, detector, classifier, sequence_buffer, summary, args, post_processor=None):
    detections = detector.detect(packet.frame)
    if args.detector_mode == "mock":
        detections = ensure_mock_keypoints(detections)
    boxes = normalize_detections(detections)
    frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
    summary["frames_processed"] += 1
    summary["bbox_detections"] += len(boxes)
    summary["keypoints_extracted"] += frame_keypoint_count
    summary["latest_frame_bbox"] = len(boxes)
    summary["latest_frame_keypoints"] = frame_keypoint_count

    classifier_input = getattr(args, "classifier_input", None)
    if classifier_input is None and hasattr(sequence_buffer, "resize_size"):
        classifier_input = "crops"
    if classifier_input == "crops":
        sequence = sequence_buffer.add(packet.frame_idx, packet.frame, boxes)
    else:
        sequence = sequence_buffer.add(packet.frame_idx, detections, packet.frame.shape)
    prediction = classifier.predict(sequence) if sequence else None
    if sequence:
        summary["generated_sequences"] += 1
    if prediction:
        summary["lstm_predictions"] += 1
        summary["latest_prediction_label"] = prediction.get("label")
        summary["latest_faint_probability"] = faint_probability(prediction)
    event_triggered = False
    consecutive_faint = 0
    if post_processor is not None:
        event_triggered = post_processor.should_trigger(args.camera_id, prediction, packet.timestamp)
        consecutive_faint = post_processor.consecutive_count(args.camera_id)
    elif prediction and prediction.get("label") != "Normal":
        event_triggered = True
        consecutive_faint = 1
    summary["latest_consecutive_faint"] = consecutive_faint
    annotate_boxes_with_action(boxes, prediction, args, consecutive_faint, event_triggered)
    if event_triggered:
        payload = build_event_payload(
            camera_id=args.camera_id,
            frame_idx=packet.frame_idx,
            timestamp=packet.timestamp,
            event_type=prediction["label"],
            score=prediction["score"],
            boxes=boxes,
            snapshot_path=None,
        )
        payload["sequence_window"] = {"start": sequence["start_frame"], "end": sequence["end_frame"]}
        summary["events_generated"] += 1
        if summary["sample_event"] is None:
            summary["sample_event"] = payload
        if args.print_events:
            print(f"[ai-overlay-event] {json.dumps(payload, ensure_ascii=False)}", flush=True)
    maybe_log_debug(packet, boxes, summary, prediction, args)

    update_overlay_runtime(summary)
    overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
    draw_metrics_panel(overlay, summary, args, prediction)
    return overlay


def maybe_log_debug(packet, boxes, summary, prediction, args):
    every_n = max(0, int(getattr(args, "debug_every_n", 30)))
    missing_detection = len(boxes) == 0
    should_log = missing_detection or (every_n > 0 and summary["frames_processed"] % every_n == 0)
    if not should_log:
        return
    faint_prob = faint_probability(prediction)
    faint_text = "None" if faint_prob is None else f"{faint_prob:.4f}"
    print(
        "[ai-overlay-debug] "
        f"frame={packet.frame_idx} "
        f"bbox={len(boxes)} "
        f"keypoints={summary.get('latest_frame_keypoints', 0)} "
        f"seq={summary['generated_sequences']} "
        f"pred={summary['lstm_predictions']} "
        f"latest_faint_prob={faint_text}",
        flush=True,
    )


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
        while not self.stop_event.is_set():
            if self.args.classifier_input == "crops":
                sequence_buffer = CropSequenceBuffer(self.args.sequence_length, self.args.sequence_stride, self.args.resize_size)
            else:
                sequence_buffer = KeypointSequenceBuffer(self.args.sequence_length, self.args.sequence_stride)
            try:
                with VideoReader(self.args.rtsp_url) as reader:
                    print(f"[ai-overlay] connected: {redact_url(self.args.rtsp_url)}", flush=True)
                    while not self.stop_event.is_set():
                        packet = reader.read()
                        if packet is None:
                            break
                        overlay = process_frame(packet, detector, classifier, sequence_buffer, summary, self.args, post_processor=post_processor)
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
