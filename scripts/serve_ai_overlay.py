import argparse
import json
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.publishers.event_publisher import build_event_payload
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay
from scripts.run_rtsp_inference import (
    DEFAULT_CAMERA_COOLDOWN_SECONDS,
    DEFAULT_FAINT_THRESHOLD,
    DEFAULT_MIN_CONSECUTIVE_FAINT,
    FaintEventPostProcessor,
    create_classifier,
    create_detector,
    ensure_mock_keypoints,
    normalize_detections,
)
from stream.rtsp_reader import redact_url


class OverlayState:
    def __init__(self):
        self.frame = None
        self.summary = initial_summary()
        self.connected = False
        self.last_error = ""
        self.lock = threading.Lock()

    def update_frame(self, frame, summary):
        with self.lock:
            self.frame = frame
            self.summary = dict(summary)
            self.connected = True
            self.last_error = ""

    def update_error(self, message):
        with self.lock:
            self.connected = False
            self.last_error = message

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "last_error": self.last_error,
                "summary": dict(self.summary),
            }


def initial_summary():
    return {
        "frames_processed": 0,
        "bbox_detections": 0,
        "keypoints_extracted": 0,
        "generated_sequences": 0,
        "lstm_predictions": 0,
        "events_generated": 0,
        "latest_faint_probability": None,
        "latest_prediction_label": None,
        "latest_consecutive_faint": 0,
        "latest_frame_keypoints": 0,
        "sample_event": None,
    }


def process_frame(packet, detector, classifier, sequence_buffer, summary, args, post_processor=None):
    detections = detector.detect(packet.frame)
    if args.detector_mode == "mock":
        detections = ensure_mock_keypoints(detections)
    boxes = normalize_detections(detections)
    frame_keypoint_count = sum(1 for item in detections if item.get("keypoints"))
    summary["frames_processed"] += 1
    summary["bbox_detections"] += len(boxes)
    summary["keypoints_extracted"] += frame_keypoint_count
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

    overlay = draw_overlay(packet.frame, boxes, prediction, packet.frame_idx)
    draw_metrics_panel(overlay, summary, args, prediction)
    return overlay


def faint_probability(prediction):
    if not prediction:
        return None
    probabilities = prediction.get("probabilities") or {}
    if "Faint" in probabilities:
        return float(probabilities["Faint"])
    if prediction.get("label") == "Faint":
        return float(prediction.get("score", 0.0))
    return None


def format_action_overlay_text(prediction, threshold, consecutive_faint=0, event_triggered=False):
    faint_prob = faint_probability(prediction)
    if faint_prob is None:
        return f"pred: waiting | th: {float(threshold):.2f}"
    label = prediction.get("label", "unknown")
    if event_triggered:
        return f"ALERT Faint: {faint_prob:.2f}"
    return f"Faint: {faint_prob:.2f} | pred: {label} | th: {float(threshold):.2f} | seq: {int(consecutive_faint)}"


def annotate_boxes_with_action(boxes, prediction, args, consecutive_faint=0, event_triggered=False):
    text = format_action_overlay_text(prediction, getattr(args, "action_threshold", DEFAULT_FAINT_THRESHOLD), consecutive_faint, event_triggered)
    for box in boxes:
        box["action_overlay"] = text
        box["event_triggered"] = bool(event_triggered)
    return boxes


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


def draw_metrics_panel(frame, summary, args, prediction):
    import cv2

    lines = [
        f"camera: {args.camera_id}",
        f"detector: {args.detector_mode}",
        f"frames: {summary['frames_processed']}",
        f"bbox: {summary['bbox_detections']}  keypoints: {summary['keypoints_extracted']}",
        f"seq: {summary['generated_sequences']}  pred: {summary['lstm_predictions']}  events: {summary['events_generated']}",
        f"threshold: {getattr(args, 'action_threshold', DEFAULT_FAINT_THRESHOLD):.2f}  consecutive_faint: {summary.get('latest_consecutive_faint', 0)}",
    ]
    if prediction:
        faint_prob = faint_probability(prediction)
        faint_text = "n/a" if faint_prob is None else f"{faint_prob:.2f}"
        lines.append(f"label: {prediction['label']}  confidence: {prediction['score']:.2f}  faint: {faint_text}")

    x, y = 12, 62
    width = 520
    height = 24 + 24 * len(lines)
    cv2.rectangle(frame, (x - 8, y - 22), (x + width, y - 22 + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 8, y - 22), (x + width, y - 22 + height), (80, 220, 120), 1)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)


def make_placeholder(message):
    import cv2
    import numpy as np

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (18, 22, 30)
    cv2.putText(image, "AI OVERLAY", (28, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, message, (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 185, 205), 2, cv2.LINE_AA)
    return image


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


class OverlayHandler(BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json({"status": "ok", **self.state.status()})
            return
        if path == "/cameras":
            camera_id = getattr(self.server, "camera_id", "camera-1")
            self.send_json({
                "cameras": [
                    {
                        "id": camera_id,
                        "connected": self.state.status()["connected"],
                    }
                ]
            })
            return
        if path == "/summary":
            self.send_json(self.state.status()["summary"])
            return
        if path == "/" or path.startswith("/stream"):
            self.stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt, *args):
        print(f"[ai-overlay-http] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        import cv2

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        delay = 1.0 / max(1.0, float(self.server.target_fps))
        while True:
            frame = self.state.snapshot()
            if frame is None:
                status = self.state.status()
                frame = make_placeholder(status["last_error"] or "waiting for RTSP frames")
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                time.sleep(delay)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(delay)


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
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--action-model", default=None)
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
    OverlayHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), OverlayHandler)
    server.target_fps = args.mjpeg_fps
    server.camera_id = args.camera_id

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
