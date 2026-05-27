import csv
from queue import Queue
from pathlib import Path

from ai.action.classifier import MockActionClassifier
from ai.action.sequence_buffer import CropSequenceBuffer
from ai.config import parse_config
from ai.detection.yolo_person_detector import MockPersonDetector, YoloPersonDetector
from ai.events.clip_worker import ClipWriterWorker, enqueue_event_clip
from ai.events.event_clip import EventClipBuffer
from ai.evaluation.event_evaluator import evaluate_frame_level
from ai.labels.event_label_loader import load_dataset_rows, load_event_label
from ai.publishers.event_publisher import ConsoleEventPublisher, build_event_payload
from ai.streams.video_reader import VideoReader
from ai.visualization.draw import draw_overlay


def create_detector(config):
    if config.detector_mode == "mock":
        return MockPersonDetector(conf=config.yolo_conf)
    return YoloPersonDetector(config.yolo_model, conf=config.yolo_conf, iou=config.yolo_iou)


def run_one_video(config, video_path, label_path):
    label = load_event_label(label_path) if label_path else None
    detector = create_detector(config)
    classifier = MockActionClassifier()
    buffer = CropSequenceBuffer(config.sequence_length, config.sequence_stride, config.resize_size)
    publisher = ConsoleEventPublisher()
    records = []
    writer = None
    clip_queue = None
    clip_worker = None
    clip_buffer = None

    with VideoReader(str(video_path)) as reader:
        if config.event_clip_enabled:
            clip_queue = Queue(maxsize=max(1, config.event_clip_queue_size))
            clip_buffer = EventClipBuffer(
                pre_event_frame_count=config.event_clip_pre_frames,
                post_event_frame_count=config.event_clip_post_frames,
                cooldown_seconds=config.event_clip_cooldown_seconds,
                fps=reader.fps,
                output_dir=config.event_clip_output_dir,
            )
            clip_worker = ClipWriterWorker(clip_queue)
            clip_worker.start()

        while True:
            packet = reader.read()
            if packet is None:
                break
            if config.max_frames > 0 and packet.frame_idx >= config.max_frames:
                break

            clip_task = clip_buffer.add_frame(packet.frame) if clip_buffer else None
            if clip_task and clip_queue:
                enqueue_event_clip(clip_queue, clip_task)

            detection = detector.detect(packet.frame, packet.frame_idx)
            boxes = detection["boxes"]
            sequence = buffer.add(packet.frame_idx, packet.frame, boxes)
            prediction = classifier.predict(sequence) if sequence else None
            pred_active = bool(prediction and prediction["label"] in {"Fight", "Fall"})
            gt_active = label.is_active(packet.frame_idx) if label else False
            records.append({"frame_idx": packet.frame_idx, "pred_active": pred_active, "gt_active": gt_active})

            if prediction and pred_active:
                payload = build_event_payload(
                    camera_id=config.camera_id,
                    frame_idx=packet.frame_idx,
                    timestamp=packet.timestamp,
                    event_type=prediction["label"],
                    score=prediction["score"],
                    boxes=boxes,
                    snapshot_path=None,
                )
                publisher.publish(payload)
                if clip_buffer:
                    clip_buffer.trigger_event(
                        event_type=prediction["label"],
                        camera_id=config.camera_id,
                        metadata={
                            "frame_idx": packet.frame_idx,
                            "timestamp": packet.timestamp,
                            "score": prediction["score"],
                            "boxes": boxes,
                        },
                    )

            if config.output_video:
                import cv2

                overlay = draw_overlay(packet.frame, boxes, prediction if pred_active else None, packet.frame_idx)
                if writer is None:
                    h, w = overlay.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(config.output_video, fourcc, packet.fps, (w, h))
                writer.write(overlay)

    if writer is not None:
        writer.release()
    if clip_worker is not None:
        clip_worker.stop()
    metrics = evaluate_frame_level(records)
    print(f"[evaluation] {metrics}")
    return metrics


def main():
    config = parse_config()
    if config.dataset_csv:
        rows = load_dataset_rows(config.dataset_csv, split=config.split)
        all_metrics = []
        for row in rows:
            all_metrics.append(run_one_video(config, row["video_path"], row["label_path"]))
        return 0

    if not config.input_uri:
        raise SystemExit("--input or INPUT_URI is required when --dataset-csv is not provided")
    run_one_video(config, config.input_uri, config.label_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
