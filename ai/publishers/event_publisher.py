import json


class EventPublisher:
    def publish(self, payload):
        raise NotImplementedError


class ConsoleEventPublisher(EventPublisher):
    def publish(self, payload):
        print(f"[event] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def build_event_payload(camera_id, frame_idx, timestamp, event_type, score, boxes, snapshot_path=None):
    return {
        "camera_id": camera_id,
        "frame_idx": int(frame_idx),
        "timestamp": float(timestamp),
        "event_type": event_type,
        "score": float(score),
        "confidence": float(score),
        "boxes": boxes,
        "bbox": boxes[0] if boxes else None,
        "snapshot_path": snapshot_path,
    }
