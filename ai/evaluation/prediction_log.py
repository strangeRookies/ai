import json
from pathlib import Path


POSITIVE_LABEL = "Faint"
NEGATIVE_LABEL = "Normal"


def normalize_ground_truth(value):
    if value is None:
        return None
    label = str(value).strip()
    if not label:
        return None
    lowered = label.lower().replace("-", "_").replace(" ", "_")
    if lowered in {"faint", "fall", "fall_down", "1", "true", "positive"}:
        return POSITIVE_LABEL
    return NEGATIVE_LABEL


def prediction_label(prediction):
    label = str((prediction or {}).get("label") or NEGATIVE_LABEL)
    return POSITIVE_LABEL if label == POSITIVE_LABEL else NEGATIVE_LABEL


def prediction_confidence(prediction):
    if not prediction:
        return 0.0
    probabilities = prediction.get("probabilities") or {}
    if POSITIVE_LABEL in probabilities:
        return float(probabilities[POSITIVE_LABEL])
    return float(prediction.get("score", 0.0))


def result_type(prediction, ground_truth):
    truth = normalize_ground_truth(ground_truth)
    if truth is None:
        return None
    predicted = prediction_label(prediction)
    if predicted == POSITIVE_LABEL and truth == POSITIVE_LABEL:
        return "TP"
    if predicted == POSITIVE_LABEL and truth == NEGATIVE_LABEL:
        return "FP"
    if predicted == NEGATIVE_LABEL and truth == POSITIVE_LABEL:
        return "FN"
    return "TN"


def sequence_quality(sequence):
    detections = (sequence or {}).get("detections") or []
    keypoint_count = 0
    missing_count = 0
    confidence_total = 0.0
    confidence_count = 0
    for detection in detections:
        keypoints = detection.get("keypoints") or []
        if not keypoints:
            missing_count += 17
            keypoint_count += 17
            continue
        for keypoint in keypoints:
            keypoint_count += 1
            confidence = _keypoint_confidence(keypoint)
            if confidence is None or confidence <= 0.0:
                missing_count += 1
                continue
            confidence_total += confidence
            confidence_count += 1
    if keypoint_count == 0:
        return {"keypoint_missing_rate": 1.0, "avg_keypoint_conf": 0.0, "sequence_length": 0}
    return {
        "keypoint_missing_rate": round(missing_count / keypoint_count, 6),
        "avg_keypoint_conf": round(confidence_total / confidence_count, 6) if confidence_count else 0.0,
        "sequence_length": len(detections),
    }


def build_prediction_log_row(
    args,
    packet,
    sequence,
    prediction,
    consecutive_faint_count,
    cooldown_active,
    event_emitted,
    ground_truth=None,
):
    quality = sequence_quality(sequence)
    truth = normalize_ground_truth(ground_truth)
    row = {
        "source_id": getattr(args, "source_id", None) or getattr(args, "video_id", None) or getattr(args, "rtsp_url", None),
        "video_id": getattr(args, "video_id", None) or getattr(args, "source_id", None) or getattr(args, "rtsp_url", None),
        "camera_id": getattr(args, "camera_id", None),
        "frame_idx": int(packet.frame_idx),
        "timestamp": float(packet.timestamp),
        "track_id": (sequence or {}).get("track_id"),
        "prediction": prediction_label(prediction),
        "confidence": round(prediction_confidence(prediction), 6),
        "ground_truth": truth,
        "result_type": result_type(prediction, truth),
        "keypoint_missing_rate": quality["keypoint_missing_rate"],
        "avg_keypoint_conf": quality["avg_keypoint_conf"],
        "sequence_length": quality["sequence_length"],
        "consecutive_faint_count": int(consecutive_faint_count),
        "cooldown_active": bool(cooldown_active),
        "event_emitted": bool(event_emitted),
    }
    if row["track_id"] is not None:
        row["track_id"] = int(row["track_id"])
    return row


def append_prediction_jsonl(path, row):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _keypoint_confidence(keypoint):
    if isinstance(keypoint, dict):
        value = keypoint.get("confidence")
    elif isinstance(keypoint, (list, tuple)) and len(keypoint) >= 3:
        value = keypoint[2]
    else:
        value = None
    if value is None:
        return None
    return float(value)
