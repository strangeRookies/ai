from __future__ import annotations

from ai.evaluation.prediction_log import NEGATIVE_LABEL, POSITIVE_LABEL
from ai.learning.self_improving_error_mining import FEATURE_SCHEMA, PredictionRow


def sample_prediction_rows(include_invalid: bool = False) -> list[PredictionRow]:
    rows = [
        sample_prediction_row("clip-fp", POSITIVE_LABEL, NEGATIVE_LABEL, 10),
        sample_prediction_row("clip-fn", NEGATIVE_LABEL, POSITIVE_LABEL, 40),
    ]
    if include_invalid:
        rows.append(sample_prediction_row("clip-missing-bbox", POSITIVE_LABEL, NEGATIVE_LABEL, 70, with_bbox=False))
        invalid = sample_prediction_row("clip-unverified", NEGATIVE_LABEL, POSITIVE_LABEL, 100)
        invalid["label_interval_verified"] = False
        rows.append(invalid)
    return rows


def sample_prediction_row(
    clip_id: str,
    prediction: str,
    truth: str,
    start_frame: int,
    with_bbox: bool = True,
) -> PredictionRow:
    return {
        "source_video": "sample_source.mp4",
        "clip_id": clip_id,
        "start_frame": start_frame,
        "end_frame": start_frame + 2,
        "frameId": start_frame + 2,
        "prediction_label": prediction,
        "prediction_score": 0.82 if prediction == POSITIVE_LABEL else 0.18,
        "ground_truth_label": truth,
        "threshold": 0.3,
        "model_name": "sample-lstm",
        "checkpoint_path": "sample_checkpoint.pt",
        "sequence_length": 3,
        "sequence_stride": 1,
        "feature_schema": FEATURE_SCHEMA,
        "sequence": sample_sequence(with_bbox),
        "label_interval_verified": True,
    }


def sample_sequence(with_bbox: bool = True) -> dict[str, list[dict[str, object]] | list[tuple[int, int, int]]]:
    detections: list[dict[str, object]] = []
    for frame in range(3):
        detection: dict[str, object] = {
            "keypoints": [
                {"x": 10.0 + index, "y": 20.0 + index + frame, "confidence": 0.9}
                for index in range(17)
            ],
        }
        if with_bbox:
            detection["bbox"] = [10.0, 20.0, 110.0, 220.0]
        detections.append(detection)
    return {"detections": detections, "frame_shapes": [(400, 400, 3), (400, 400, 3), (400, 400, 3)]}
