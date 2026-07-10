from __future__ import annotations

from collections.abc import Sequence
from typing import Final


CLASS_NAMES: Final[tuple[str, str]] = ("Normal", "Fight")


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, object]:
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth)][int(prediction)] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": {"labels": list(CLASS_NAMES), "matrix": matrix},
        "per_class_metrics": per_class_metrics(matrix),
    }


def per_class_metrics(matrix: list[list[int]]) -> dict[str, dict[str, float | int]]:
    total = sum(sum(row) for row in matrix)
    metrics: dict[str, dict[str, float | int]] = {}
    for class_id, label in enumerate(CLASS_NAMES):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in range(len(CLASS_NAMES)) if row != class_id)
        false_negative = sum(matrix[class_id][column] for column in range(len(CLASS_NAMES)) if column != class_id)
        true_negative = total - true_positive - false_positive - false_negative
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "support": int(sum(matrix[class_id])),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
        }
    return metrics
