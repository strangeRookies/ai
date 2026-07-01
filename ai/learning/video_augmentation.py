from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


SUPPORTED_AUGMENTATIONS: Final = {
    "brightness",
    "noise",
    "blur",
    "compression",
    "scale_down",
    "partial_occlusion",
    "horizontal_flip",
}


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    output_path: Path
    augmentation_config: str
    frames_written: int


def generate_augmented_video(input_path: Path, output_path: Path, augmentation_type: str, random_seed: int = 42) -> AugmentationResult:
    if augmentation_type not in SUPPORTED_AUGMENTATIONS:
        raise RuntimeError(f"Unsupported augmentation_type={augmentation_type}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Input video has invalid dimensions: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")
    rng = np.random.default_rng(random_seed)
    config = _config(augmentation_type, random_seed)
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_augment_frame(frame, augmentation_type, rng))
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    if frames_written == 0:
        raise RuntimeError(f"No frames written for input video: {input_path}")
    return AugmentationResult(output_path=output_path, augmentation_config=config, frames_written=frames_written)


def _augment_frame(frame: np.ndarray, augmentation_type: str, rng: np.random.Generator) -> np.ndarray:
    match augmentation_type:
        case "brightness":
            return cv2.convertScaleAbs(frame, alpha=0.75, beta=-12)
        case "noise":
            noise = rng.normal(0, 8, frame.shape).astype(np.int16)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        case "blur":
            return cv2.GaussianBlur(frame, (5, 5), 0)
        case "compression":
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
            if not ok:
                raise RuntimeError("JPEG compression augmentation failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("JPEG decompression augmentation failed")
            return decoded
        case "scale_down":
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (max(1, int(w * 0.6)), max(1, int(h * 0.6))), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        case "partial_occlusion":
            output = frame.copy()
            h, w = output.shape[:2]
            x0 = int(w * 0.55)
            y0 = int(h * 0.15)
            output[y0 : min(h, y0 + int(h * 0.35)), x0 : min(w, x0 + int(w * 0.3))] = 0
            return output
        case "horizontal_flip":
            return cv2.flip(frame, 1)
        case unreachable:
            raise RuntimeError(f"Unsupported augmentation_type={unreachable}")


def _config(augmentation_type: str, random_seed: int) -> str:
    params = {
        "brightness": {"alpha": 0.75, "beta": -12},
        "noise": {"sigma": 8},
        "blur": {"kernel": 5},
        "compression": {"jpeg_quality": 35},
        "scale_down": {"scale": 0.6},
        "partial_occlusion": {"occlusion_ratio": 0.35},
        "horizontal_flip": {"flip": True},
    }
    return json.dumps({"seed": random_seed, "type": augmentation_type, "params": params[augmentation_type]}, sort_keys=True)
