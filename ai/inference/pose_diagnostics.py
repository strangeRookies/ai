from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ai.inference.pose_diagnostic_records import (
    DEFAULT_MIN_KEYPOINT_CONFIDENCE,
    bbox_for_diagnostic_image,
    build_pose_diagnostic_record,
    minimal_jsonl_record,
)
from ai.inference.pose_diagnostic_summary import PoseStats


@dataclass(frozen=True, slots=True)
class PoseDiagnosticsConfig:
    enabled: bool = False
    summary_every_n: int = 60
    min_keypoint_confidence: float = DEFAULT_MIN_KEYPOINT_CONFIDENCE
    image_output_enabled: bool = False
    image_output_dir: Path = Path("runs/pose_debug")
    image_every_n: int = 300
    jsonl_enabled: bool = False
    jsonl_path: Path = Path("runs/diagnostics/pose_tracking_diag.jsonl")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pose_debug_enabled() -> bool:
    return env_bool("POSE_DEBUG") or env_bool("TRACKING_DEBUG")


def config_from_args(args) -> PoseDiagnosticsConfig:
    enabled = bool(getattr(args, "pose_debug", False)) or pose_debug_enabled()
    image_output_dir = Path(str(getattr(args, "pose_debug_image_dir", "runs/pose_debug")))
    jsonl_path = Path(str(getattr(args, "pose_tracking_diag_jsonl_path", "runs/diagnostics/pose_tracking_diag.jsonl")))
    return PoseDiagnosticsConfig(
        enabled=enabled,
        summary_every_n=max(1, int(getattr(args, "pose_debug_summary_every_n", 60))),
        min_keypoint_confidence=float(getattr(args, "pose_min_keypoint_confidence", DEFAULT_MIN_KEYPOINT_CONFIDENCE)),
        image_output_enabled=bool(getattr(args, "pose_debug_save_images", False)),
        image_output_dir=image_output_dir,
        image_every_n=max(1, int(getattr(args, "pose_debug_image_every_n", 300))),
        jsonl_enabled=bool(getattr(args, "pose_tracking_diag_jsonl", env_bool("POSE_TRACKING_DIAG_JSONL", False))),
        jsonl_path=jsonl_path,
    )


class PoseDiagnosticsReporter:
    def __init__(self, config: PoseDiagnosticsConfig | None = None) -> None:
        self.config = config or PoseDiagnosticsConfig()
        self._stats_by_camera: dict[str, PoseStats] = {}
        self._observed_frames = 0

    def observe(
        self,
        camera_login_id: str,
        source_url: str,
        assigned_video_path: str | None,
        frame_id: int | None,
        timestamp_ms: int | None,
        raw_detections: list[dict],
        tracker_diagnostics: dict,
        sequence_ready_count: int,
        sequence_diagnostics: dict | None = None,
        frame=None,
    ) -> dict:
        record = build_pose_diagnostic_record(
            camera_login_id=camera_login_id,
            source_url=source_url,
            assigned_video_path=assigned_video_path,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            raw_detections=raw_detections,
            tracker_diagnostics=tracker_diagnostics,
            sequence_ready_count=sequence_ready_count,
            min_keypoint_confidence=self.config.min_keypoint_confidence,
            sequence_diagnostics=sequence_diagnostics,
        )
        stats = self._stats_by_camera.setdefault(camera_login_id, PoseStats())
        stats.observe(record)
        self._observed_frames += 1
        if self.config.jsonl_enabled:
            self._append_jsonl(record)
        if self.config.enabled:
            print(f"[pose-diagnostics] {json.dumps(record, ensure_ascii=False)}", flush=True)
            if self._observed_frames % self.config.summary_every_n == 0:
                self.log_summary()
            if frame is not None and self.config.image_output_enabled:
                frame_id_value = int(frame_id or self._observed_frames)
                if frame_id_value % self.config.image_every_n == 0:
                    self.save_sample_image(camera_login_id, frame_id_value, timestamp_ms, frame, raw_detections)
        return record

    def _append_jsonl(self, record: dict) -> None:
        self.config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        payload = minimal_jsonl_record(record)
        with self.config.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_summary(self) -> dict:
        record = {
            "stage": "pose_summary",
            "cameras": {
                camera_login_id: stats.summary()
                for camera_login_id, stats in sorted(self._stats_by_camera.items())
            },
        }
        if self.config.enabled:
            print(f"[pose-diagnostics] {json.dumps(record, ensure_ascii=False)}", flush=True)
        return record

    def log_final_summary(self) -> dict:
        record = {
            "stage": "pose_final_summary",
            "cameras": {
                camera_login_id: stats.final_summary()
                for camera_login_id, stats in sorted(self._stats_by_camera.items())
            },
        }
        if self.config.enabled:
            print(f"[pose-diagnostics] {json.dumps(record, ensure_ascii=False)}", flush=True)
        return record

    def save_sample_image(
        self,
        camera_login_id: str,
        frame_id: int | None,
        timestamp_ms: int | None,
        frame,
        detections: list[dict],
    ) -> Path | None:
        if not self.config.enabled or not self.config.image_output_enabled:
            return None
        self.config.image_output_dir.mkdir(parents=True, exist_ok=True)
        safe_camera = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in camera_login_id)
        output = self.config.image_output_dir / f"{safe_camera}_frame-{frame_id or 0}_ts-{timestamp_ms or 0}.jpg"
        try:
            import cv2

            image = frame.copy()
            for detection in detections:
                bbox = bbox_for_diagnostic_image(detection)
                if bbox:
                    x1, y1, x2, y2 = [int(value) for value in bbox]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for point in detection.get("keypoints") or []:
                    if float(point.get("confidence", 0.0)) <= 0:
                        continue
                    cv2.circle(image, (int(point.get("x", 0)), int(point.get("y", 0))), 2, (0, 0, 255), -1)
            cv2.imwrite(str(output), image)
        except (ImportError, AttributeError, OSError):
            output.write_bytes(b"pose debug image unavailable")
        return output
