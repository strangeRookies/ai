from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ai.inference.tensorrt_runtime import (
    RUNTIME_PYTORCH_FALLBACK,
    RUNTIME_TENSORRT,
    EngineValidationResult,
    RuntimeSelection,
    default_pytorch_fallback_path,
    is_tensorrt_engine_path,
    log_selected_runtime,
    resolve_yolo_model_runtime,
    validate_engine_for_inference,
)


class YoloPoseDetector:
    def __init__(
        self,
        model_name,
        device="auto",
        imgsz=640,
        conf=0.25,
        *,
        yolo_cls: Callable[[str], Any] | None = None,
        engine_validator: Callable[[str | Path], EngineValidationResult] | None = None,
        fallback_model_path: str | Path | None = None,
    ):
        self.device = None if device == "auto" else device
        self.imgsz = imgsz
        self.conf = conf
        self.fallback_occurred = False
        self.engine_validation = None
        self.tensorrt_error = None
        self.requested_model_path = str(model_name)

        yolo_loader = yolo_cls
        if yolo_loader is None:
            try:
                from ultralytics import YOLO as ultralytics_yolo
            except ImportError as exc:
                raise RuntimeError(f"ultralytics is not installed: {exc}") from exc
            yolo_loader = ultralytics_yolo

        selection, model = self._load_with_runtime_selection(
            model_name=str(model_name),
            yolo_loader=yolo_loader,
            engine_validator=engine_validator,
            fallback_model_path=fallback_model_path,
        )
        self.model = model
        self.runtime = selection.runtime
        self.model_path = selection.model_path
        self.model_name = selection.model_path
        self.engine_validation = selection.engine_validation
        self.fallback_occurred = selection.fallback_occurred
        self.tensorrt_error = selection.tensorrt_error
        log_selected_runtime(selection)

    def _load_with_runtime_selection(
        self,
        *,
        model_name: str,
        yolo_loader: Callable[[str], Any],
        engine_validator: Callable[[str | Path], EngineValidationResult] | None,
        fallback_model_path: str | Path | None,
    ) -> tuple[RuntimeSelection, Any]:
        if not is_tensorrt_engine_path(model_name):
            selection = resolve_yolo_model_runtime(
                model_name,
                fallback_model_path=fallback_model_path,
                engine_validator=engine_validator,
            )
            try:
                model = yolo_loader(selection.model_path)
                return selection, model
            except Exception as exc:
                raise RuntimeError(f"Failed to load YOLO model '{selection.model_path}': {exc}") from exc

        # --- TensorRT .engine path ---
        # Custom validator (tests / advanced callers): resolve first, then load.
        if engine_validator is not None:
            selection = resolve_yolo_model_runtime(
                model_name,
                fallback_model_path=fallback_model_path,
                engine_validator=engine_validator,
            )
            if selection.runtime == RUNTIME_TENSORRT:
                try:
                    model = yolo_loader(selection.model_path)
                    return selection, model
                except Exception as load_exc:
                    return self._fallback_after_engine_failure(
                        model_name=model_name,
                        yolo_loader=yolo_loader,
                        fallback_model_path=fallback_model_path,
                        tensorrt_error=f"TensorRT engine load/init failed: {load_exc}",
                        prior_validation=selection.engine_validation,
                        load_exc=load_exc,
                    )
            try:
                model = yolo_loader(selection.model_path)
                return selection, model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLO model '{selection.model_path}' after TensorRT failure. "
                    f"tensorrt_error={selection.tensorrt_error}; pytorch_error={exc}"
                ) from exc

        # Production path: file checks + raw TRT probe + ultralytics authoritative load.
        # Capture the YOLO(engine) instance so we do not load twice.
        loaded_model: dict[str, Any] = {}

        def _capturing_loader(path: str) -> Any:
            model = yolo_loader(path)
            loaded_model["model"] = model
            loaded_model["path"] = path
            return model

        validation = validate_engine_for_inference(
            model_name,
            yolo_loader=_capturing_loader,
            probe_raw_tensorrt=True,
        )
        validation_dict = validation.to_dict()

        if validation.ok and "model" in loaded_model:
            selection = RuntimeSelection(
                runtime=RUNTIME_TENSORRT,
                model_path=str(model_name),
                requested_model_path=str(model_name),
                engine_validation=validation_dict,
                fallback_occurred=False,
                tensorrt_error=None,
            )
            return selection, loaded_model["model"]

        return self._fallback_after_engine_failure(
            model_name=model_name,
            yolo_loader=yolo_loader,
            fallback_model_path=fallback_model_path,
            tensorrt_error=validation.reason or "TensorRT engine validation failed",
            prior_validation=validation_dict,
            load_exc=None,
        )

    def _fallback_after_engine_failure(
        self,
        *,
        model_name: str,
        yolo_loader: Callable[[str], Any],
        fallback_model_path: str | Path | None,
        tensorrt_error: str,
        prior_validation: dict[str, Any] | None,
        load_exc: Exception | None,
    ) -> tuple[RuntimeSelection, Any]:
        fallback = (
            Path(fallback_model_path)
            if fallback_model_path is not None
            else default_pytorch_fallback_path(model_name)
        )
        if not (fallback.exists() and fallback.is_file() and fallback.stat().st_size > 0):
            raise RuntimeError(
                f"{tensorrt_error}; PyTorch fallback unavailable at {fallback}"
            ) from load_exc

        try:
            model = yolo_loader(str(fallback))
        except Exception as fallback_exc:
            raise RuntimeError(
                f"TensorRT load failed and PyTorch fallback also failed. "
                f"tensorrt_error={tensorrt_error}; pytorch_error={fallback_exc}"
            ) from fallback_exc

        engine_validation = dict(prior_validation or {})
        engine_validation.setdefault("path", str(model_name))
        engine_validation["ok"] = False
        engine_validation["reason"] = tensorrt_error
        engine_validation["load_error"] = tensorrt_error

        selection = RuntimeSelection(
            runtime=RUNTIME_PYTORCH_FALLBACK,
            model_path=str(fallback),
            requested_model_path=str(model_name),
            engine_validation=engine_validation,
            fallback_occurred=True,
            tensorrt_error=tensorrt_error,
        )
        return selection, model

    def detect(self, frame):
        results = self.model.predict(
            frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            xyxy = boxes.xyxy.detach().float().cpu().tolist()
            confs = boxes.conf.detach().float().cpu().tolist() if boxes.conf is not None else []
            track_ids = (
                boxes.id.detach().int().cpu().tolist()
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )

            keypoint_conf = None
            keypoint_xy = None
            keypoints = getattr(result, "keypoints", None)
            if keypoints is not None and keypoints.conf is not None:
                keypoint_conf = keypoints.conf.detach().float().cpu()
                keypoint_xy = keypoints.xy.detach().float().cpu() if keypoints.xy is not None else None

            for idx, bbox in enumerate(xyxy):
                pose_horizontal = _is_pose_horizontal(keypoint_xy, keypoint_conf, idx)
                keypoints = _extract_keypoints(keypoint_xy, keypoint_conf, idx)
                detections.append(
                    {
                        "track_id": track_ids[idx] if idx < len(track_ids) else None,
                        "bbox": [round(float(v), 2) for v in bbox],
                        "confidence": float(confs[idx]) if idx < len(confs) else 0.0,
                        "pose_state": "LYING" if pose_horizontal else "UNKNOWN",
                        "pose_horizontal": pose_horizontal,
                        "keypoints": keypoints,
                        "keypoint_confidence": _average_keypoint_confidence(keypoints),
                        "model_name": self.model_name,
                    }
                )
        return detections


def _is_pose_horizontal(keypoint_xy, keypoint_conf, person_idx, threshold=0.3):
    if keypoint_xy is None or keypoint_conf is None or person_idx >= keypoint_conf.shape[0]:
        return False

    left_shoulder, right_shoulder, left_hip, right_hip = 5, 6, 11, 12
    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    conf = keypoint_conf[person_idx]
    if any(idx >= conf.numel() or float(conf[idx]) < threshold for idx in required):
        return False

    xy = keypoint_xy[person_idx]
    shoulder_center = (xy[left_shoulder] + xy[right_shoulder]) / 2
    hip_center = (xy[left_hip] + xy[right_hip]) / 2
    torso_dx = abs(float(shoulder_center[0] - hip_center[0]))
    torso_dy = abs(float(shoulder_center[1] - hip_center[1]))
    return torso_dx > 0 and torso_dx / max(torso_dy, 1.0) >= 1.3


def _extract_keypoints(keypoint_xy, keypoint_conf, person_idx):
    if keypoint_xy is None or keypoint_conf is None or person_idx >= keypoint_conf.shape[0]:
        return None
    xy = keypoint_xy[person_idx]
    conf = keypoint_conf[person_idx]
    keypoints = []
    for idx in range(min(xy.shape[0], conf.shape[0])):
        keypoints.append(
            {
                "x": round(float(xy[idx][0]), 2),
                "y": round(float(xy[idx][1]), 2),
                "confidence": round(float(conf[idx]), 4),
            }
        )
    return keypoints


def _average_keypoint_confidence(keypoints):
    if not keypoints:
        return None
    return round(sum(item["confidence"] for item in keypoints) / len(keypoints), 4)
