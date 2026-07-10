"""TensorRT engine validation and YOLO runtime selection with PyTorch fallback.

When a ``.engine`` path is requested we always force-validate the file and
confirm the engine is usable for the real inference stack (ultralytics YOLO).

Raw ``tensorrt.Runtime.deserialize_cuda_engine`` is still probed for diagnostics:
on some GPU stacks (e.g. ultralytics-built engines vs pip ``tensorrt``) raw
deserialize fails with magicTag mismatch while YOLO can load the engine. In that
case we accept the engine for inference and record both outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

RUNTIME_TENSORRT = "tensorrt"
RUNTIME_PYTORCH = "pytorch"
RUNTIME_PYTORCH_FALLBACK = "pytorch_fallback"
RUNTIME_MOCK = "mock"

EngineValidator = Callable[[str | Path], "EngineValidationResult"]


@dataclass(frozen=True)
class EngineValidationResult:
    ok: bool
    path: str
    reason: str | None = None
    raw_deserialize_ok: bool | None = None
    validation_method: str | None = None
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "path": self.path, "reason": self.reason}
        if self.raw_deserialize_ok is not None:
            payload["raw_deserialize_ok"] = self.raw_deserialize_ok
        if self.validation_method is not None:
            payload["validation_method"] = self.validation_method
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class RuntimeSelection:
    runtime: str
    model_path: str
    requested_model_path: str
    engine_validation: dict[str, Any] | None
    fallback_occurred: bool
    tensorrt_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "model_path": self.model_path,
            "requested_model_path": self.requested_model_path,
            "engine_validation": self.engine_validation,
            "fallback_occurred": self.fallback_occurred,
            "tensorrt_error": self.tensorrt_error,
        }


def is_tensorrt_engine_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".engine"


def default_pytorch_fallback_path(engine_path: str | Path) -> Path:
    return Path(engine_path).with_suffix(".pt")


def validate_engine_file(path: str | Path) -> EngineValidationResult:
    """Check engine path existence, extension, and non-zero size."""
    engine_path = Path(path)
    if engine_path.suffix.lower() != ".engine":
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=f"invalid extension for TensorRT engine: {engine_path.suffix!r} (expected .engine)",
            validation_method="file_checks",
        )
    if not engine_path.exists():
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=f"engine file does not exist: {engine_path}",
            validation_method="file_checks",
        )
    if not engine_path.is_file():
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=f"engine path is not a file: {engine_path}",
            validation_method="file_checks",
        )
    size = engine_path.stat().st_size
    if size <= 0:
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason="engine file is empty (0 bytes)",
            validation_method="file_checks",
        )
    return EngineValidationResult(
        ok=True,
        path=str(engine_path),
        reason=None,
        validation_method="file_checks",
    )


def _import_tensorrt() -> Any:
    import tensorrt  # type: ignore[import-not-found]

    return tensorrt


def deserialize_tensorrt_engine(
    path: str | Path,
    *,
    trt_module: Any | None = None,
) -> EngineValidationResult:
    """Validate the engine file and deserialize it via pip TensorRT runtime.

    This is a strict raw-TRT probe. Prefer :func:`validate_engine_for_inference`
    when deciding whether the engine may be used with ultralytics YOLO.
    ``trt_module`` may be injected for unit tests so real TensorRT/CUDA is not required.
    """
    file_check = validate_engine_file(path)
    if not file_check.ok:
        return file_check

    engine_path = Path(path)
    try:
        trt = trt_module if trt_module is not None else _import_tensorrt()
    except ImportError as exc:
        reason = f"tensorrt import failed: {exc}"
        logger.warning("[tensorrt] raw deserialize probe failed path=%s reason=%s", engine_path, reason)
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=reason,
            raw_deserialize_ok=False,
            validation_method="raw_tensorrt_deserialize",
        )

    try:
        logger_obj = None
        if hasattr(trt, "Logger"):
            severity = getattr(trt.Logger, "WARNING", 1)
            logger_obj = trt.Logger(severity)
        runtime = trt.Runtime(logger_obj) if logger_obj is not None else trt.Runtime()
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            reason = (
                "TensorRT deserialize_cuda_engine returned None "
                "(possible CUDA/TensorRT version mismatch or incompatible engine)"
            )
            logger.warning("[tensorrt] raw deserialize probe failed path=%s reason=%s", engine_path, reason)
            return EngineValidationResult(
                ok=False,
                path=str(engine_path),
                reason=reason,
                raw_deserialize_ok=False,
                validation_method="raw_tensorrt_deserialize",
            )
        logger.info(
            "[tensorrt] raw deserialize OK path=%s size_bytes=%s",
            engine_path,
            len(engine_bytes),
        )
        return EngineValidationResult(
            ok=True,
            path=str(engine_path),
            reason=None,
            raw_deserialize_ok=True,
            validation_method="raw_tensorrt_deserialize",
        )
    except Exception as exc:  # noqa: BLE001 - surface any TensorRT/CUDA failure as validation error
        reason = (
            f"TensorRT engine deserialize failed: {exc} "
            "(check CUDA/TensorRT version compatibility)"
        )
        logger.warning("[tensorrt] raw deserialize probe failed path=%s reason=%s", engine_path, reason)
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=reason,
            raw_deserialize_ok=False,
            validation_method="raw_tensorrt_deserialize",
        )


def validate_engine_for_inference(
    path: str | Path,
    *,
    yolo_loader: Callable[[str], Any] | None = None,
    trt_module: Any | None = None,
    probe_raw_tensorrt: bool = True,
) -> EngineValidationResult:
    """Force-validate an engine for actual YOLO inference.

    Order:
    1. file existence / extension / non-zero size (hard fail)
    2. optional raw pip-TensorRT deserialize probe (diagnostic)
    3. if ``yolo_loader`` is provided, ultralytics-style load is authoritative
    4. if no ``yolo_loader``, raw deserialize result is authoritative
    """
    file_check = validate_engine_file(path)
    if not file_check.ok:
        return file_check

    engine_path = Path(path)
    raw: EngineValidationResult | None = None
    if probe_raw_tensorrt:
        raw = deserialize_tensorrt_engine(engine_path, trt_module=trt_module)

    if yolo_loader is None:
        if raw is None:
            return EngineValidationResult(
                ok=True,
                path=str(engine_path),
                reason=None,
                validation_method="file_checks",
            )
        return raw

    try:
        yolo_loader(str(engine_path))
    except Exception as exc:  # noqa: BLE001
        raw_reason = raw.reason if raw is not None and not raw.ok else None
        reason = f"ultralytics/YOLO engine load failed: {exc}"
        if raw_reason:
            reason = f"{reason}; raw_tensorrt={raw_reason}"
        logger.warning("[tensorrt] engine rejected for inference path=%s reason=%s", engine_path, reason)
        return EngineValidationResult(
            ok=False,
            path=str(engine_path),
            reason=reason,
            raw_deserialize_ok=None if raw is None else raw.ok,
            validation_method="ultralytics_yolo_load",
        )

    details = None
    if raw is not None and not raw.ok:
        details = (
            f"raw pip-tensorrt deserialize failed ({raw.reason}); "
            "accepted because ultralytics YOLO load succeeded"
        )
        logger.warning(
            "[tensorrt] raw deserialize failed but ultralytics load OK path=%s details=%s",
            engine_path,
            details,
        )
    else:
        logger.info("[tensorrt] engine validated for inference via ultralytics path=%s", engine_path)

    return EngineValidationResult(
        ok=True,
        path=str(engine_path),
        reason=None,
        raw_deserialize_ok=None if raw is None else raw.ok,
        validation_method="ultralytics_yolo_load",
        details=details,
    )


def resolve_yolo_model_runtime(
    model_path: str | Path,
    *,
    fallback_model_path: str | Path | None = None,
    engine_validator: EngineValidator | None = None,
) -> RuntimeSelection:
    """Select the inference model path and runtime label without loading YOLO weights.

    For ``.engine`` requests:
    - always run forced validation
    - on failure, switch to sibling ``.pt`` (or explicit fallback) when present
    - never return a failed engine path for inference

    Default validator is file checks only. Callers that own a YOLO loader should
    pass :func:`validate_engine_for_inference` (with that loader) or load via
    :class:`detector.yolo_pose_detector.YoloPoseDetector`.
    """
    requested = Path(model_path)
    requested_str = str(requested)

    if not is_tensorrt_engine_path(requested):
        selection = RuntimeSelection(
            runtime=RUNTIME_PYTORCH,
            model_path=requested_str,
            requested_model_path=requested_str,
            engine_validation=None,
            fallback_occurred=False,
            tensorrt_error=None,
        )
        logger.info(
            "[yolo-runtime] selected runtime=%s model_path=%s fallback=%s",
            selection.runtime,
            selection.model_path,
            selection.fallback_occurred,
        )
        return selection

    validator = engine_validator or validate_engine_file
    validation = validator(requested)
    validation_dict = validation.to_dict() if isinstance(validation, EngineValidationResult) else dict(validation)
    is_ok = validation.ok if isinstance(validation, EngineValidationResult) else bool(validation_dict.get("ok"))

    if is_ok:
        selection = RuntimeSelection(
            runtime=RUNTIME_TENSORRT,
            model_path=requested_str,
            requested_model_path=requested_str,
            engine_validation=validation_dict,
            fallback_occurred=False,
            tensorrt_error=None,
        )
        logger.info(
            "[yolo-runtime] selected runtime=%s model_path=%s fallback=%s",
            selection.runtime,
            selection.model_path,
            selection.fallback_occurred,
        )
        return selection

    tensorrt_error = (
        validation.reason
        if isinstance(validation, EngineValidationResult)
        else validation_dict.get("reason")
    ) or "TensorRT engine validation failed"
    fallback = Path(fallback_model_path) if fallback_model_path is not None else default_pytorch_fallback_path(requested)

    if fallback.exists() and fallback.is_file() and fallback.stat().st_size > 0:
        selection = RuntimeSelection(
            runtime=RUNTIME_PYTORCH_FALLBACK,
            model_path=str(fallback),
            requested_model_path=requested_str,
            engine_validation=validation_dict,
            fallback_occurred=True,
            tensorrt_error=str(tensorrt_error),
        )
        logger.warning(
            "[yolo-runtime] TensorRT engine rejected; falling back to PyTorch. "
            "runtime=%s model_path=%s tensorrt_error=%s",
            selection.runtime,
            selection.model_path,
            selection.tensorrt_error,
        )
        return selection

    message = (
        f"TensorRT engine validation failed and PyTorch fallback is unavailable. "
        f"tensorrt_error={tensorrt_error}; fallback_path={fallback}"
    )
    logger.error("[yolo-runtime] %s", message)
    raise RuntimeError(message)


def detector_runtime_summary(
    detector: Any,
    *,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Build the runtime fields that belong in a run summary JSON."""
    engine_validation = getattr(detector, "engine_validation", None)
    if engine_validation is not None and not isinstance(engine_validation, Mapping):
        engine_validation = dict(engine_validation) if hasattr(engine_validation, "keys") else engine_validation
    model_path = getattr(detector, "model_path", None) or getattr(detector, "model_name", requested_model)
    return {
        "runtime": getattr(detector, "runtime", None),
        "model_path": model_path,
        "engine_validation": engine_validation,
    }


def attach_runtime_summary_fields(
    summary: dict[str, Any],
    detector: Any,
    *,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Add runtime fields to an existing summary without removing prior keys."""
    summary.update(detector_runtime_summary(detector, requested_model=requested_model))
    return summary


def log_selected_runtime(selection: RuntimeSelection | Mapping[str, Any], *, prefix: str = "[yolo-runtime]") -> None:
    data = selection.to_dict() if isinstance(selection, RuntimeSelection) else dict(selection)
    logger.info(
        "%s final runtime=%s model_path=%s fallback_occurred=%s tensorrt_error=%s",
        prefix,
        data.get("runtime"),
        data.get("model_path"),
        data.get("fallback_occurred"),
        data.get("tensorrt_error"),
    )
    # Match existing CLI-style logging used across the RTSP pipeline.
    print(
        f"{prefix} final runtime={data.get('runtime')} "
        f"model_path={data.get('model_path')} "
        f"fallback_occurred={data.get('fallback_occurred')} "
        f"tensorrt_error={data.get('tensorrt_error')}",
        flush=True,
    )
