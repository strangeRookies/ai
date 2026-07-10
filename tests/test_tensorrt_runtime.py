"""Unit tests for TensorRT engine validation and PyTorch fallback (no real TensorRT/CUDA)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai.inference.tensorrt_runtime import (
    RUNTIME_PYTORCH,
    RUNTIME_PYTORCH_FALLBACK,
    RUNTIME_TENSORRT,
    EngineValidationResult,
    default_pytorch_fallback_path,
    deserialize_tensorrt_engine,
    resolve_yolo_model_runtime,
    validate_engine_file,
    validate_engine_for_inference,
)
from detector.yolo_pose_detector import YoloPoseDetector


class FakeTrtLogger:
    WARNING = 1

    def __init__(self, *args, **kwargs):
        pass


class FakeTrtRuntime:
    def __init__(self, logger=None, *, engine=None, raise_on_deserialize=None):
        self.logger = logger
        self._engine = engine
        self._raise_on_deserialize = raise_on_deserialize

    def deserialize_cuda_engine(self, data):
        if self._raise_on_deserialize is not None:
            raise self._raise_on_deserialize
        return self._engine


class FakeTensorRtModule:
    Logger = FakeTrtLogger

    def __init__(self, *, engine=object(), raise_on_deserialize=None):
        self._engine = engine
        self._raise_on_deserialize = raise_on_deserialize

    def Runtime(self, logger=None):
        return FakeTrtRuntime(
            logger,
            engine=self._engine,
            raise_on_deserialize=self._raise_on_deserialize,
        )


class EngineFileValidationTest(unittest.TestCase):
    def test_rejects_missing_engine_file(self):
        result = validate_engine_file("does_not_exist.engine")
        self.assertFalse(result.ok)
        self.assertIn("does not exist", result.reason or "")

    def test_rejects_wrong_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"not-an-engine")
            result = validate_engine_file(path)
            self.assertFalse(result.ok)
            self.assertIn("extension", (result.reason or "").lower())

    def test_rejects_empty_engine_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.engine"
            path.write_bytes(b"")
            result = validate_engine_file(path)
            self.assertFalse(result.ok)
            self.assertIn("0", result.reason or "")

    def test_accepts_non_empty_engine_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.engine"
            path.write_bytes(b"fake-engine-bytes")
            result = validate_engine_file(path)
            self.assertTrue(result.ok)
            self.assertIsNone(result.reason)


class EngineDeserializeValidationTest(unittest.TestCase):
    def test_rejects_when_deserialize_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.engine"
            path.write_bytes(b"incompatible")
            trt = FakeTensorRtModule(engine=None)
            result = deserialize_tensorrt_engine(path, trt_module=trt)
            self.assertFalse(result.ok)
            self.assertIn("None", result.reason or "")

    def test_rejects_when_deserialize_raises_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.engine"
            path.write_bytes(b"old-engine")
            trt = FakeTensorRtModule(
                raise_on_deserialize=RuntimeError("CUDA version mismatch / TensorRT incompatible")
            )
            result = deserialize_tensorrt_engine(path, trt_module=trt)
            self.assertFalse(result.ok)
            self.assertIn("deserialize", (result.reason or "").lower())
            self.assertIn("CUDA", result.reason or "")

    def test_rejects_when_tensorrt_import_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.engine"
            path.write_bytes(b"engine-bytes")
            with patch("ai.inference.tensorrt_runtime._import_tensorrt", side_effect=ImportError("no tensorrt")):
                result = deserialize_tensorrt_engine(path, trt_module=None)
            self.assertFalse(result.ok)
            self.assertIn("import", (result.reason or "").lower())

    def test_accepts_when_deserialize_returns_engine_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "good.engine"
            path.write_bytes(b"valid-engine")
            trt = FakeTensorRtModule(engine=object())
            result = deserialize_tensorrt_engine(path, trt_module=trt)
            self.assertTrue(result.ok)
            self.assertIsNone(result.reason)


class InferenceValidationTest(unittest.TestCase):
    def test_ultralytics_load_accepts_engine_when_raw_deserialize_fails(self):
        """Matches GPU PC: pip tensorrt magicTag mismatch, ultralytics YOLO OK."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yolo.engine"
            path.write_bytes(b"ultralytics-engine-bytes")
            trt = FakeTensorRtModule(engine=None)
            loaded = []

            def yolo_loader(model_path):
                loaded.append(model_path)
                return object()

            result = validate_engine_for_inference(
                path,
                yolo_loader=yolo_loader,
                trt_module=trt,
            )
            self.assertTrue(result.ok)
            self.assertFalse(result.raw_deserialize_ok)
            self.assertEqual(result.validation_method, "ultralytics_yolo_load")
            self.assertIn("ultralytics", (result.details or "").lower())
            self.assertEqual(loaded, [str(path)])

    def test_inference_validation_fails_when_yolo_load_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yolo.engine"
            path.write_bytes(b"engine")
            trt = FakeTensorRtModule(engine=None)

            def yolo_loader(model_path):
                raise RuntimeError("cannot open engine")

            result = validate_engine_for_inference(
                path,
                yolo_loader=yolo_loader,
                trt_module=trt,
            )
            self.assertFalse(result.ok)
            self.assertIn("cannot open engine", result.reason or "")


class RuntimeResolveAndFallbackTest(unittest.TestCase):
    def test_pytorch_path_selects_pytorch_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "yolo26n-pose.pt"
            pt.write_bytes(b"pt-weights")
            selection = resolve_yolo_model_runtime(pt)
            self.assertEqual(selection.runtime, RUNTIME_PYTORCH)
            self.assertEqual(Path(selection.model_path), pt)
            self.assertFalse(selection.fallback_occurred)
            self.assertIsNone(selection.engine_validation)

    def test_invalid_engine_is_rejected_and_falls_back_to_pytorch(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo26n-pose.engine"
            pt = Path(tmp) / "yolo26n-pose.pt"
            engine.write_bytes(b"")  # 0-byte engine
            pt.write_bytes(b"pt-weights")

            selection = resolve_yolo_model_runtime(
                engine,
                engine_validator=lambda path: EngineValidationResult(False, str(path), "engine file is empty (0 bytes)"),
            )
            self.assertEqual(selection.runtime, RUNTIME_PYTORCH_FALLBACK)
            self.assertEqual(Path(selection.model_path), pt)
            self.assertTrue(selection.fallback_occurred)
            self.assertIsNotNone(selection.engine_validation)
            self.assertFalse(selection.engine_validation["ok"])
            self.assertIn("empty", selection.engine_validation["reason"])

    def test_valid_engine_selects_tensorrt_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo26n-pose.engine"
            engine.write_bytes(b"engine-bytes")
            selection = resolve_yolo_model_runtime(
                engine,
                engine_validator=lambda path: EngineValidationResult(True, str(path), None),
            )
            self.assertEqual(selection.runtime, RUNTIME_TENSORRT)
            self.assertEqual(Path(selection.model_path), engine)
            self.assertFalse(selection.fallback_occurred)
            self.assertTrue(selection.engine_validation["ok"])

    def test_both_fail_raises_with_tensorrt_cause_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "missing.engine"
            # no .pt fallback either
            with self.assertRaises(RuntimeError) as ctx:
                resolve_yolo_model_runtime(
                    engine,
                    engine_validator=lambda path: EngineValidationResult(
                        False, str(path), "engine file does not exist"
                    ),
                )
            message = str(ctx.exception)
            self.assertIn("engine file does not exist", message)
            self.assertIn("fallback", message.lower())

    def test_default_pytorch_fallback_path_uses_pt_suffix(self):
        self.assertEqual(
            default_pytorch_fallback_path("models/yolo26n-pose.engine"),
            Path("models/yolo26n-pose.pt"),
        )


class YoloPoseDetectorRuntimeTest(unittest.TestCase):
    def test_invalid_engine_never_loaded_for_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "broken.engine"
            pt = Path(tmp) / "broken.pt"
            engine.write_bytes(b"")
            pt.write_bytes(b"pt")

            loaded_paths: list[str] = []

            class FakeYOLO:
                def __init__(self, model_path):
                    loaded_paths.append(str(model_path))
                    self.model_path = model_path

            detector = YoloPoseDetector(
                str(engine),
                yolo_cls=FakeYOLO,
                engine_validator=lambda path: EngineValidationResult(
                    False, str(path), "engine file is empty (0 bytes)"
                ),
            )

            self.assertEqual(detector.runtime, RUNTIME_PYTORCH_FALLBACK)
            self.assertEqual(Path(detector.model_path), pt)
            self.assertEqual(loaded_paths, [str(pt)])
            self.assertNotIn(str(engine), loaded_paths)
            self.assertFalse(detector.engine_validation["ok"])

    def test_tensorrt_load_failure_falls_back_to_pytorch(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo.engine"
            pt = Path(tmp) / "yolo.pt"
            engine.write_bytes(b"engine")
            pt.write_bytes(b"pt")

            loaded_paths: list[str] = []

            class FakeYOLO:
                def __init__(self, model_path):
                    loaded_paths.append(str(model_path))
                    if str(model_path).endswith(".engine"):
                        raise RuntimeError("TensorRT init failed: incompatible engine")
                    self.model_path = model_path

            detector = YoloPoseDetector(
                str(engine),
                yolo_cls=FakeYOLO,
                engine_validator=lambda path: EngineValidationResult(True, str(path), None),
            )

            self.assertEqual(detector.runtime, RUNTIME_PYTORCH_FALLBACK)
            self.assertEqual(Path(detector.model_path), pt)
            self.assertEqual(loaded_paths, [str(engine), str(pt)])
            self.assertTrue(detector.fallback_occurred)

    def test_successful_tensorrt_path_sets_tensorrt_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo.engine"
            engine.write_bytes(b"engine")

            class FakeYOLO:
                def __init__(self, model_path):
                    self.model_path = model_path

            detector = YoloPoseDetector(
                str(engine),
                yolo_cls=FakeYOLO,
                engine_validator=lambda path: EngineValidationResult(True, str(path), None),
            )

            self.assertEqual(detector.runtime, RUNTIME_TENSORRT)
            self.assertEqual(Path(detector.model_path), engine)
            self.assertFalse(detector.fallback_occurred)
            self.assertTrue(detector.engine_validation["ok"])

    def test_production_path_uses_ultralytics_when_raw_trt_probe_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo.engine"
            engine.write_bytes(b"engine")
            loaded = []

            class FakeYOLO:
                def __init__(self, model_path):
                    loaded.append(str(model_path))
                    self.model_path = model_path

            with patch(
                "ai.inference.tensorrt_runtime.deserialize_tensorrt_engine",
                return_value=EngineValidationResult(
                    ok=False,
                    path=str(engine),
                    reason="magicTag mismatch",
                    raw_deserialize_ok=False,
                    validation_method="raw_tensorrt_deserialize",
                ),
            ):
                detector = YoloPoseDetector(str(engine), yolo_cls=FakeYOLO)

            self.assertEqual(detector.runtime, RUNTIME_TENSORRT)
            self.assertEqual(Path(detector.model_path), engine)
            self.assertFalse(detector.fallback_occurred)
            self.assertTrue(detector.engine_validation["ok"])
            self.assertFalse(detector.engine_validation.get("raw_deserialize_ok"))
            self.assertEqual(detector.engine_validation.get("validation_method"), "ultralytics_yolo_load")
            self.assertEqual(loaded, [str(engine)])

    def test_both_tensorrt_and_pytorch_fail_preserves_tensorrt_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "yolo.engine"
            pt = Path(tmp) / "yolo.pt"
            engine.write_bytes(b"engine")
            pt.write_bytes(b"pt")

            class FakeYOLO:
                def __init__(self, model_path):
                    raise RuntimeError(f"cannot load {model_path}")

            with self.assertRaises(RuntimeError) as ctx:
                YoloPoseDetector(
                    str(engine),
                    yolo_cls=FakeYOLO,
                    engine_validator=lambda path: EngineValidationResult(True, str(path), None),
                )
            message = str(ctx.exception)
            self.assertIn("cannot load", message)
            self.assertTrue(
                "tensorrt" in message.lower() or "engine" in message.lower() or "fallback" in message.lower()
            )


class RuntimeSummaryFieldsTest(unittest.TestCase):
    def test_detector_runtime_summary_fields(self):
        from ai.inference.tensorrt_runtime import detector_runtime_summary

        detector = SimpleNamespace(
            runtime=RUNTIME_PYTORCH_FALLBACK,
            model_path="yolo26n-pose.pt",
            engine_validation={"ok": False, "reason": "deserialize failed"},
            fallback_occurred=True,
            model_name="yolo26n-pose.pt",
        )
        summary = detector_runtime_summary(detector, requested_model="yolo26n-pose.engine")
        self.assertEqual(summary["runtime"], RUNTIME_PYTORCH_FALLBACK)
        self.assertEqual(summary["model_path"], "yolo26n-pose.pt")
        self.assertEqual(summary["engine_validation"]["ok"], False)
        self.assertIn("deserialize", summary["engine_validation"]["reason"])

    def test_run_summary_includes_runtime_fields_with_mock_detector(self):
        """Summary builder path used by run_rtsp_inference must be backward compatible."""
        from ai.inference.tensorrt_runtime import attach_runtime_summary_fields

        base = {
            "rtsp_url": "rtsp://x",
            "camera_id": "cam_01",
            "yolo_model": "yolo26n-pose.engine",
            "frames_processed": 0,
        }
        detector = SimpleNamespace(
            runtime=RUNTIME_TENSORRT,
            model_path="yolo26n-pose.engine",
            engine_validation={"ok": True, "reason": None},
            fallback_occurred=False,
        )
        attach_runtime_summary_fields(base, detector, requested_model="yolo26n-pose.engine")
        self.assertEqual(base["runtime"], RUNTIME_TENSORRT)
        self.assertEqual(base["backend"], RUNTIME_TENSORRT)
        self.assertEqual(base["model_path"], "yolo26n-pose.engine")
        self.assertTrue(base["engine_validation"]["ok"])
        self.assertIn("inference_count", base)
        self.assertIn("avg_latency_ms", base)
        self.assertIn("fps", base)
        # existing keys preserved
        self.assertEqual(base["camera_id"], "cam_01")
        self.assertEqual(base["yolo_model"], "yolo26n-pose.engine")
        # JSON serializable
        json.dumps(base)


if __name__ == "__main__":
    unittest.main()
