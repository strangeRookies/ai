"""OBJECTIVE tests: TensorRT backend logging + tunable fall lifecycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai.action.fall_event_state import (
    EVENT_TYPE_FAINT_SUSPECTED,
    EVENT_TYPE_FALL_UNRECOVERED,
    FallEventStateMachine,
    LifecycleKind,
)
from ai.action.faint_post_processing import FaintEventPostProcessor
from ai.inference.tensorrt_runtime import (
    RUNTIME_PYTORCH_FALLBACK,
    RUNTIME_TENSORRT,
    EngineValidationResult,
    log_periodic_inference_metrics,
    log_worker_backend_startup,
)
from ai.runtime_metrics import RuntimeMetrics, compute_latency_report
from detector.yolo_pose_detector import YoloPoseDetector
from scripts.benchmark_yolo_backends import parse_args as bench_parse, run_benchmark, write_outputs


class TensorRtBackendLoggingTest(unittest.TestCase):
    def test_tensorrt_success_startup_log_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "m.engine"
            engine.write_bytes(b"eng")

            class FakeYOLO:
                def __init__(self, path):
                    self.path = path

            det = YoloPoseDetector(
                str(engine),
                yolo_cls=FakeYOLO,
                engine_validator=lambda p: EngineValidationResult(True, str(p), None),
            )
            record = log_worker_backend_startup(
                camera_login_id="cam_05",
                requested_model=str(engine),
                detector=det,
                device="0",
                precision="fp32",
            )
            self.assertEqual(record["cameraLoginId"], "cam_05")
            self.assertEqual(record["requested_backend"], RUNTIME_TENSORRT)
            self.assertEqual(record["actual_backend"], RUNTIME_TENSORRT)
            self.assertEqual(record["model_path"], str(engine))
            self.assertEqual(record["engine_path"], str(engine))
            self.assertFalse(record["fallback"])
            self.assertIsNone(record["fallback_reason"])

    def test_tensorrt_failure_logs_fallback_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "m.engine"
            pt = Path(tmp) / "m.pt"
            engine.write_bytes(b"")
            pt.write_bytes(b"pt")

            class FakeYOLO:
                def __init__(self, path):
                    if str(path).endswith(".engine"):
                        raise RuntimeError("engine load fail")
                    self.path = path

            det = YoloPoseDetector(
                str(engine),
                yolo_cls=FakeYOLO,
                engine_validator=lambda p: EngineValidationResult(False, str(p), "engine file is empty (0 bytes)"),
            )
            self.assertEqual(det.runtime, RUNTIME_PYTORCH_FALLBACK)
            self.assertTrue(det.fallback_occurred)
            self.assertTrue(det.tensorrt_error)
            record = log_worker_backend_startup(
                camera_login_id="cam_01",
                requested_model=str(engine),
                detector=det,
                device="0",
            )
            self.assertTrue(record["fallback"])
            self.assertIsNotNone(record["fallback_reason"])
            self.assertIn("empty", str(record["fallback_reason"]).lower())

    def test_periodic_infer_metrics_include_percentiles(self):
        metrics = RuntimeMetrics()
        metrics.set_warmup_skip(0)
        for ms in [10, 12, 11, 50, 13]:
            metrics.add_yolo_ms(ms)
        record = log_periodic_inference_metrics(
            camera_login_id="cam_02",
            backend="tensorrt",
            metrics=metrics,
        )
        self.assertEqual(record["cameraLoginId"], "cam_02")
        self.assertEqual(record["backend"], "tensorrt")
        self.assertEqual(record["frames"], 5)
        self.assertIsNotNone(record["avg_infer_ms"])
        self.assertIsNotNone(record["p50_infer_ms"])
        self.assertIsNotNone(record["p95_infer_ms"])
        self.assertIsNotNone(record["fps"])


class BenchmarkSchemaTest(unittest.TestCase):
    def test_compute_latency_report_schema(self):
        lat = [20.0] * 30 + [10.0] * 100
        rep = compute_latency_report(lat, warmup=20)
        self.assertEqual(rep["frames"], 110)
        self.assertIn("avg_infer_ms", rep)
        self.assertIn("p50_infer_ms", rep)
        self.assertIn("p95_infer_ms", rep)
        self.assertIn("fps", rep)

    def test_benchmark_script_dry_synthetic_writes_json_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "b.json"
            out_csv = Path(tmp) / "b.csv"
            args = bench_parse(
                [
                    "--synthetic",
                    "--frames",
                    "5",
                    "--warmup",
                    "1",
                    "--skip-engine",
                    "--skip-pt",
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                ]
            )
            report = run_benchmark(args)
            write_outputs(report, str(out_json), str(out_csv))
            self.assertTrue(out_json.exists())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("backends", data)
            self.assertTrue(out_csv.exists())
            text = out_csv.read_text(encoding="utf-8")
            self.assertIn("avg_infer_ms", text)
            self.assertIn("p50_infer_ms", text)
            self.assertIn("p95_infer_ms", text)
            self.assertIn("fps", text)


class FallTuningAndPersistentTest(unittest.TestCase):
    def test_no_new_fall_after_cooldown_while_lying_but_persistent_emits(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=5,
            recover_consecutive=4,
            require_upright_to_lying=False,
            use_posture_estimator=False,
            unrecovered_after_seconds=5.0,
            unrecovered_repeat_seconds=30.0,
        )
        self.assertFalse(proc.should_trigger("cam", {"label": "Faint"}, 1.0, track_id=1))
        self.assertTrue(proc.should_trigger("cam", {"label": "Faint"}, 2.0, track_id=1))
        # Within unrecovered delay: suppress NEW_FALL only (do not call evaluate twice for unrecovered)
        mid = proc.evaluate("cam", {"label": "Faint"}, 4.0, track_id=1, posture_label="lying_like")
        self.assertFalse(mid.emit)
        self.assertFalse(mid.is_new_fall)
        # confirmed at t=2; unrecovered_after=5 → t>=7 emits persistent
        d = proc.evaluate(
            "cam",
            {"label": "Faint"},
            8.0,
            track_id=1,
            posture_label="lying_like",
        )
        self.assertTrue(d.emit)
        self.assertTrue(d.is_unrecovered)
        self.assertIn(d.event_type, {EVENT_TYPE_FAINT_SUSPECTED, EVENT_TYPE_FALL_UNRECOVERED})
        self.assertIsNotNone(d.original_event_id)
        self.assertEqual(d.state, "POST_FALL_LYING")
        # NEW_FALL still blocked after cooldown
        self.assertFalse(proc.should_trigger("cam", {"label": "Faint"}, 20.0, track_id=1))

    def test_movement_low_prefers_faint_suspected(self):
        sm = FallEventStateMachine(min_consecutive_faint=1, unrecovered_after_seconds=1.0)
        sm.update("c", 0.0, track_id=1, is_alert=True, prediction={"label": "Faint"})
        d = sm.update(
            "c",
            2.0,
            track_id=1,
            is_alert=True,
            prediction={"label": "Faint"},
            movement_level="low",
            posture_label="lying_like",
            lying_like=True,
        )
        self.assertEqual(d.kind, LifecycleKind.UNRECOVERED)
        self.assertEqual(d.event_type, EVENT_TYPE_FAINT_SUSPECTED)

    def test_recover_allows_new_fall(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=2,
            cooldown_seconds=1,
            recover_consecutive=2,
            require_upright_to_lying=False,
            use_posture_estimator=False,
            unrecovered_after_seconds=100.0,
        )
        proc.should_trigger("cam", {"label": "Faint"}, 1.0, track_id=2)
        self.assertTrue(proc.should_trigger("cam", {"label": "Faint"}, 2.0, track_id=2))
        proc.evaluate("cam", {"label": "Normal"}, 3.0, track_id=2)
        proc.evaluate("cam", {"label": "Normal"}, 4.0, track_id=2)
        proc.should_trigger("cam", {"label": "Faint"}, 10.0, track_id=2)
        self.assertTrue(proc.should_trigger("cam", {"label": "Faint"}, 11.0, track_id=2))

    def test_reset_clears_persistent_state(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=1,
            require_upright_to_lying=False,
            use_posture_estimator=False,
            unrecovered_after_seconds=1.0,
        )
        proc.should_trigger("cam", {"label": "Faint"}, 1.0, track_id=9)
        proc.evaluate("cam", {"label": "Faint"}, 5.0, track_id=9, posture_label="lying_like")
        proc.reset()
        # after reset, first confirm is NEW_FALL again not unrecovered
        d = proc.evaluate("cam", {"label": "Faint"}, 6.0, track_id=9)
        self.assertTrue(d.is_new_fall)

    def test_multi_track_independence(self):
        sm = FallEventStateMachine(min_consecutive_faint=2)
        sm.update("cam", 1.0, track_id=1, is_alert=True)
        sm.update("cam", 2.0, track_id=1, is_alert=True)
        sm.update("cam", 2.0, track_id=2, is_alert=True)
        d2 = sm.update("cam", 3.0, track_id=2, is_alert=True)
        self.assertEqual(d2.kind, LifecycleKind.NEW_FALL)
        d1 = sm.update("cam", 4.0, track_id=1, is_alert=True)
        self.assertEqual(d1.kind, LifecycleKind.SUPPRESS_NEW_FALL)

    def test_require_upright_blocks_start_lying(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=2,
            require_upright_to_lying=True,
            use_posture_estimator=True,
        )
        lying = {
            "track_id": 3,
            "bbox": [10, 80, 150, 120],
            "pose_horizontal": True,
            "keypoints": [
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 5,
                {"x": 20, "y": 90, "confidence": 0.9},
                {"x": 100, "y": 92, "confidence": 0.9},
                *[{"x": 0, "y": 0, "confidence": 0.0}] * 4,
                {"x": 25, "y": 100, "confidence": 0.9},
                {"x": 95, "y": 102, "confidence": 0.9},
            ],
        }
        for t in range(1, 6):
            self.assertFalse(
                proc.should_trigger("cam", {"label": "Faint"}, float(t), track_id=3, detection=lying)
            )


class FallParamDefaultsTest(unittest.TestCase):
    def test_cli_defaults_match_objective_table(self):
        from scripts.rtsp_inference_args import parse_args

        args = parse_args([])
        self.assertTrue(args.use_fall_state_machine)
        self.assertEqual(args.consecutive_required, 2)
        self.assertEqual(args.normal_recover_required, 4)
        self.assertEqual(args.cooldown_sec, 10.0)
        self.assertEqual(args.persistent_delay_sec, 10.0)
        self.assertEqual(args.persistent_repeat_sec, 30.0)
        self.assertEqual(args.track_lost_grace_sec, 3.0)
        self.assertFalse(args.require_upright_to_lying)
        self.assertEqual(args.lying_aspect_ratio, 1.2)
        self.assertEqual(args.upright_aspect_ratio, 1.3)
        self.assertEqual(args.min_keypoint_conf, 0.3)
        self.assertEqual(args.lying_frames_required, 2)
        self.assertEqual(args.upright_frames_required, 2)

    def test_faint_threshold_gates_alert_prediction(self):
        from ai.action.faint_post_processing import is_alert_prediction

        pred = {"label": "Faint", "score": 0.55, "probabilities": {"Faint": 0.55}}
        self.assertFalse(is_alert_prediction(pred, faint_threshold=0.6))
        self.assertTrue(is_alert_prediction(pred, faint_threshold=0.5))

    def test_track_lost_grace_prunes_state(self):
        proc = FaintEventPostProcessor(
            min_consecutive_faint=1,
            require_upright_to_lying=False,
            use_posture_estimator=False,
            track_lost_grace_sec=3.0,
        )
        proc.should_trigger("cam", {"label": "Faint"}, 1.0, track_id=5)
        self.assertEqual(proc._state_machine.get_state("cam", 5).value, "POST_FALL_LYING")
        pruned = proc.prune_lost_tracks("cam", active_track_ids=[], timestamp=5.0)
        self.assertTrue(pruned)
        self.assertEqual(proc._state_machine.get_state("cam", 5).value, "NORMAL")

    def test_build_faint_post_processor_from_args_wires_params(self):
        from ai.action.fall_lifecycle_config import build_faint_post_processor_from_args

        args = SimpleNamespace(
            consecutive_required=2,
            cooldown_sec=10.0,
            use_fall_state_machine=True,
            normal_recover_required=4,
            require_upright_to_lying=False,
            persistent_delay_sec=10.0,
            persistent_repeat_sec=30.0,
            lying_aspect_ratio=1.2,
            upright_aspect_ratio=1.3,
            min_keypoint_conf=0.3,
            lying_frames_required=2,
            upright_frames_required=2,
            movement_low_threshold=12.0,
            faint_threshold=0.6,
            fall_threshold=0.6,
            track_lost_grace_sec=3.0,
        )
        proc = build_faint_post_processor_from_args(args)
        self.assertEqual(proc.min_consecutive_faint, 2)
        self.assertEqual(proc.cooldown_seconds, 10.0)
        self.assertEqual(proc.faint_threshold, 0.6)
        self.assertEqual(proc.track_lost_grace_sec, 3.0)
        self.assertFalse(proc.require_upright_to_lying)

    def test_serve_ai_overlay_source_has_worker_backend_logging(self):
        # Structural: production overlay path must call startup/periodic helpers.
        text = Path("scripts/serve_ai_overlay.py").read_text(encoding="utf-8")
        self.assertIn("log_worker_backend_startup", text)
        self.assertIn("log_periodic_inference_metrics", text)
        self.assertIn("build_faint_post_processor_from_args", text)


if __name__ == "__main__":
    unittest.main()
