"""Unit tests for production tracking defaults + optional canary override."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai.tracking_canary import (
    PRODUCTION_NEAR_DUP_SUPPRESS_MODE,
    PRODUCTION_NEW_TRACK_THRESH,
    apply_canary_env,
    canary_signature_fragment,
    clear_canary_config,
    production_tracking_defaults,
    resolve_canary_settings,
    write_canary_config,
)


class TrackingCanaryTest(unittest.TestCase):
    def test_default_is_production_not_canary(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            env = {
                "TRACKING_CANARY_CONFIG": str(cfg),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop("TRACKING_CANARY_CAMERA_IDS", None)
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                os.environ.pop("SIMPLE_TRACK_NEW_TRACK_THRESH", None)
                s = resolve_canary_settings("cam_03")
                self.assertFalse(s["canary"])
                self.assertEqual(s["near_dup_suppress_mode"], PRODUCTION_NEAR_DUP_SUPPRESS_MODE)
                self.assertEqual(s["new_track_thresh"], PRODUCTION_NEW_TRACK_THRESH)
                self.assertEqual(s["configSource"], "production-default")

    def test_file_enables_only_listed_camera(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            write_canary_config(
                {
                    "cam_03": {
                        "enabled": True,
                        "near_dup_suppress_mode": "claimed_iou",
                        "new_track_thresh": 0.35,
                    }
                },
                path=cfg,
            )
            with mock.patch.dict(os.environ, {"TRACKING_CANARY_CONFIG": str(cfg)}, clear=False):
                os.environ.pop("TRACKING_CANARY_CAMERA_IDS", None)
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                s3 = resolve_canary_settings("cam_03")
                s2 = resolve_canary_settings("cam_02")
                self.assertTrue(s3["canary"])
                self.assertEqual(s3["near_dup_suppress_mode"], "claimed_iou")
                self.assertEqual(s3["new_track_thresh"], 0.35)
                self.assertFalse(s2["canary"])
                self.assertEqual(s2["near_dup_suppress_mode"], PRODUCTION_NEAR_DUP_SUPPRESS_MODE)

    def test_apply_env_production_for_non_canary(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            clear_canary_config(path=cfg)
            with mock.patch.dict(
                os.environ,
                {"TRACKING_CANARY_CONFIG": str(cfg)},
                clear=False,
            ):
                os.environ.pop("TRACKING_CANARY_CAMERA_IDS", None)
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                os.environ.pop("SIMPLE_TRACK_NEW_TRACK_THRESH", None)
                env2: dict[str, str] = {}
                apply_canary_env("cam_02", env2)
                self.assertEqual(env2["NEAR_DUP_SUPPRESS_MODE"], "hybrid_kp")
                self.assertEqual(env2["SIMPLE_TRACK_NEW_TRACK_THRESH"], "0.3")
                self.assertEqual(env2["TRACKING_CANARY"], "false")

    def test_canary_override_does_not_force_others_to_none(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            write_canary_config(
                {"cam_03": {"enabled": True, "near_dup_suppress_mode": "claimed_iou", "new_track_thresh": 0.35}},
                path=cfg,
            )
            with mock.patch.dict(os.environ, {"TRACKING_CANARY_CONFIG": str(cfg)}, clear=False):
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                env3: dict[str, str] = {}
                env2: dict[str, str] = {}
                apply_canary_env("cam_03", env3)
                apply_canary_env("cam_02", env2)
                self.assertEqual(env3["NEAR_DUP_SUPPRESS_MODE"], "claimed_iou")
                self.assertEqual(env3["TRACKING_CANARY"], "true")
                # Non-canary uses production default hybrid_kp, not forced none
                self.assertEqual(env2["NEAR_DUP_SUPPRESS_MODE"], "hybrid_kp")
                self.assertEqual(env2["TRACKING_CANARY"], "false")

    def test_signature_changes_for_canary_camera(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            with mock.patch.dict(os.environ, {"TRACKING_CANARY_CONFIG": str(cfg)}, clear=False):
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                clear_canary_config(path=cfg)
                before = canary_signature_fragment("cam_03")
                write_canary_config(
                    {"cam_03": {"enabled": True, "near_dup_suppress_mode": "claimed_iou", "new_track_thresh": 0.35}},
                    path=cfg,
                )
                after3 = canary_signature_fragment("cam_03")
                after2 = canary_signature_fragment("cam_02")
                self.assertNotEqual(before, after3)
                self.assertFalse(after2["canary"])
                self.assertEqual(after2["near_dup_suppress_mode"], PRODUCTION_NEAR_DUP_SUPPRESS_MODE)

    def test_clear_canary_returns_to_production(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "canary_config.json"
            write_canary_config(
                {"cam_03": {"enabled": True, "near_dup_suppress_mode": "claimed_iou", "new_track_thresh": 0.35}},
                path=cfg,
            )
            clear_canary_config(path=cfg)
            with mock.patch.dict(os.environ, {"TRACKING_CANARY_CONFIG": str(cfg)}, clear=False):
                os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
                s = resolve_canary_settings("cam_03")
                self.assertFalse(s["canary"])
                self.assertEqual(s["near_dup_suppress_mode"], PRODUCTION_NEAR_DUP_SUPPRESS_MODE)

    def test_production_defaults_helper(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEAR_DUP_SUPPRESS_MODE", None)
            os.environ.pop("SIMPLE_TRACK_NEW_TRACK_THRESH", None)
            d = production_tracking_defaults()
            self.assertEqual(d["near_dup_suppress_mode"], "hybrid_kp")
            self.assertEqual(d["new_track_thresh"], 0.30)


if __name__ == "__main__":
    unittest.main()
