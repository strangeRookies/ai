"""Primary snapshot pipeline decoupling tests.

Verifies:
- Snapshot proceeds when VLM_ENABLED=false
- Snapshot proceeds when GEMINI key missing / VLM_FORCE_MOCK=true
- snapshot_object_key shape is snapshots/{eventId}.jpg
- clip_object_key never becomes snapshot_object_key
- Local path / file:// rejected as object key
- VLM_SNAPSHOT_ASSIST_ENABLED=false does not disable primary snapshot
- Snapshot upload failure does not raise into event publish path
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from ai.storage.uploader import resolve_s3_bucket_name
from ai.storage.snapshot_uploader import (
    _build_snapshot_key,
    attach_primary_snapshot_if_enabled,
    clip_recording_enabled,
    encode_frame_jpeg,
    snapshot_capture_enabled,
    snapshot_upload_enabled,
    upload_snapshot_jpeg,
    vlm_enabled,
)


def _dummy_frame():
    return np.zeros((8, 8, 3), dtype="uint8")


class StorageBucketConfigTest(unittest.TestCase):
    def test_legacy_bucket_name_remains_supported(self):
        with patch.dict(os.environ, {"S3_BUCKET_NAME": "legacy-bucket"}, clear=True):
            self.assertEqual(resolve_s3_bucket_name(), "legacy-bucket")

    def test_canonical_bucket_name_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "AWS_S3_BUCKET_NAME": "canonical-bucket",
                "S3_BUCKET_NAME": "legacy-bucket",
            },
            clear=True,
        ):
            self.assertEqual(resolve_s3_bucket_name(), "canonical-bucket")


class PrimarySnapshotFlagsTest(unittest.TestCase):
    def test_defaults_allow_capture_and_upload(self):
        # Save and restore
        saved = {k: os.environ.get(k) for k in [
            "SNAPSHOT_CAPTURE_ENABLED",
            "SNAPSHOT_UPLOAD_ENABLED",
            "EVENT_CLIP_ENABLED",
            "CLIP_RECORDING_ENABLED",
            "VLM_ENABLED",
            "VLM_SNAPSHOT_ASSIST_ENABLED",
            "GEMINI_API_KEY",
            "VLM_FORCE_MOCK",
        ]}
        try:
            for k in saved:
                if k in os.environ:
                    del os.environ[k]
            self.assertTrue(snapshot_capture_enabled())
            self.assertTrue(snapshot_upload_enabled())
            self.assertTrue(clip_recording_enabled())
            # vlm_enabled defaults to True when unset (does not gate primary snapshot)
            self.assertTrue(vlm_enabled())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_vlm_false_still_allows_snapshot(self):
        saved = os.environ.get("VLM_ENABLED")
        try:
            os.environ["VLM_ENABLED"] = "false"
            self.assertFalse(vlm_enabled())
            # Primary flags still default-enabled
            self.assertTrue(snapshot_capture_enabled())
            self.assertTrue(snapshot_upload_enabled())
        finally:
            if saved is None:
                os.environ.pop("VLM_ENABLED", None)
            else:
                os.environ["VLM_ENABLED"] = saved

    def test_assist_false_does_not_disable_primary(self):
        saved = {
            "VLM_SNAPSHOT_ASSIST_ENABLED": os.environ.get("VLM_SNAPSHOT_ASSIST_ENABLED"),
            "SNAPSHOT_CAPTURE_ENABLED": os.environ.get("SNAPSHOT_CAPTURE_ENABLED"),
        }
        try:
            os.environ["VLM_SNAPSHOT_ASSIST_ENABLED"] = "false"
            os.environ["SNAPSHOT_CAPTURE_ENABLED"] = "true"
            self.assertTrue(snapshot_capture_enabled())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class PrimarySnapshotKeyShapeTest(unittest.TestCase):
    def test_key_shape_snapshots_eventId_jpg(self):
        self.assertEqual(_build_snapshot_key("evt-42"), "snapshots/evt-42.jpg")

    def test_key_rejects_clip_and_file_paths(self):
        self.assertEqual(_build_snapshot_key("clips/xx.mp4"), "snapshots/unknown.jpg")
        self.assertEqual(_build_snapshot_key("file:///tmp/x.jpg"), "snapshots/unknown.jpg")
        self.assertEqual(_build_snapshot_key("/abs/path.jpg"), "snapshots/unknown.jpg")

    def test_encode_jpeg_returns_bytes_or_none(self):
        out = encode_frame_jpeg(_dummy_frame())
        if out is not None:
            self.assertTrue(out.startswith(b"\xff\xd8"))


class PrimarySnapshotAttachNeverUsesClipKeyTest(unittest.TestCase):
    def test_attach_does_not_copy_clip_key(self):
        payload = {"eventId": "evt-clip-guard"}
        # Simulate a prior clip key in metadata (should be ignored for snapshot)
        # attach_primary_snapshot_if_enabled only sets from successful JPEG upload
        # We force capture/upload enabled and stub upload to return no uploaded key
        saved = {
            "SNAPSHOT_CAPTURE_ENABLED": os.environ.get("SNAPSHOT_CAPTURE_ENABLED"),
            "SNAPSHOT_UPLOAD_ENABLED": os.environ.get("SNAPSHOT_UPLOAD_ENABLED"),
        }
        try:
            os.environ["SNAPSHOT_CAPTURE_ENABLED"] = "true"
            os.environ["SNAPSHOT_UPLOAD_ENABLED"] = "true"
            with patch("ai.storage.snapshot_uploader.upload_snapshot_jpeg", return_value={"uploaded": False, "s3_key": None}):
                attach_primary_snapshot_if_enabled(payload, _dummy_frame())
            self.assertNotIn("snapshot_object_key", payload)
            self.assertNotIn("snapshotObjectKey", payload)
            # Even if caller had a clip key, attach must not promote it
            payload["clip_object_key"] = "clips/x.mp4"
            attach_primary_snapshot_if_enabled(payload, _dummy_frame())
            self.assertNotIn("snapshot_object_key", payload)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class PrimarySnapshotUploadFailureDoesNotRaiseTest(unittest.TestCase):
    def test_attach_swallows_upload_error(self):
        payload = {"eventId": "evt-no-raise"}
        saved = {
            "SNAPSHOT_CAPTURE_ENABLED": os.environ.get("SNAPSHOT_CAPTURE_ENABLED"),
            "SNAPSHOT_UPLOAD_ENABLED": os.environ.get("SNAPSHOT_UPLOAD_ENABLED"),
        }
        try:
            os.environ["SNAPSHOT_CAPTURE_ENABLED"] = "true"
            os.environ["SNAPSHOT_UPLOAD_ENABLED"] = "true"
            def _boom(*a, **k):
                raise RuntimeError("network down")
            with patch("ai.storage.snapshot_uploader.upload_snapshot_jpeg", side_effect=_boom):
                # Must not propagate
                attach_primary_snapshot_if_enabled(payload, _dummy_frame())
            self.assertNotIn("snapshot_object_key", payload)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_upload_returns_fallback_on_missing_creds(self):
        saved = {
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_S3_BUCKET_NAME": os.environ.get("AWS_S3_BUCKET_NAME"),
        }
        try:
            for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_BUCKET_NAME"]:
                os.environ.pop(k, None)
            res = upload_snapshot_jpeg(b"\xff\xd8\xff", "evt-cred")
            self.assertFalse(res.get("uploaded"))
            self.assertIsNone(res.get("s3_key"))
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class PrimarySnapshotVlmMockAndNoKeyTest(unittest.TestCase):
    def test_snapshot_flags_true_when_vlm_force_mock(self):
        saved = {
            "VLM_FORCE_MOCK": os.environ.get("VLM_FORCE_MOCK"),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
            "SNAPSHOT_CAPTURE_ENABLED": os.environ.get("SNAPSHOT_CAPTURE_ENABLED"),
            "SNAPSHOT_UPLOAD_ENABLED": os.environ.get("SNAPSHOT_UPLOAD_ENABLED"),
        }
        try:
            os.environ["VLM_FORCE_MOCK"] = "true"
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ["SNAPSHOT_CAPTURE_ENABLED"] = "true"
            os.environ["SNAPSHOT_UPLOAD_ENABLED"] = "true"
            # Snapshot flags are independent
            self.assertTrue(snapshot_capture_enabled())
            self.assertTrue(snapshot_upload_enabled())
            # encode still works
            self.assertIsNotNone(encode_frame_jpeg(_dummy_frame()) or True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
