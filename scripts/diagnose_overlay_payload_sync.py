from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.publishers.mqtt_payloads import (
    build_confirmed_event_payload,
    build_frame_sync_payload,
    build_overlay_payload,
)

REQUIRED_SYNC_FIELDS: Final = (
    "cameraLoginId",
    "frameId",
    "timestampMs",
    "capturedAtMs",
    "processedAtMs",
    "publishedAtMs",
    "frameWidth",
    "frameHeight",
)


def build_diagnostic_samples(camera_login_id: str = "qa_lobby_01") -> dict[str, object]:
    captured_at_ms = 1782180000100
    processed_at_ms = 1782180000120
    published_at_ms = 1782180000123
    frame_id = 4
    overlay = build_overlay_payload(
        stream_id=camera_login_id,
        frame_width=640,
        frame_height=360,
        boxes=[],
        timestamp_ms=published_at_ms,
        frame_id=frame_id,
        captured_at_ms=captured_at_ms,
        processed_at_ms=processed_at_ms,
        published_at_ms=published_at_ms,
        dropped_frame_count=2,
    )
    event = build_confirmed_event_payload(
        stream_id=camera_login_id,
        frame_width=640,
        frame_height=360,
        prediction={"label": "Faint", "score": 0.91},
        sequence={"bbox": [120, 80, 320, 230], "track_id": 7},
        boxes=[],
        timestamp_ms=published_at_ms,
        frame_id=frame_id,
        captured_at_ms=captured_at_ms,
        processed_at_ms=processed_at_ms,
        published_at_ms=published_at_ms,
        dropped_frame_count=2,
    )
    frame_sync = build_frame_sync_payload(
        camera_login_id=camera_login_id,
        frame_id=frame_id,
        captured_at_ms=captured_at_ms,
        processed_at_ms=processed_at_ms,
        published_at_ms=published_at_ms,
        queue_lag_ms=23,
        dropped_frame_count=2,
    )
    normalized = expected_normalized_overlay(overlay, received_at_ms=1782180000158)
    return {
        "rawOverlay": overlay,
        "rawEvent": event,
        "rawFrameSync": frame_sync,
        "expectedNormalizedOverlay": normalized,
        "classification": classify_sync_payload(overlay, normalized),
        "expectedOverlaySyncLog": overlay_sync_log(normalized),
    }


def missing_sync_fields(payload: dict[str, object]) -> list[str]:
    return [field for field in REQUIRED_SYNC_FIELDS if payload.get(field) is None]


def classify_sync_payload(raw_payload: dict[str, object], normalized_payload: dict[str, object]) -> str:
    if missing_sync_fields(raw_payload):
        return "upstream_raw_payload_missing_fields"
    if missing_sync_fields(normalized_payload):
        return "frontend_normalization_missing_fields"
    return "sync_fields_present"


def expected_normalized_overlay(raw_payload: dict[str, object], received_at_ms: int) -> dict[str, object]:
    published_at_ms = int(raw_payload["publishedAtMs"])
    captured_at_ms = int(raw_payload["capturedAtMs"])
    return {
        "cameraLoginId": raw_payload["cameraLoginId"],
        "messageType": raw_payload["messageType"],
        "timestampMs": raw_payload["timestampMs"],
        "frameId": raw_payload["frameId"],
        "frameWidth": raw_payload["frameWidth"],
        "frameHeight": raw_payload["frameHeight"],
        "capturedAtMs": captured_at_ms,
        "processedAtMs": raw_payload["processedAtMs"],
        "publishedAtMs": published_at_ms,
        "receivedAtMs": received_at_ms,
        "networkLatencyMs": received_at_ms - published_at_ms,
        "endToEndLatencyMs": received_at_ms - captured_at_ms,
        "overlayBufferSize": 2,
    }


def overlay_sync_log(normalized_payload: dict[str, object]) -> str:
    return (
        f"[overlay-sync] camera={normalized_payload['cameraLoginId']} "
        f"frameId={normalized_payload['frameId']} "
        f"capturedAtMs={normalized_payload['capturedAtMs']} "
        f"publishedAtMs={normalized_payload['publishedAtMs']} "
        f"receivedAtMs={normalized_payload['receivedAtMs']} "
        f"networkLatencyMs={normalized_payload['networkLatencyMs']} "
        f"endToEndLatencyMs={normalized_payload['endToEndLatencyMs']} "
        "selectedOverlayAgeMs=120 overlayTimestampDeltaMs=45 "
        f"bufferSize={normalized_payload['overlayBufferSize']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local AI overlay sync payload diagnosis sample.")
    parser.add_argument("--camera-login-id", default="qa_lobby_01")
    args = parser.parse_args()
    print(json.dumps(build_diagnostic_samples(args.camera_login_id), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
