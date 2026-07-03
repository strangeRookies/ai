"""
카메라 RTSP 연결 상태를 MQTT safety/cameras/status 토픽으로 발행하는 퍼블리셔.

백엔드 CameraStatusEventDto 스펙에 맞춘 페이로드를 발행한다.
- 상태 전환이 있을 때만 발행 (CONNECTED → DISCONNECTED 등)
- 이전 상태와 동일하면 발행하지 않는다
"""

import json
import os
import sys
import time


# safety/cameras/status 토픽으로 발행하는 상태값
# 백엔드 CameraConnectionStatus enum과 정확히 일치해야 한다
STATUS_CONNECTED    = "CONNECTED"
STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_RECONNECTING = "RECONNECTING"
STATUS_ERROR        = "ERROR"
STATUS_DISABLED     = "DISABLED"


def build_camera_status_payload(
    camera_login_id: str,
    status: str,
    previous_status: str | None = None,
    reason: str | None = None,
    edge_device_id: str | None = None,
    rtsp_url_masked: str | None = None,
    camera_id: str | None = None,
) -> dict:
    """
    백엔드 CameraStatusEventDto 스펙에 맞는 페이로드를 생성한다.

    Args:
        camera_login_id: 백엔드 DB cameras.camera_login_id 와 일치해야 함 (필수)
        status:          현재 연결 상태 (CONNECTED|DISCONNECTED|RECONNECTING|ERROR|DISABLED)
        previous_status: 이전 연결 상태 (선택)
        reason:          상태 변경 사유 e.g. RTSP_TIMEOUT, RECONNECT_ATTEMPT, AUTH_FAILED
        edge_device_id:  AI 엣지 서버 식별자 (선택)
        rtsp_url_masked: RTSP URL 마스킹 버전 (선택, 보안상 패스워드 제거)
        camera_id:       AI 내부 카메라 ID (camera_login_id와 다를 수 있음)
    """
    payload = {
        "message_type": "CAMERA_STATUS",
        "camera_login_id": camera_login_id,
        "status": status,
        "detected_at": _utc_iso_now(),
    }
    if previous_status is not None:
        payload["previous_status"] = previous_status
    if reason is not None:
        payload["reason"] = reason
    if edge_device_id is not None:
        payload["edge_device_id"] = edge_device_id
    if rtsp_url_masked is not None:
        payload["rtsp_url_masked"] = rtsp_url_masked
    if camera_id is not None:
        payload["camera_id"] = camera_id
    return payload


def _utc_iso_now() -> str:
    """현재 UTC 시각을 ISO-8601 문자열로 반환한다."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mask_rtsp_url(rtsp_url: str | None) -> str | None:
    """RTSP URL에서 패스워드를 마스킹한다."""
    if not rtsp_url:
        return None
    try:
        # rtsp://user:password@host:port/path → rtsp://user:***@host:port/path
        import re
        return re.sub(r"(:)([^/@]+)(@)", r"\1***\3", rtsp_url)
    except Exception:
        return "rtsp://***"


class CameraStatusPublisher:
    """
    카메라 RTSP 연결 상태 변화를 감지해 MQTT safety/cameras/status 토픽으로 발행.

    상태가 바뀔 때만 발행한다 (중복 발행 방지).
    """

    CAMERA_STATUS_TOPIC = "safety/cameras/status"

    def __init__(self, mqtt_publisher, camera_login_id: str,
                 edge_device_id: str | None = None, rtsp_url: str | None = None,
                 status_topic: str | None = None):
        """
        Args:
            mqtt_publisher:   MqttEventPublisher 인스턴스 (publish 메서드가 있으면 모두 수용)
            camera_login_id:  DB cameras.camera_login_id 와 일치하는 식별자
            edge_device_id:   엣지 디바이스 ID (서버 hostname 등)
            rtsp_url:         RTSP URL (로깅용, 마스킹 후 페이로드에 포함)
            status_topic:     카메라 상태 발행 토픽 (기본: safety/cameras/status)
        """
        self._publisher = mqtt_publisher
        self._camera_login_id = camera_login_id
        self._edge_device_id = edge_device_id or os.getenv("EDGE_DEVICE_ID", "edge-ai-01")
        self._rtsp_url_masked = _mask_rtsp_url(rtsp_url)
        self._status_topic = status_topic or os.getenv("MQTT_STATUS_TOPIC", self.CAMERA_STATUS_TOPIC)
        self._current_status: str | None = None  # 마지막으로 발행한 상태


    def notify_connected(self):
        """RTSP 연결 성공 시 호출."""
        self._transition(STATUS_CONNECTED, reason="RTSP_CONNECTED")

    def notify_disconnected(self, reason: str = "RTSP_TIMEOUT"):
        """RTSP 프레임 수신 중단 / 연결 끊김 시 호출."""
        self._transition(STATUS_DISCONNECTED, reason=reason)

    def notify_reconnecting(self, attempt: int = 1):
        """재연결 시도 시 호출."""
        self._transition(STATUS_RECONNECTING, reason=f"RECONNECT_ATTEMPT_{attempt}")

    def notify_error(self, reason: str = "UNKNOWN_ERROR"):
        """인증 실패, 주소 오류 등 명확한 장애 시 호출."""
        self._transition(STATUS_ERROR, reason=reason)

    def _transition(self, new_status: str, reason: str | None = None):
        """상태가 변경된 경우에만 MQTT 발행."""
        previous = self._current_status
        if previous == new_status:
            return  # 동일 상태 중복 발행 방지

        payload = build_camera_status_payload(
            camera_login_id=self._camera_login_id,
            status=new_status,
            previous_status=previous,
            reason=reason,
            edge_device_id=self._edge_device_id,
            rtsp_url_masked=self._rtsp_url_masked,
        )

        # topic을 직접 지정해서 발행 (MqttEventPublisher는 self.topic으로 발행하므로 우회)
        if hasattr(self._publisher, "client") and self._publisher.client is not None:
            try:
                self._publisher.client.publish(
                    self._status_topic,
                    json.dumps(payload, ensure_ascii=False),
                    qos=0,
                )
                print(
                    f"[camera-status] published: {previous} → {new_status}, "
                    f"camera_login_id={self._camera_login_id}, reason={reason}",
                    flush=True,
                )
                self._current_status = new_status
            except Exception as exc:
                print(
                    f"[camera-status] publish failed: {exc}",
                    file=sys.stderr, flush=True,
                )
        else:
            # console fallback
            print(
                f"[camera-status][mock] {previous} → {new_status}: {json.dumps(payload, ensure_ascii=False)}",
                flush=True,
            )
            self._current_status = new_status
