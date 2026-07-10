from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


SAFE_STREAM_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


def build_overlay_frame_message(
    overlay_payload: Mapping[str, Any],
    *,
    video_queued_at_ms: int | None = None,
    metadata_sent_at_ms: int | None = None,
) -> dict[str, Any]:
    """Create the DataChannel message for one processed video frame.

    The MQTT overlay payload is left unchanged; this creates a separate
    WebRTC-specific envelope so existing backend/STOMP contracts are not
    affected.
    """

    video_queued_at_ms = current_timestamp_ms() if video_queued_at_ms is None else int(video_queued_at_ms)
    metadata_sent_at_ms = (
        current_timestamp_ms() if metadata_sent_at_ms is None else int(metadata_sent_at_ms)
    )
    message = dict(overlay_payload)
    message["messageType"] = "overlay_frame"
    message["videoQueuedAtMs"] = video_queued_at_ms
    message["metadataSentAtMs"] = metadata_sent_at_ms
    message["syncTransport"] = "webrtc-datachannel"
    return message


@dataclass(frozen=True, slots=True)
class WebRtcSyncConfig:
    host: str
    port: int
    stream_id: str
    token: str | None = None
    target_fps: float = 15.0


class WebRtcSyncServer:
    """Optional AI-origin WebRTC video + DataChannel sync server.

    This class intentionally imports aiortc/aiohttp lazily so the existing AI
    worker can run unchanged unless --webrtc-sync-enabled is set.
    """

    def __init__(self, config: WebRtcSyncConfig):
        if not SAFE_STREAM_ID.match(config.stream_id):
            raise ValueError(f"unsafe stream_id for WebRTC sync: {config.stream_id!r}")
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner = None
        self._pcs: set[Any] = set()
        self._channels: set[Any] = set()
        self._latest_frame: np.ndarray | None = None
        self._latest_message: dict[str, Any] | None = None
        self._sequence = 0
        self._condition: asyncio.Condition | None = None

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/webrtc/{self.config.stream_id}/offer"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._validate_dependencies()
        self._thread = threading.Thread(target=self._run_loop, name="ai-webrtc-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def publish_frame(self, frame: np.ndarray, overlay_payload: Mapping[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        video_queued_at_ms = current_timestamp_ms()
        message = build_overlay_frame_message(
            overlay_payload,
            video_queued_at_ms=video_queued_at_ms,
            metadata_sent_at_ms=video_queued_at_ms,
        )
        frame_copy = np.ascontiguousarray(frame.copy())
        loop.call_soon_threadsafe(self._publish_on_loop, frame_copy, message)

    @staticmethod
    def _validate_dependencies() -> None:
        missing = []
        for module_name in ("aiohttp", "aiortc", "av"):
            try:
                __import__(module_name)
            except ImportError:
                missing.append(module_name)
        if missing:
            raise RuntimeError(
                "WebRTC sync requires optional dependencies: "
                + ", ".join(missing)
                + ". Install project requirements before enabling AI_WEBRTC_SYNC_ENABLED."
            )

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._condition = asyncio.Condition()
        self._loop.run_until_complete(self._start_app())
        print(
            f"[ai-webrtc-sync] serving streamId={self.config.stream_id} offerUrl={self.url}",
            flush=True,
        )
        self._loop.run_forever()
        self._loop.run_until_complete(self._shutdown())
        self._loop.close()

    async def _start_app(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "WebRTC sync requires optional dependencies: aiohttp and aiortc"
            ) from exc

        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/webrtc/{stream_id}/offer", self._offer)
        app.router.add_options("/webrtc/{stream_id}/offer", self._options)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, int(self.config.port))
        await site.start()

    async def _shutdown(self) -> None:
        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()
        self._channels.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _health(self, _request):
        from aiohttp import web

        return web.json_response(
            {
                "ok": True,
                "streamId": self.config.stream_id,
                "latestFrameId": (self._latest_message or {}).get("frameId"),
                "clientCount": len(self._pcs),
            },
            headers=self._cors_headers(),
        )

    async def _options(self, _request):
        from aiohttp import web

        return web.Response(status=204, headers=self._cors_headers())

    async def _offer(self, request):
        from aiohttp import web
        from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription

        stream_id = request.match_info["stream_id"]
        if stream_id != self.config.stream_id:
            return web.json_response({"error": "unknown stream"}, status=404, headers=self._cors_headers())
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401, headers=self._cors_headers())

        body = await request.json()
        offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
        pc = RTCPeerConnection()
        self._pcs.add(pc)
        pc.addTrack(create_latest_frame_video_track(self))

        def register_channel(channel: RTCDataChannel) -> None:
            self._channels.add(channel)

            @channel.on("open")
            def _on_open() -> None:
                if self._latest_message is not None:
                    channel.send(json.dumps(self._latest_message, ensure_ascii=False))

            @channel.on("close")
            def _on_close() -> None:
                self._channels.discard(channel)

        @pc.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            register_channel(channel)

        @pc.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                self._pcs.discard(pc)
                await pc.close()

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            headers=self._cors_headers(),
        )

    def _authorized(self, request) -> bool:
        token = self.config.token
        if not token:
            return True
        header = request.headers.get("Authorization", "")
        if header == f"Bearer {token}":
            return True
        return request.query.get("token") == token

    def _cors_headers(self) -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        }

    def _publish_on_loop(self, frame: np.ndarray, message: dict[str, Any]) -> None:
        metadata_sent_at_ms = current_timestamp_ms()
        message = dict(message)
        message["metadataSentAtMs"] = metadata_sent_at_ms
        self._latest_frame = frame
        self._latest_message = message
        self._sequence += 1
        if self._condition is not None:
            asyncio.create_task(self._notify_frame())
        encoded = json.dumps(message, ensure_ascii=False)
        for channel in list(self._channels):
            if getattr(channel, "readyState", None) == "open":
                try:
                    channel.send(encoded)
                except Exception:
                    self._channels.discard(channel)

    async def _notify_frame(self) -> None:
        assert self._condition is not None
        async with self._condition:
            self._condition.notify_all()

    async def wait_for_frame(self, last_sequence: int) -> tuple[int, np.ndarray]:
        assert self._condition is not None
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._sequence != last_sequence and self._latest_frame is not None
            )
            return self._sequence, self._latest_frame.copy()


def create_latest_frame_video_track(server: WebRtcSyncServer):
    from aiortc import VideoStreamTrack

    class LatestFrameVideoTrack(VideoStreamTrack):
        def __init__(self, sync_server: WebRtcSyncServer):
            super().__init__()
            self._server = sync_server
            self._last_sequence = 0

        async def recv(self):
            from av import VideoFrame

            pts, time_base = await self.next_timestamp()
            self._last_sequence, frame = await self._server.wait_for_frame(self._last_sequence)
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    return LatestFrameVideoTrack(server)
