#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_CAMERAS = ("cam_02", "cam_03", "cam_04", "cam_05")
DEFAULT_OVERLAY_PORTS = tuple(range(8010, 8014))
STREAM_PORTS = (8554, 8888, 8889, 8189)


@dataclass(frozen=True, slots=True)
class WorkerProcess:
    pid: int
    command: str
    camera_id: str | None
    camera_login_id: str | None
    port: int | None
    mqtt_host: str | None
    mqtt_port: int | None
    camera_topic: str | None
    event_topic: str | None
    status_topic: str | None
    stability_fallback: bool
    rtsp_url_env: str | None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cameras = tuple(split_csv(args.cameras)) or DEFAULT_CAMERAS
    overlay_ports = tuple(parse_ports(args.overlay_ports)) or DEFAULT_OVERLAY_PORTS
    print_header("Runtime Port / Socket Audit")
    print_environment(args)
    workers = list_overlay_workers()
    print_workers(workers)
    print_socket_checks(args.host, overlay_ports, args.mqtt_host, args.mqtt_port)
    print_overlay_health(args.host, overlay_ports)
    print_camera_path_checks(args.host, cameras)
    print_mismatch_hints(workers, overlay_ports, cameras, args)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AI worker ports, stream sockets, and MQTT target settings.")
    parser.add_argument("--host", default=os.getenv("DIAG_HOST", "127.0.0.1"))
    parser.add_argument("--cameras", default=os.getenv("DIAG_CAMERAS", ",".join(DEFAULT_CAMERAS)))
    parser.add_argument("--overlay-ports", default=os.getenv("DIAG_OVERLAY_PORTS", "8010-8013"))
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("DIAG_TIMEOUT_SECONDS", "1.5")))
    return parser.parse_args(argv)


def print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_environment(args: argparse.Namespace) -> None:
    print("\n[env]")
    keys = (
        "BACKEND_BASE_URL",
        "RTSP_BASE_URL",
        "MEDIAMTX_RTSP_BASE_URL",
        "MQTT_HOST",
        "MQTT_PORT",
        "MQTT_CAMERA_TOPIC",
        "MQTT_EVENT_TOPIC",
        "MQTT_TOPIC",
        "MQTT_STATUS_TOPIC",
        "TRACKING_STABILITY_FALLBACK",
        "TRACKING_STABILITY_FALLBACK_CAMERA_IDS",
    )
    for key in keys:
        print(f"{key}={os.getenv(key, '(not set)')}")
    print(f"diagnostic_host={args.host}")
    print(f"diagnostic_mqtt={args.mqtt_host}:{args.mqtt_port}")


def list_overlay_workers() -> list[WorkerProcess]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[workers] ps failed: {exc}", file=sys.stderr)
        return []
    workers = []
    for line in result.stdout.splitlines():
        if "serve_ai_overlay.py" not in line or "diagnose_runtime_ports.py" in line:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        workers.append(parse_worker(pid, command))
    return workers


def parse_worker(pid: int, command: str) -> WorkerProcess:
    return WorkerProcess(
        pid=pid,
        command=command,
        camera_id=arg_value(command, "--camera-id"),
        camera_login_id=arg_value(command, "--camera-login-id"),
        port=optional_int(arg_value(command, "--port")),
        mqtt_host=arg_value(command, "--mqtt-host"),
        mqtt_port=optional_int(arg_value(command, "--mqtt-port")),
        camera_topic=arg_value(command, "--mqtt-camera-topic"),
        event_topic=arg_value(command, "--mqtt-event-topic") or arg_value(command, "--mqtt-topic"),
        status_topic=arg_value(command, "--mqtt-status-topic"),
        stability_fallback="--tracking-stability-fallback" in command,
        rtsp_url_env=read_proc_env(pid).get("RTSP_URL"),
    )


def arg_value(command: str, name: str) -> str | None:
    pattern = rf"(?:^|\s){re.escape(name)}(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)"
    match = re.search(pattern, command)
    if not match:
        return None
    return match.group(1).strip("\"'")


def read_proc_env(pid: int) -> dict[str, str]:
    path = Path("/proc") / str(pid) / "environ"
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def print_workers(workers: list[WorkerProcess]) -> None:
    print("\n[serve_ai_overlay workers]")
    if not workers:
        print("NO serve_ai_overlay.py workers found")
        return
    for worker in sorted(workers, key=lambda item: (item.camera_login_id or "", item.pid)):
        print(
            "pid={pid} camera={camera} login={login} port={port} "
            "RTSP_URL={rtsp} mqtt={mqtt_host}:{mqtt_port} cameraTopic={camera_topic} "
            "eventTopic={event_topic} statusTopic={status_topic} stabilityFallback={fallback}".format(
                pid=worker.pid,
                camera=worker.camera_id,
                login=worker.camera_login_id,
                port=worker.port,
                rtsp=worker.rtsp_url_env,
                mqtt_host=worker.mqtt_host,
                mqtt_port=worker.mqtt_port,
                camera_topic=worker.camera_topic,
                event_topic=worker.event_topic,
                status_topic=worker.status_topic,
                fallback=worker.stability_fallback,
            )
        )


def print_socket_checks(host: str, overlay_ports: tuple[int, ...], mqtt_host: str, mqtt_port: int) -> None:
    print("\n[tcp connect]")
    for port in (*STREAM_PORTS, *overlay_ports):
        print_status(f"{host}:{port}", tcp_connect(host, port))
    print_status(f"{mqtt_host}:{mqtt_port} mqtt", tcp_connect(mqtt_host, mqtt_port))


def print_overlay_health(host: str, overlay_ports: tuple[int, ...]) -> None:
    print("\n[overlay health]")
    for port in overlay_ports:
        url = f"http://{host}:{port}/health"
        ok, body = http_get(url)
        if not ok:
            print(f"{url} FAIL {body}")
            continue
        summary = body.get("summary") if isinstance(body, dict) else {}
        camera = summary.get("camera_id") or summary.get("cameraLoginId") or summary.get("camera_login_id")
        active_tracks = summary.get("active_tracks")
        generated_sequences = summary.get("generated_sequences")
        mqtt_publish_count = summary.get("mqtt_publish_count")
        print(
            f"{url} OK connected={body.get('connected')} camera={camera} "
            f"activeTracks={active_tracks} seq={generated_sequences} mqttCount={mqtt_publish_count}"
        )


def print_camera_path_checks(host: str, cameras: tuple[str, ...]) -> None:
    print("\n[stream paths]")
    for camera in cameras:
        hls_url = f"http://{host}:8888/{camera}/index.m3u8"
        whep_url = f"http://{host}:8889/{camera}/whep"
        hls_ok, hls_body = http_get(hls_url, parse_json=False)
        whep_ok, whep_body = http_get(whep_url, parse_json=False)
        print(f"{camera} HLS={status_text(hls_ok, hls_body)} WHEP={status_text(whep_ok, whep_body)}")


def print_mismatch_hints(
    workers: list[WorkerProcess],
    overlay_ports: tuple[int, ...],
    cameras: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    print("\n[mismatch hints]")
    workers_by_login = {worker.camera_login_id: worker for worker in workers if worker.camera_login_id}
    used_ports = {worker.port for worker in workers if worker.port is not None}
    missing_ports = [port for port in overlay_ports if port not in used_ports]
    if missing_ports:
        print(f"overlay ports without worker: {missing_ports}")
    for camera in cameras:
        worker = workers_by_login.get(camera)
        if worker is None:
            print(f"{camera}: NO AI worker")
            continue
        expected_rtsp_suffix = f"/{camera}"
        rtsp_ok = bool(worker.rtsp_url_env and worker.rtsp_url_env.rstrip().endswith(expected_rtsp_suffix))
        mqtt_ok = (
            (worker.mqtt_host or args.mqtt_host) == args.mqtt_host
            and int(worker.mqtt_port or args.mqtt_port) == int(args.mqtt_port)
        )
        print(
            f"{camera}: workerPort={worker.port} rtspPathOk={rtsp_ok} "
            f"mqttTargetOk={mqtt_ok} stabilityFallback={worker.stability_fallback}"
        )
    print("If MQTT publish logs show rc=0 but backend sees nothing, compare AI mqttTarget with backend subscriber broker.")
    print("If frontend video works but bbox/alarm does not, compare MQTT camera/event topics and backend subscriptions.")


def tcp_connect(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            return True, "open"
    except OSError as exc:
        return False, str(exc)


def http_get(url: str, parse_json: bool = True) -> tuple[bool, dict | str]:
    try:
        with urlopen(url, timeout=1.5) as response:
            body = response.read(4096)
            if not parse_json:
                return True, f"HTTP {response.status}"
            return True, json.loads(body.decode("utf-8"))
    except (OSError, URLError, ValueError) as exc:
        return False, str(exc)


def print_status(label: str, result: tuple[bool, str]) -> None:
    ok, detail = result
    print(f"{label} {'OPEN' if ok else 'FAIL'} {detail}")


def status_text(ok: bool, detail: dict | str) -> str:
    if ok:
        return str(detail)
    return f"FAIL {detail}"


def optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ports(value: str | None) -> list[int]:
    ports = []
    for item in split_csv(value):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            ports.extend(range(int(start_text), int(end_text) + 1))
        else:
            ports.append(int(item))
    return ports


if __name__ == "__main__":
    raise SystemExit(main())
