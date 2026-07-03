#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json

def mask_rtsp_url(url: str) -> str:
    if not url:
        return "none"
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            netloc = ""
            if parsed.username:
                netloc += "***"
            if parsed.password:
                netloc += ":***"
            netloc += "@" + (parsed.hostname or "")
            if parsed.port:
                netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=netloc)
            return urlunparse(parsed)
    except Exception:
        pass
    return url

def main():
    print("=" * 60)
    print("  strange-ai MQTT & Diagnostics flow audit  ")
    print("=" * 60)

    # 1. Environment Variable Check
    print("\n[1] Environment Variables:")
    env_keys = [
        "MQTT_HOST", "MQTT_PORT", "MQTT_TOPIC", "MQTT_CAMERA_TOPIC",
        "MQTT_EVENT_TOPIC", "MQTT_STATUS_TOPIC", "MQTT_CLIENT_ID_PREFIX",
        "MQTT_USERNAME", "EDGE_DEVICE_ID", "TRACKING_DEBUG",
        "SELECTED_TRACK_ID", "SELECTED_TRACK_MODE", "SELECTED_TRACK_MISSING_FRAMES"
    ]
    for key in env_keys:
        val = os.getenv(key)
        if val is not None:
            if "PASSWORD" in key:
                print(f"  {key} = *** (masked)")
            else:
                print(f"  {key} = {val}")
        else:
            print(f"  {key} = (not set)")

    # 2. Topic Audit Resolution
    print("\n[2] Topic Audit Resolution:")
    from ai.publishers.event_publisher import mqtt_topic_settings_from_args, mqtt_settings_from_env
    
    # Mocking argparse args
    class MockArgs:
        mqtt_camera_topic = os.getenv("MQTT_CAMERA_TOPIC")
        mqtt_event_topic = os.getenv("MQTT_EVENT_TOPIC")
        mqtt_topic = os.getenv("MQTT_TOPIC")
        mqtt_status_topic = os.getenv("MQTT_STATUS_TOPIC")
        camera_id = "cam_01"
        camera_login_id = "cam_01"
        dry_run = False
        publisher = "mqtt"
        mqtt_host = os.getenv("MQTT_HOST")
        mqtt_port = os.getenv("MQTT_PORT")
    
    args = MockArgs()
    topic_settings = mqtt_topic_settings_from_args(args)
    settings = mqtt_settings_from_env()
    
    mqtt_host = args.mqtt_host or settings["host"]
    mqtt_port = args.mqtt_port or settings["port"]
    status_topic = args.mqtt_status_topic or settings["status_topic"]
    
    print(f"  Target MQTT Broker : {mqtt_host}:{mqtt_port}")
    print(f"  Resolved camera_topic (frame sync) : {topic_settings['camera_topic']}")
    print(f"  Resolved event_topic (confirmed)   : {topic_settings['event_topic']}")
    print(f"  Resolved status_topic (status)     : {status_topic}")

    # 3. Log Pattern Diagnostic Guidance
    print("\n[3] Diagnostics & Log Format Guides:")
    print("  - Startup Audit Log Format:")
    print("    [mqtt-topic-audit] cameraLoginId=cam_01 streamId=cam_01 host=localhost port=1883 camera_topic=camera event_topic=event status_topic=safety/cameras/status publisher=mqtt")
    print("  - Publish Success Log Format:")
    print("    [mqtt-publish-success] topic=camera, messageType=frame_sync, cameraLoginId=cam_01, streamId=cam_01, frameId=10, eventId=none, connected=True, rc=0, payload_bytes=142")
    print("  - Precise Exit Diagnostic Log Format:")
    print("    [registered-cameras][warning] worker exited; stopping camera=cam_01 | cameraLoginId=cam_01 | streamId=cam_01 | rtsp_url=rtsp://***:***@localhost/cam_01 | command=python scripts/serve_ai_overlay.py ...")

    # 4. Long Running QA Commands
    print("\n[4] Long-Running QA verification Commands:")
    print("  * Run mosquitto sub on all topics:")
    print("    mosquitto_sub -h localhost -p 1883 -t \"#\" -v")
    print("  * Run mosquitto sub on camera topic specifically:")
    print("    mosquitto_sub -h localhost -p 1883 -t \"camera\" -v")
    print("  * Run mosquitto sub on event topic specifically:")
    print("    mosquitto_sub -h localhost -p 1883 -t \"event\" -v")
    print("  * Run mosquitto sub on camera status topic specifically:")
    print("    mosquitto_sub -h localhost -p 1883 -t \"safety/cameras/status\" -v")
    print("  * Start AI workers with Selected Track Strict options:")
    print("    powershell -Command \"$env:TRACKING_DEBUG='true'; $env:SELECTED_TRACK_ID='1'; $env:SELECTED_TRACK_MODE='strict'; $env:SELECTED_TRACK_MISSING_FRAMES='5'; python scripts/run_registered_cameras.py\"")
    print("=" * 60)

if __name__ == "__main__":
    main()
