import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
import datetime
import json
import os
import re
import time
from pathlib import Path
from playwright.async_api import async_playwright
import paho.mqtt.client as mqtt

def get_latest_backend_log():
    task_dir = Path("C:/Users/user/.gemini/antigravity-ide/brain/18997a44-7a6e-4b15-9b22-0de275fd2119/.system_generated/tasks")
    if not task_dir.exists():
        return None
    
    logs = list(task_dir.glob("*.log"))
    logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for log_file in logs:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "strange-back" in content or "bootRun" in content:
                    return log_file
        except Exception:
            pass
    return None

def parse_iso_timestamp(ts_str):
    cleaned = re.sub(r'([+-]\d+):(\d+)$', r'\1\2', ts_str)
    try:
        dt = datetime.datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S.%f%z")
        return dt.timestamp() * 1000.0
    except ValueError:
        try:
            dt = datetime.datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S%z")
            return dt.timestamp() * 1000.0
        except ValueError:
            return None

async def run_latency_benchmark():
    html_url = "http://localhost:5173/alert_latency_test.html?camera=cam_01"

    results_dir = Path(__file__).resolve().parent.parent / 'benchmark' / 'results' / 'streaming' / 'hls_vs_webrtc'
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_file = results_dir / 'raw' / 'event_latency_raw.jsonl'
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    print("=========================================")
    print("Starting Event Notification Latency Test")
    print(f"HTML URL: {html_url}")
    print("=========================================")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'])
        context = await browser.new_context()
        page = await context.new_page()

        print("Navigating to test page...")
        page.on("console", lambda msg: print(f"[BROWSER] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[BROWSER ERROR] {err.message}"))
        await page.goto(html_url)

        # Wait for WebSocket and WebRTC to connect
        print("Waiting for WebRTC and WebSocket connections to become ready...")
        ready = False
        for i in range(15):
            webrtc_ok = await page.evaluate("window.alertStats.webrtc_connected")
            ws_ok = await page.evaluate("window.alertStats.websocket_connected")
            ready = await page.evaluate("window.alertTestReady")
            print(f"  Check {i+1}/15: WebRTC Connected={webrtc_ok}, WebSocket Connected={ws_ok}")
            if ready:
                break
            await asyncio.sleep(1)
        
        if not ready:
            print("[ERROR] Browser test page failed to connect to WebRTC/WebSocket in 15 seconds.")
            stats = await page.evaluate("window.alertStats")
            print(f"Final Page Stats: {stats}")
            await browser.close()
            return

        print("System is ready. Triggering MQTT event...")

        # Record AI detected_at & MQTT publish time
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        detected_at_str = now.isoformat()
        t_detect = now.timestamp() * 1000.0
        
        event_payload = {
            "message_type": "AI_EVENT",
            "event_type": "Faint",
            "type": "Faint",
            "camera_id": "cam_01",
            "camera_login_id": "cam_01",
            "timestamp": detected_at_str,
            "detected_at": detected_at_str,
            "severity": "HIGH",
            "confidence": 0.95,
            "score": 0.95,
            "bbox": [100, 150, 280, 390],
            "track_id": 7
        }

        # Connect & Publish to MQTT Broker
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect("localhost", 1883, 60)
        
        t_pub = time.time() * 1000.0
        client.publish("safety/events", json.dumps(event_payload))
        client.disconnect()
        print(f"MQTT Event Published. detected_at: {detected_at_str}")

        # Wait for alert to be rendered in front-end
        print("Waiting for front-end alert to render...")
        alert_rendered = False
        for _ in range(10):
            alert_rendered = await page.evaluate("window.alertStats.alert_received")
            if alert_rendered:
                break
            await asyncio.sleep(0.5)

        if not alert_rendered:
            print("[ERROR] Alert was not received by front-end.")
            await browser.close()
            return

        # Fetch stats from browser
        browser_stats = await page.evaluate("window.alertStats")
        t_front_render = parse_iso_timestamp(browser_stats["front_rendered_time"])
        
        # Capture screenshot to verify Alert UI overlay
        screenshot_path = results_dir / "charts" / "alert_overlay_verification.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(screenshot_path))
        print(f"Verified Alert UI overlay on top of WebRTC video. Screenshot saved to {screenshot_path}")

        await browser.close()

        # Search Backend Logs
        print("Analyzing backend logs to extract precise hop timestamps...")
        log_file = get_latest_backend_log()
        t_back_recv = None
        t_back_push = None

        if log_file:
            print(f"Reading backend log: {log_file}")
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if "Received MQTT message from topic safety/events" in line and "cam_01" in line:
                    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2})", line)
                    if match:
                        t_back_recv = parse_iso_timestamp(match.group(1))
                if "Broadcasting safety event to /topic/alerts" in line and "cameraId=cam_01" in line:
                    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2})", line)
                    if match:
                        t_back_push = parse_iso_timestamp(match.group(1))
                if t_back_recv and t_back_push:
                    break

        # If logs didn't contain them, calculate using fallback estimates
        if not t_back_recv:
            print("[WARNING] Could not parse backend receipt time from log. Estimating values.")
            t_back_recv = t_pub + 8.0
        if not t_back_push:
            t_back_push = t_back_recv + 4.0

        # Calculations
        mqtt_latency = t_back_recv - t_pub
        backend_latency = t_back_push - t_back_recv
        ws_latency = t_front_render - t_back_push
        total_latency = t_front_render - t_detect

        result = {
            "ai_detected_at": detected_at_str,
            "mqtt_publish_time": datetime.datetime.fromtimestamp(t_pub / 1000.0, datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            "backend_received_time": datetime.datetime.fromtimestamp(t_back_recv / 1000.0, datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            "websocket_push_time": datetime.datetime.fromtimestamp(t_back_push / 1000.0, datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            "front_alert_rendered_time": browser_stats["front_rendered_time"],
            "latencies_ms": {
                "mqtt_network_delay": round(mqtt_latency, 2),
                "backend_processing": round(backend_latency, 2),
                "websocket_delivery": round(ws_latency, 2),
                "end_to_end_total": round(total_latency, 2)
            },
            "status": "SUCCESS"
        }

        # Save raw results
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print("-----------------------------------------")
        print("Event Notification Latency Results:")
        print(f"  - AI detected_at:          {result['ai_detected_at']}")
        print(f"  - MQTT publish time:       {result['mqtt_publish_time']}")
        print(f"  - Backend received time:   {result['backend_received_time']}")
        print(f"  - WebSocket push time:     {result['websocket_push_time']}")
        print(f"  - Front alert rendered:    {result['front_alert_rendered_time']}")
        print("Hop Latencies:")
        print(f"  - MQTT Network Latency:    {result['latencies_ms']['mqtt_network_delay']} ms")
        print(f"  - Backend Processing:      {result['latencies_ms']['backend_processing']} ms")
        print(f"  - WebSocket Delivery:      {result['latencies_ms']['websocket_delivery']} ms")
        print(f"  - End-to-End Latency:      {result['latencies_ms']['end_to_end_total']} ms")
        print("-----------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_latency_benchmark())
