import argparse
import asyncio
import json
import os
import psutil
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def run_benchmark(mode, cameras, duration, results_dir, run_id):
    html_path = Path(__file__).resolve().parent.parent / 'benchmark' / 'streaming_test_page.html'
    html_url = html_path.as_uri()

    raw_dir = results_dir / 'raw' / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_dir / f"{mode}_{len(cameras)}_camera_raw.jsonl"
    sys_metrics_file = raw_dir / "system_metrics.csv"

    if not sys_metrics_file.exists():
        with open(sys_metrics_file, 'w') as f:
            f.write("timestamp,mode,cameras,cpu_percent,memory_mb\n")

    print(f"=========================================")
    print(f"Starting [{mode.upper()}] benchmark (Run ID: {run_id})")
    print(f"Cameras: {cameras}")
    print(f"Duration: {duration}s")
    print(f"=========================================")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            '--use-fake-ui-for-media-stream',
            '--autoplay-policy=no-user-gesture-required'
        ])
        context = await browser.new_context()
        
        pages = []
        
        # 1. Concurrent Page Loading
        async def load_page(cam):
            page = await context.new_page()
            page.on("console", lambda msg, c=cam: print(f"[{c} BROWSER] {msg.text}"))
            page.on("pageerror", lambda err, c=cam: print(f"[{c} BROWSER ERROR] {err.message}"))
            
            if mode == 'hls':
                stream_url = f"http://127.0.0.1:8888/{cam}/index.m3u8"
            else:
                stream_url = f"http://127.0.0.1:8889/{cam}/whep"
                
            test_url = f"{html_url}?url={stream_url}&mode={mode}"
            await page.goto(test_url)
            return {'cam': cam, 'page': page}

        print(f"[PREPARE] Loading {len(cameras)} camera pages concurrently...")
        pages = await asyncio.gather(*[load_page(cam) for cam in cameras])

        # 2. Wait for all pages to become ready
        async def wait_ready(item):
            for _ in range(50): # 5s timeout
                try:
                    ready = await item['page'].evaluate("window.benchmarkReady")
                    if ready:
                        return True
                except:
                    pass
                await asyncio.sleep(0.1)
            return False

        ready_statuses = await asyncio.gather(*[wait_ready(item) for item in pages])
        if not all(ready_statuses):
            print("[WARNING] Some pages did not signal readiness in time!")

        await asyncio.sleep(1)

        # 3. Synchronized Trigger
        print("[TRIGGER] Dispatching start triggers simultaneously to all pages...")
        trigger_tasks = [item['page'].evaluate("window.__START_BENCHMARK__()") for item in pages]
        await asyncio.gather(*trigger_tasks)
        
        # 4. Measure metrics
        with open(output_file, 'w') as f_out, open(sys_metrics_file, 'a') as f_sys:
            start_time = time.time()
            last_sys_time = start_time
            
            while time.time() - start_time < duration:
                await asyncio.sleep(1)
                current_time = time.time()
                
                # System metrics
                if current_time - last_sys_time >= 1:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().used / (1024 * 1024)
                    f_sys.write(f"{current_time},{mode},{len(cameras)},{cpu},{mem}\n")
                    last_sys_time = current_time

                # Browser stats collection
                for item in pages:
                    try:
                        stats = await item['page'].evaluate("window.benchmarkStats")
                        record = {
                            "timestamp": current_time,
                            "camera": item['cam'],
                            "mode": mode,
                            "stats": stats
                        }
                        f_out.write(json.dumps(record) + "\n")
                    except Exception as e:
                        print(f"[ERROR] Failed evaluating stats for {item['cam']}: {e}")

        await browser.close()
    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Streaming Benchmark Script")
    parser.add_argument('--mode', choices=['hls', 'webrtc', 'both'], required=True)
    parser.add_argument('--cameras', nargs='+', required=True)
    parser.add_argument('--duration', type=int, default=15)
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / 'benchmark' / 'results' / 'streaming' / 'hls_vs_webrtc'
    run_id = args.run_id if args.run_id else f"run_{int(time.time())}"

    if args.mode == 'both':
        asyncio.run(run_benchmark('hls', args.cameras, args.duration, results_dir, run_id))
        asyncio.run(run_benchmark('webrtc', args.cameras, args.duration, results_dir, run_id))
    else:
        asyncio.run(run_benchmark(args.mode, args.cameras, args.duration, results_dir, run_id))
