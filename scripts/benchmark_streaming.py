import argparse
import asyncio
import json
import os
import psutil
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def run_benchmark(mode, cameras, duration, results_dir):
    html_path = Path(__file__).resolve().parent.parent / 'benchmark' / 'streaming_test_page.html'
    html_url = html_path.as_uri()

    raw_dir = results_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    cam_str = 'single' if len(cameras) == 1 else 'multi'
    output_file = raw_dir / f"{mode}_{cam_str}_camera_raw.jsonl"
    sys_metrics_file = results_dir / "system_metrics.csv"

    if not sys_metrics_file.exists():
        with open(sys_metrics_file, 'w') as f:
            f.write("timestamp,mode,cameras,cpu_percent,memory_mb\n")

    print(f"=========================================")
    print(f"Starting [{mode.upper()}] benchmark")
    print(f"Cameras: {cameras}")
    print(f"Duration: {duration}s")
    print(f"=========================================")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'])
        context = await browser.new_context()
        
        pages = []
        for cam in cameras:
            page = await context.new_page()
            if mode == 'hls':
                stream_url = f"http://127.0.0.1:8888/{cam}/index.m3u8"
            else:
                stream_url = f"http://127.0.0.1:8889/{cam}/whep"
            
            test_url = f"{html_url}?url={stream_url}&mode={mode}"
            await page.goto(test_url)
            pages.append({'cam': cam, 'page': page})

        with open(output_file, 'w') as f_out, open(sys_metrics_file, 'a') as f_sys:
            start_time = time.time()
            last_sys_time = start_time
            
            while time.time() - start_time < duration:
                await asyncio.sleep(1)
                current_time = time.time()
                
                # Record System metrics
                if current_time - last_sys_time >= 1:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().used / (1024 * 1024)
                    f_sys.write(f"{current_time},{mode},{len(cameras)},{cpu},{mem}\n")
                    last_sys_time = current_time

                # Record Browser metrics
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
                        pass # Ignore errors if page is not fully loaded yet

        await browser.close()
    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Streaming Benchmark Script")
    parser.add_argument('--mode', choices=['hls', 'webrtc', 'both'], required=True)
    parser.add_argument('--cameras', nargs='+', required=True)
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / 'benchmark' / 'results' / 'streaming' / 'hls_vs_webrtc'
    
    if args.mode == 'both':
        asyncio.run(run_benchmark('hls', args.cameras, args.duration, results_dir))
        asyncio.run(run_benchmark('webrtc', args.cameras, args.duration, results_dir))
    else:
        asyncio.run(run_benchmark(args.mode, args.cameras, args.duration, results_dir))
