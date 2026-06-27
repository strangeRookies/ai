import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_video(input_path: Path, output_path: Path) -> bool:
    """
    ffmpeg를 이용해 640 해상도로 비디오를 변환합니다.
    -vf "scale=640:-2": 가로 640 기준 비율 유지
    -an: 오디오 제거
    -c:v libx264: H.264 인코딩
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 이미 파일이 존재하고 크기가 0이 아니면 스킵
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {output_path.name}")
        return True

    print(f"[CONVERT] Starting: {input_path.name} -> {output_path}")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "scale=640:-2",
        "-an",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        str(output_path)
    ]
    
    try:
        # 백그라운드 프로세스로 실행 및 출력 억제
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
        print(f"[SUCCESS] Converted: {input_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert {input_path.name}. Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Exception during convert {input_path.name}: {e}")
        return False

def copy_annotation(input_path: Path, output_path: Path) -> bool:
    """
    어노테이션 XML 파일을 복사합니다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy annotation {input_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate 640px proxy videos and copy annotations.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw source directory")
    parser.add_argument("--output-dir", default="data/raw_640", help="Path to raw 640 output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel ffmpeg workers")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    out_root = Path(args.output_dir)

    if not raw_root.exists():
        print(f"[ERROR] Raw source directory does not exist: {raw_root}")
        return

    # 1. 비디오 파일 수집 (.mp4, .avi)
    extensions = [".mp4", ".avi", ".MP4", ".AVI"]
    video_files = []
    for ext in extensions:
        video_files.extend(raw_root.rglob(f"*{ext}"))
    video_files = sorted(set(video_files))

    print(f"[*] Found {len(video_files)} video files in {raw_root}")

    # 2. 어노테이션 파일 수집 (.xml)
    xml_files = sorted(set(raw_root.rglob("*.xml")))
    print(f"[*] Found {len(xml_files)} annotation files in {raw_root}")

    # 3. 어노테이션 파일 복사 (단순 I/O 이므로 빠르게 먼저 처리)
    copied_annotations = 0
    for xml_file in xml_files:
        rel_path = xml_file.relative_to(raw_root)
        dest_path = out_root / rel_path
        if copy_annotation(xml_file, dest_path):
            copied_annotations += 1
    print(f"[*] Copied {copied_annotations}/{len(xml_files)} annotation files to {out_root}")

    # 4. 비디오 파일 병렬 변환
    conversion_tasks = []
    for video_file in video_files:
        rel_path = video_file.relative_to(raw_root)
        dest_path = out_root / rel_path
        conversion_tasks.append((video_file, dest_path))

    success_count = 0
    print(f"[*] Starting parallel video conversion with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_video, inp, out): (inp, out)
            for inp, out in conversion_tasks
        }
        
        for future in as_completed(futures):
            inp, out = futures[future]
            if future.result():
                success_count += 1

    print(f"[FINISHED] Processed {len(video_files)} videos. Successful conversions/skips: {success_count}/{len(video_files)}")

if __name__ == "__main__":
    main()
