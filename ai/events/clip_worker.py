import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from ai.events.event_clip import EventClipTask
from ai.storage.uploader import upload_clip

# 브라우저 재생 호환을 위해 우선 시도할 ffmpeg 인코더 순서 (GPU면 nvenc가 더 빠름)
_FFMPEG_ENCODER_CANDIDATES = ("h264_nvenc", "libx264")


def _ffmpeg_bin():
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def _encode_with_ffmpeg(frames, fps, width, height, output_path):
    ffmpeg_bin = _ffmpeg_bin()
    if shutil.which(ffmpeg_bin) is None:
        print(f"[clip-worker] ffmpeg binary not found ({ffmpeg_bin}); skipping ffmpeg encode", file=sys.stderr)
        return None

    raw_input = b"".join(frame.tobytes() for frame in frames)

    for encoder in _FFMPEG_ENCODER_CANDIDATES:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(float(fps or 30.0)),
            "-i", "-",
            "-an",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # communicate()는 stdin에 쓰는 동안 stdout/stderr도 동시에 비워줘서
            # ffmpeg 로그로 파이프가 꽉 차 서로 블로킹되는 데드락을 피함
            _, stderr = proc.communicate(input=raw_input)
        except Exception as exc:
            print(f"[clip-worker] ffmpeg encoder={encoder} failed to run: {exc}", file=sys.stderr)
            continue

        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        tail = (stderr or b"").decode("utf-8", errors="replace").strip().splitlines()[-5:]
        print(
            f"[clip-worker] ffmpeg encoder={encoder} failed (rc={proc.returncode}): {' | '.join(tail)}",
            file=sys.stderr,
        )

    return None


def enqueue_event_clip(task_queue, task):
    try:
        task_queue.put_nowait(task)
        return True
    except queue.Full:
        try:
            dropped = task_queue.get_nowait()
            print(
                f"[clip-worker] clip queue full; dropped oldest task: "
                f"event_type={dropped.event_type} camera_id={dropped.camera_id}",
                file=sys.stderr,
            )
            task_queue.put_nowait(task)
            return True
        except queue.Empty:
            return False
        except queue.Full:
            print("[clip-worker] clip queue full; new task dropped", file=sys.stderr)
            return False

#save_clip_to_mp4(task) 함수 내부에 VideoWriter 구현
def save_clip_to_mp4(task):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"OpenCV is required to save event clips: {exc}") from exc

    frames = [frame for frame in task.frames if hasattr(frame, "shape")]
    if not frames:
        raise ValueError("event clip has no image frames to encode")

    first = frames[0]
    height, width = first.shape[:2]
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromtimestamp(task.created_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence_suffix = _evidence_suffix(task.metadata)
    filename = f"{_safe(task.event_type)}_{_safe(task.camera_id)}_{timestamp}{evidence_suffix}.mp4"
    output_path = output_dir / filename

    # 1. 인코딩 전에 모든 프레임에 대해 BBox 블러 처리(비식별화) 선제 적용
    bbox = task.metadata.get("bbox") if task.metadata else None
    if bbox and len(bbox) == 4:
        for idx in range(len(frames)):
            frame = frames[idx]
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
                frames[idx] = frame
            
            x1, y1, x2, y2 = bbox
            # 0~1 사이의 정규화된 좌표인 경우 픽셀 크기 적용
            if all(0.0 <= val <= 1.0 for val in (x1, y1, x2, y2)) and max(x1, y1, x2, y2) <= 1.0:
                x1_px = int(x1 * width)
                y1_px = int(y1 * height)
                x2_px = int(x2 * width)
                y2_px = int(y2 * height)
            else:
                x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)
            
            # 좌표 유효성 검사 및 정렬
            x1_px, x2_px = max(0, min(x1_px, x2_px)), min(width, max(x1_px, x2_px))
            y1_px, y2_px = max(0, min(y1_px, y2_px)), min(height, max(y1_px, y2_px))
            
            if x2_px > x1_px and y2_px > y1_px:
                roi = frame[y1_px:y2_px, x1_px:x2_px]
                ksize = max(5, int(min(x2_px - x1_px, y2_px - y1_px) * 0.3))
                if ksize % 2 == 0:
                    ksize += 1
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                frame[y1_px:y2_px, x1_px:x2_px] = blurred_roi

    # 2. 비디오 라이팅 수행
    # 프레임 크기를 첫 프레임 기준으로 통일 (ffmpeg/cv2 둘 다 고정 해상도 필요)
    for idx in range(len(frames)):
        if frames[idx].shape[:2] != (height, width):
            frames[idx] = cv2.resize(frames[idx], (width, height))

    # 브라우저 재생 가능한 H.264로 인코딩 시도 (ffmpeg 서브프로세스, opencv-python엔 라이선스상 H.264 인코더가 없는 경우가 흔함)
    if _encode_with_ffmpeg(frames, task.fps, width, height, output_path):
        return output_path

    # ffmpeg 사용 불가 시 최종 폴백: 예전 cv2.VideoWriter 방식 (mp4v로 떨어지면 브라우저 재생은 안 될 수 있음)
    print(f"[clip-worker] ffmpeg encode failed for {output_path}; falling back to cv2.VideoWriter (may not be browser-playable)", file=sys.stderr)
    writer = None
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_path), fourcc, float(task.fps or 30.0), (width, height))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"failed to open VideoWriter for {output_path}")

    try:
        for frame in frames:
            writer.write(frame) # <-- 실제로 로컬 파일로 인코딩하여 기록하는 부분
    finally:
        writer.release() # 작업이 끝나면 해제(저장 완료)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"encoded clip is empty: {output_path}")
    return output_path

#ClipWriterWorker는 태스크를 큐에서 꺼내어 관리하는 작업 스레드의 역할만 담당하며, 실제 비디오 파일로 인코딩하여 기록하는 연산은 save_clip_to_mp4 함수가 처리하는 구조
class ClipWriterWorker:
    def __init__(self, task_queue, uploader=upload_clip, publisher=None, mqtt_event_topic=None):
        self.task_queue = task_queue
        self.uploader = uploader
        self.publisher = publisher
        self.mqtt_event_topic = mqtt_event_topic
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="clip-writer-worker", daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout=2):
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop_event.is_set() or not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                output_path = save_clip_to_mp4(task)
                try:
                    upload_result = self.uploader(output_path, task.metadata)
                    print(f"[clip-worker] clip ready: path={output_path}", file=sys.stderr)

                    # 업로드 성공 후 publisher가 있으면 최종 MQTT 이벤트 발행
                    if upload_result and upload_result.get("uploaded") and upload_result.get("url") and self.publisher:
                        s3_url = upload_result["url"]
                        meta = task.metadata or {}

                        # 백엔드 DTO(SafetyEventDto) 규격에 맞게 페이로드 작성
                        event_payload = {
                            "type": task.event_type,
                            "camera_id": task.camera_id,
                            "camera_login_id": task.camera_id,
                            "timestamp": meta.get("event_timestamp") or datetime.now(timezone.utc).isoformat(),
                            "severity": "HIGH",
                            "message": "Fall-like safety event detected. (Video Uploaded)",
                            "source": "edge-ai",
                            "eventId": meta.get("evidenceId") or meta.get("event_timestamp"),
                            "track_id": str(meta.get("track_id", "")),
                            "clip_url": s3_url,
                            "clip_path": str(output_path)
                        }

                        topic = self.mqtt_event_topic or "safety/events"
                        if hasattr(self.publisher, "publish_event"):
                            self.publisher.publish_event(event_payload)
                        else:
                            self.publisher.publish(event_payload, topic=topic)
                        print(f"[clip-worker] published final event with clip_url: eventId={event_payload['eventId']} url={s3_url} to topic={topic}", file=sys.stderr)
                except Exception as exc:
                    print(f"[clip-worker] upload failed; local file retained: {exc}", file=sys.stderr)
            except Exception as exc:
                print(
                    f"[clip-worker] clip task failed: event_type={task.event_type} "
                    f"camera_id={task.camera_id} error={exc}",
                    file=sys.stderr,
                )
            finally:
                self.task_queue.task_done()


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _evidence_suffix(metadata):
    if not isinstance(metadata, dict):
        return ""
    evidence_id = metadata.get("evidenceId")
    if evidence_id is None:
        return ""
    return f"_evidence-{_safe(evidence_id)}"
