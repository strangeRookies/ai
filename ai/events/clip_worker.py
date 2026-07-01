import os
import queue
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from ai.events.event_clip import EventClipTask
from ai.storage.uploader import upload_clip


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

    writer = None
     # 1. 사용할 코덱(avc1, H264 등)을 탐색하며 VideoWriter 인스턴스를 생성
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_path), fourcc, float(task.fps or 30.0), (width, height))#여기서 인스턴스 생성
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"failed to open VideoWriter for {output_path}")
    # 2. 버퍼링된 프레임들을 하나씩 VideoWriter에 작성(write)합니다
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height)) 
            writer.write(frame) # <-- 실제로 로컬 파일로 인코딩하여 기록하는 부분
    finally:
        writer.release() # 작업이 끝나면 해제(저장 완료)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"encoded clip is empty: {output_path}")
    return output_path

#ClipWriterWorker는 태스크를 큐에서 꺼내어 관리하는 작업 스레드의 역할만 담당하며, 실제 비디오 파일로 인코딩하여 기록하는 연산은 save_clip_to_mp4 함수가 처리하는 구조
class ClipWriterWorker:
    def __init__(self, task_queue, uploader=upload_clip):
        self.task_queue = task_queue
        self.uploader = uploader
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
                    self.uploader(output_path, task.metadata)
                    print(f"[clip-worker] clip ready: path={output_path}", file=sys.stderr)
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
