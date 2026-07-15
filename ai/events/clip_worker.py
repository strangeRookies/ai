import math
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

# 얼굴 블러 대상 keypoint (COCO 17포인트: 0=코, 1=왼눈, 2=오른눈, 3=왼귀, 4=오른귀)
_HEAD_KEYPOINT_INDICES = (0, 1, 2, 3, 4)
_SHOULDER_KEYPOINT_INDICES = (5, 6)
_MIN_KEYPOINT_CONF = 0.3
# 우선순위 3(직전 프레임 얼굴 위치 재사용) 최대 허용 연속 프레임 수 (~0.17s @30fps).
# 이보다 오래 재사용하면 사람이 실제로 움직였을 때 옛 위치를 계속 우려먹게 되므로 제한.
_MAX_STALE_FACE_BOX_FRAMES = 5


def _kp_point(keypoints, index, conf_min=_MIN_KEYPOINT_CONF):
    if not keypoints or index >= len(keypoints):
        return None
    item = keypoints[index]
    conf = float(item.get("confidence", item.get("conf", 0.0)) or 0.0)
    if conf < conf_min:
        return None
    return float(item["x"]), float(item["y"])


def _track_ids_match(a, b):
    if a is None or b is None:
        return False
    try:
        return int(float(str(a))) == int(float(str(b)))
    except (TypeError, ValueError):
        return False


def _find_track_box(boxes, track_id):
    """이 프레임의 boxes 리스트에서 target track_id에 해당하는 항목을 찾는다."""
    if not boxes or track_id is None:
        return None
    for box in boxes:
        if _track_ids_match(box.get("track_id"), track_id):
            return box
    return None


def _face_box_from_keypoints(box, width, height):
    """1순위: 코/눈/귀 keypoint 기반으로 얼굴 영역만 추정."""
    keypoints = box.get("keypoints") if box else None
    if not keypoints:
        return None
    points = []
    for idx in _HEAD_KEYPOINT_INDICES:
        point = _kp_point(keypoints, idx)
        if point is not None:
            points.append(point)
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # keypoint는 점이라 얼굴 전체를 못 덮으므로 padding 필요.
    # 어깨너비를 척도로 써서 padding을 산출하고(사람 크기에 비례), 어깨가 안 잡히면
    # bbox 크기 기준으로 대체.
    shoulder_l = _kp_point(keypoints, _SHOULDER_KEYPOINT_INDICES[0])
    shoulder_r = _kp_point(keypoints, _SHOULDER_KEYPOINT_INDICES[1])
    if shoulder_l is not None and shoulder_r is not None:
        scale = math.hypot(shoulder_r[0] - shoulder_l[0], shoulder_r[1] - shoulder_l[1])
    else:
        bx1, by1, bx2, by2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
        if None not in (bx1, by1, bx2, by2):
            scale = min(abs(float(bx2) - float(bx1)), abs(float(by2) - float(by1))) * 0.4
        else:
            scale = max(max_x - min_x, max_y - min_y, 1.0) * 2.0
    pad = max(scale * 0.35, 8.0)

    x1_px = int(max(0, min_x - pad))
    y1_px = int(max(0, min_y - pad))
    x2_px = int(min(width, max_x + pad))
    y2_px = int(min(height, max_y + pad))
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    return (x1_px, y1_px, x2_px, y2_px)


def _upper_body_fallback_box(box, width, height, ratio=0.32):
    """2순위: keypoint 신뢰도가 낮을 때 전신 bbox 상단 일부를 넓게 블러."""
    if not box:
        return None
    x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
    if None in (x1, y1, x2, y2):
        return None
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    band_height = (y2 - y1) * ratio

    x1_px = int(max(0, min(x1, width)))
    x2_px = int(max(0, min(x2, width)))
    y1_px = int(max(0, min(y1, height)))
    y2_px = int(max(0, min(y1 + band_height, height)))
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    return (x1_px, y1_px, x2_px, y2_px)


def _blur_region(cv2_module, frame, region):
    x1_px, y1_px, x2_px, y2_px = region
    roi = frame[y1_px:y2_px, x1_px:x2_px]
    ksize = max(5, int(min(x2_px - x1_px, y2_px - y1_px) * 0.3))
    if ksize % 2 == 0:
        ksize += 1
    frame[y1_px:y2_px, x1_px:x2_px] = cv2_module.GaussianBlur(roi, (ksize, ksize), 0)


def _apply_per_frame_face_blur(cv2_module, frames, frame_boxes, track_id, width, height):
    """프레임마다 얼굴 위치를 다시 계산해서 블러 적용 (전신 고정 블러 대체).

    우선순위: 1) keypoint 기반 얼굴 박스  2) 전신 bbox 상단 일부(상위 %)
    3) 직전 성공 위치 재사용(최대 _MAX_STALE_FACE_BOX_FRAMES 프레임)  4) 포기(블러 없음)
    """
    tier_counts = {"keypoint": 0, "upper_body": 0, "stale_reuse": 0, "skipped": 0}
    stale_box = None
    stale_streak = 0

    for idx, frame in enumerate(frames):
        boxes_for_frame = frame_boxes[idx] if idx < len(frame_boxes) else None
        matched_box = _find_track_box(boxes_for_frame, track_id)

        region = None
        tier = None
        if matched_box is not None:
            region = _face_box_from_keypoints(matched_box, width, height)
            if region is not None:
                tier = "keypoint"
            else:
                region = _upper_body_fallback_box(matched_box, width, height)
                if region is not None:
                    tier = "upper_body"

        if region is None:
            if stale_box is not None and stale_streak < _MAX_STALE_FACE_BOX_FRAMES:
                region = stale_box
                tier = "stale_reuse"
                stale_streak += 1
            else:
                # 재사용 한도 초과(또는 재사용할 위치 자체가 없음): 오래된 위치를 계속
                # 우려먹지 않기 위해 포기. 안 보이는 것보다 낫다고 오판하지 않도록
                # 여기서 전신 블러로 확대하지 않음(이 프레임은 그대로 둠).
                stale_box = None
                stale_streak = 0
                tier_counts["skipped"] += 1
                continue
        else:
            stale_box = region
            stale_streak = 0

        tier_counts[tier] += 1
        _blur_region(cv2_module, frame, region)

    return tier_counts


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

    # frame_boxes를 frames와 같은 인덱스로 유지해야 프레임별 블러 위치가 안 어긋남
    # (아래 이미지가 아닌 프레임 필터링과 동일한 필터를 같이 적용)
    raw_frame_boxes = list(getattr(task, "frame_boxes", None) or [])
    frames = []
    frame_boxes = []
    for idx, frame in enumerate(task.frames):
        if not hasattr(frame, "shape"):
            continue
        frames.append(frame)
        frame_boxes.append(raw_frame_boxes[idx] if idx < len(raw_frame_boxes) else None)
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

    # 1. 비디오 라이팅 전에 프레임 크기를 첫 프레임 기준으로 통일
    # (ffmpeg/cv2 둘 다 고정 해상도 필요, 블러 좌표 계산도 이 크기 기준)
    for idx in range(len(frames)):
        if frames[idx].shape[:2] != (height, width):
            frames[idx] = cv2.resize(frames[idx], (width, height))

    # 2. 인코딩 전에 프레임마다 얼굴 위치를 다시 계산해서 블러 적용(비식별화).
    # 트리거 시점 1회 고정 bbox가 아니라, 저장된 그 프레임 자체의 keypoint로
    # 매번 새로 계산하므로 대상이 움직여도 얼굴을 계속 따라가며 가림.
    track_id = task.metadata.get("track_id") if task.metadata else None
    blur_tier_counts = _apply_per_frame_face_blur(cv2, frames, frame_boxes, track_id, width, height)
    print(
        f"[clip-worker] face-blur tiers event_type={task.event_type} camera_id={task.camera_id} "
        f"frames={len(frames)} counts={blur_tier_counts}",
        file=sys.stderr,
    )

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
