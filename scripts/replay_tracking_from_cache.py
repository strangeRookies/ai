#!/usr/bin/env python3
"""Deterministic tracker-only A/B replay from a fixed detection cache.

Does not re-run YOLO when cache exists. Imports production SimpleTrackAssigner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.tracker_timebase import resolve_tracker_fps  # noqa: E402
from tracking.simple_tracker import (  # noqa: E402
    SimpleTrackAssigner,
    bbox_iou,
    center_distance_ratio,
    predicted_bbox,
)


@dataclass
class TrackerConfig:
    name: str
    track_thresh: float = 0.10
    new_track_thresh: float = 0.25
    match_thresh: float = 0.20
    track_buffer: int = 90
    max_missing_seconds: float = 4.0
    center_match_ratio: float = 0.70
    soft_iou_scale: float = 0.45
    soft_center_scale: float = 1.25
    ultra_soft_enabled: bool = True
    ultra_soft_mode: str = "current"  # current | disabled | stricter
    use_predicted_bbox: bool = True
    use_max_prev_pred_iou: bool = False
    assumed_fps: float = 30.0
    bbox_smoothing_alpha: float = 0.60
    # Near-dup mint suppression (off by default for production parity configs A–J).
    near_dup_suppress_mode: str = "none"
    near_dup_iou_thresh: float = 0.70
    near_dup_center_ratio: float = 0.25
    near_dup_area_ratio_min: float = 0.55
    near_dup_area_ratio_max: float = 1.80
    near_dup_keypoint_dist: float = 0.35
    sort_detections_by_conf: bool = False


PRESETS: dict[str, tuple[str, ...]] = {
    "new-track-thresholds": ("A_current", "B_new_thresh_020", "C_new_thresh_030"),
}


CONFIGS: dict[str, TrackerConfig] = {
    "A_current": TrackerConfig(name="A_current"),
    "B_new_thresh_020": TrackerConfig(name="B_new_thresh_020", new_track_thresh=0.20),
    "C_new_thresh_030": TrackerConfig(name="C_new_thresh_030", new_track_thresh=0.30),
    "D_ultra_soft_off": TrackerConfig(name="D_ultra_soft_off", ultra_soft_enabled=False, ultra_soft_mode="disabled"),
    "E_ultra_soft_strict": TrackerConfig(name="E_ultra_soft_strict", ultra_soft_mode="stricter", soft_iou_scale=0.35, soft_center_scale=1.10),
    "F_match_thresh_015": TrackerConfig(name="F_match_thresh_015", match_thresh=0.15),
    "G_prev_bbox_only": TrackerConfig(name="G_prev_bbox_only", use_predicted_bbox=False),
    "H_max_prev_pred_iou": TrackerConfig(name="H_max_prev_pred_iou", use_max_prev_pred_iou=True),
    # Combinations chosen after single-factor A~H (not pure single-variable).
    "I_new030_prev_bbox": TrackerConfig(
        name="I_new030_prev_bbox",
        new_track_thresh=0.30,
        use_predicted_bbox=False,
    ),
    "J_new030_match015_prev": TrackerConfig(
        name="J_new030_match015_prev",
        new_track_thresh=0.30,
        match_thresh=0.15,
        use_predicted_bbox=False,
    ),
    # Near-duplicate mint suppression on top of I (MULTI_DET_EXTRA residual).
    "K_I_claimed_iou_suppress": TrackerConfig(
        name="K_I_claimed_iou_suppress",
        new_track_thresh=0.30,
        use_predicted_bbox=False,
        near_dup_suppress_mode="claimed_iou",
        near_dup_iou_thresh=0.70,
        sort_detections_by_conf=True,
    ),
    "L_I_hybrid_suppress": TrackerConfig(
        name="L_I_hybrid_suppress",
        new_track_thresh=0.30,
        use_predicted_bbox=False,
        near_dup_suppress_mode="hybrid",
        near_dup_iou_thresh=0.70,
        near_dup_center_ratio=0.25,
        near_dup_area_ratio_min=0.55,
        near_dup_area_ratio_max=1.80,
        sort_detections_by_conf=True,
    ),
    "M_I_hybrid_kp_safe": TrackerConfig(
        name="M_I_hybrid_kp_safe",
        new_track_thresh=0.30,
        use_predicted_bbox=False,
        near_dup_suppress_mode="hybrid_kp",
        near_dup_iou_thresh=0.70,
        near_dup_center_ratio=0.25,
        near_dup_area_ratio_min=0.55,
        near_dup_area_ratio_max=1.80,
        near_dup_keypoint_dist=0.35,
        sort_detections_by_conf=True,
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_cache(cache_path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    frames: list[dict] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_type") == "meta":
                meta = obj
            else:
                frames.append(obj)
    frames.sort(key=lambda x: int(x["frame_id"]))
    return meta, frames


def validate_cache(meta: dict, frames: list[dict]) -> dict:
    """Validate ordered finite detection rows before tracker replay."""
    previous = None
    for row in frames:
        frame_id = int(row["frame_id"])
        if previous is not None and frame_id <= previous:
            raise ValueError("cache frame_id must be strictly increasing")
        previous = frame_id
        for detection in row.get("detections") or []:
            bbox = detection.get("bbox_xyxy") or []
            if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
                raise ValueError("cache bbox must contain four finite values")
    if meta.get("frames_written") is not None and int(meta["frames_written"]) != len(frames):
        raise ValueError("cache frames_written does not match rows")
    return {"frames": len(frames), "last_frame_id": previous}

def build_cache_from_video(
    video_path: Path,
    cache_path: Path,
    *,
    model_name: str,
    imgsz: int,
    conf: float,
    frame_start: int,
    frame_end: int,
    device: str = "cpu",
) -> dict:
    import cv2
    from detector.yolo_pose_detector import YoloPoseDetector

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    det = YoloPoseDetector(model_name, device=device, imgsz=imgsz, conf=conf)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_end < 0:
        frame_end = total - 1
    frame_end = min(frame_end, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    meta = {
        "_type": "meta",
        "video_path": str(video_path),
        "video_sha256": sha256_file(video_path),
        "model": model_name,
        "imgsz": imgsz,
        "detector_conf": conf,
        "source_fps": src_fps,
        "source_width": width,
        "source_height": height,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with cache_path.open("w", encoding="utf-8") as out:
        out.write(json.dumps(meta, ensure_ascii=False) + "\n")
        written = 0
        for frame_id in range(frame_start, frame_end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            # Match production publish path: analyze at 1280x720 letterbox-pad style content.
            # Cache detections in published coordinate space for tracker fidelity.
            if width != 1280 or height != 720:
                frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame
            raw = det.detect(frame_resized)
            dets = []
            for i, d in enumerate(raw):
                kps = d.get("keypoints") or []
                dets.append(
                    {
                        "detection_index": i,
                        "bbox_xyxy": [float(x) for x in (d.get("bbox") or [])],
                        "confidence": float(d.get("confidence") or 0.0),
                        "class_id": 0,
                        "keypoints": kps,
                        "avg_keypoint_conf": float(d.get("keypoint_confidence") or 0.0),
                        "valid_keypoints": sum(1 for k in kps if float(k.get("confidence") or 0.0) >= 0.25),
                    }
                )
            row = {
                "frame_id": frame_id,
                "timestamp_ms": int(round(1000.0 * (frame_id - frame_start) / max(src_fps, 1e-6))),
                "source_width": 1280,
                "source_height": 720,
                "detections": dets,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"[cache] wrote {written} frames (frame_id={frame_id})", flush=True)
    cap.release()
    meta["frames_written"] = written
    print(f"[cache] done frames={written} path={cache_path}", flush=True)
    return meta


class ConfigurableTracker(SimpleTrackAssigner):
    """SimpleTrackAssigner with experimental association policy knobs for A/B only."""

    def __init__(self, cfg: TrackerConfig):
        super().__init__(
            track_thresh=cfg.track_thresh,
            new_track_thresh=cfg.new_track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            max_missing_seconds=cfg.max_missing_seconds,
            center_match_ratio=cfg.center_match_ratio,
            soft_iou_scale=cfg.soft_iou_scale,
            soft_center_scale=cfg.soft_center_scale,
            bbox_smoothing_alpha=cfg.bbox_smoothing_alpha,
            assumed_fps=cfg.assumed_fps,
            min_box_area=10.0,
            near_dup_suppress_mode=cfg.near_dup_suppress_mode,
            near_dup_iou_thresh=cfg.near_dup_iou_thresh,
            near_dup_center_ratio=cfg.near_dup_center_ratio,
            near_dup_area_ratio_min=cfg.near_dup_area_ratio_min,
            near_dup_area_ratio_max=cfg.near_dup_area_ratio_max,
            near_dup_keypoint_dist=cfg.near_dup_keypoint_dist,
            sort_detections_by_conf=cfg.sort_detections_by_conf,
        )
        self.cfg = cfg
        if cfg.ultra_soft_mode == "disabled" or not cfg.ultra_soft_enabled:
            self._ultra_soft_recover = lambda *a, **k: None  # type: ignore[method-assign]
        elif cfg.ultra_soft_mode == "stricter":
            self._ultra_soft_recover = self._ultra_soft_recover_strict  # type: ignore[method-assign]

    def _ultra_soft_recover_strict(self, match_meta, assigned_track_ids):
        rejected = list((match_meta or {}).get("rejectedCandidates") or [])
        if not rejected:
            return None
        unmatched = [tid for tid in self._tracks if tid not in assigned_track_ids]
        nearby = [
            item
            for item in rejected
            if item.get("centerRatio") is not None and float(item["centerRatio"]) <= 1.4
        ]
        if len(nearby) >= 2 and len(unmatched) >= 2:
            return None
        best = rejected[0]
        track_id = best.get("trackId")
        if track_id is None or int(track_id) in assigned_track_ids:
            return None
        center = best.get("centerRatio")
        iou = float(best.get("iou") or 0.0)
        if center is None:
            return None
        center = float(center)
        # Stricter than production ultra-soft.
        if not (center <= 1.05 or (iou >= 0.12 and center <= 1.25) or (len(unmatched) == 1 and center <= 1.35)):
            return None
        meta = {
            "mode": "ultra_soft_strict",
            "iou": round(iou, 4),
            "centerRatio": round(center, 4),
            "score": round(iou + max(0.0, 1.2 - center), 4),
            "rejectedCandidates": rejected[:5],
        }
        return int(track_id), meta

    def _match_existing_track_detailed(self, bbox, assigned_track_ids, now=None):
        # Optional: previous-only or max(prev, pred) IoU for experimental configs.
        if self.cfg.use_predicted_bbox and not self.cfg.use_max_prev_pred_iou:
            return super()._match_existing_track_detailed(bbox, assigned_track_ids, now=now)

        best_track_id = None
        best_score = -1.0
        best_meta = None
        rejected = []
        soft_iou = self.iou_threshold * self.soft_iou_scale
        soft_center = self.center_match_ratio * self.soft_center_scale
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            prev = track.get("smoothed_bbox") or track.get("bbox")
            pred = predicted_bbox(track, now=now) if self.cfg.use_predicted_bbox else prev
            if self.cfg.use_max_prev_pred_iou:
                iou_score = max(bbox_iou(bbox, prev), bbox_iou(bbox, pred))
                # center against previous for stability
                center_ratio = center_distance_ratio(bbox, prev)
            else:
                # previous only
                iou_score = bbox_iou(bbox, prev)
                center_ratio = center_distance_ratio(bbox, prev)
            hard_ok = iou_score >= self.iou_threshold or center_ratio <= self.center_match_ratio
            soft_ok = iou_score >= soft_iou and center_ratio <= soft_center
            sole_track = len(self._tracks) == 1 and center_ratio <= soft_center
            if not (hard_ok or soft_ok or sole_track):
                rejected.append(
                    {
                        "trackId": int(track_id),
                        "iou": round(iou_score, 4),
                        "centerRatio": round(center_ratio, 4) if center_ratio != float("inf") else None,
                        "reject": "below_iou_and_center",
                    }
                )
                continue
            mode = "hard" if hard_ok else ("soft" if soft_ok else "sole")
            score = iou_score + max(0.0, self.center_match_ratio - center_ratio)
            if soft_ok and not hard_ok:
                score += 0.15
            if sole_track and not hard_ok:
                score += 0.05
            if score > best_score:
                best_score = score
                best_track_id = track_id
                best_meta = {
                    "mode": mode,
                    "iou": round(iou_score, 4),
                    "centerRatio": round(center_ratio, 4) if center_ratio != float("inf") else None,
                    "score": round(score, 4),
                    "rejectedCandidates": rejected[-5:],
                }
        if best_track_id is None and rejected:
            rejected_sorted = sorted(
                rejected,
                key=lambda item: (
                    item.get("centerRatio") if item.get("centerRatio") is not None else 1e9,
                    -(item.get("iou") or 0.0),
                ),
            )
            best_meta = {"mode": None, "rejectedCandidates": rejected_sorted[:8]}
        return best_track_id, best_meta


def classify_iou_failure(event: dict, gap_seconds: float, prev_bbox, det_bbox, pred_bbox) -> str:
    """Classify a residual association failure from tracker evidence."""
    confidence = float(event.get("confidence") or 0.0)
    rejected = list((event.get("bestRejected") or {}).get("rejectedCandidates") or [])
    if confidence < 0.20:
        return "LOW_CONFIDENCE_DETECTION"
    if len(rejected) >= 2:
        return "MULTI_TRACK_COMPETITION"
    if gap_seconds >= 0.4:
        return "FRAME_GAP"
    if det_bbox and (float(det_bbox[0]) < 8 or float(det_bbox[1]) < 8 or float(det_bbox[2]) > 1272 or float(det_bbox[3]) > 712):
        return "SCREEN_BOUNDARY"
    if prev_bbox and det_bbox:
        prev_ratio = max(float(prev_bbox[2]) - float(prev_bbox[0]), 1e-3) / max(float(prev_bbox[3]) - float(prev_bbox[1]), 1e-3)
        det_ratio = max(float(det_bbox[2]) - float(det_bbox[0]), 1e-3) / max(float(det_bbox[3]) - float(det_bbox[1]), 1e-3)
        if abs(det_ratio / prev_ratio - 1.0) >= 0.5:
            return "BBOX_ASPECT_RATIO_CHANGE"
    if pred_bbox and prev_bbox and center_distance_ratio(pred_bbox, prev_bbox) > 1.5:
        return "PREDICTED_BBOX_DRIFT"
    if rejected and rejected[0].get("iou") is not None and abs(float(rejected[0]["iou"]) - 0.20) <= 0.03:
        return "THRESHOLD_EDGE"
    if rejected and rejected[0].get("centerRatio") is not None and float(rejected[0]["centerRatio"]) > 1.8:
        return "VELOCITY_OVERSHOOT"
    return "UNKNOWN"


def run_tracker_config(
    frames: list[dict],
    cfg: TrackerConfig,
    out_dir: Path,
    source_fps: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_source_fps = source_fps
    effective_fps = resolve_tracker_fps(raw_source_fps, cfg.assumed_fps)
    try:
        valid_source_fps = float(raw_source_fps)
    except (TypeError, ValueError):
        valid_source_fps = 0.0
    tracker_fps_source = "cache_metadata" if math.isfinite(valid_source_fps) and valid_source_fps > 0.0 else "configured_fallback"
    source_fps = effective_fps if tracker_fps_source != "cache_metadata" else valid_source_fps
    cfg = replace(cfg, assumed_fps=effective_fps)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    tracker = ConfigurableTracker(cfg)
    frame_path = out_dir / "frame_results.jsonl"
    switch_path = out_dir / "switch_events.jsonl"

    total_new = 0
    total_lost = 0
    iou_below = 0
    multi_det_extra = 0
    near_dup_suppress = 0
    switch_events = 0
    switch_reason_counts: dict[str, int] = {}
    match_mode_counts = {"hard": 0, "soft": 0, "ultra_soft": 0, "sole": 0}
    previous_input_frame_id = None
    frame_gap_values: list[int] = []
    ghost_events = 0
    max_ghost_duration = 0.0
    duplicate_frames = 0
    unique_duplicate_pairs: set[tuple[int, int]] = set()
    person_present_frames = 0
    dominant_track_frames = 0
    track_id_hist: dict[int, int] = {}
    first_person_frame = None
    first_track_frame = None
    occlusion_gaps = 0
    occlusion_recovered = 0
    last_active_ids: set[int] = set()
    last_person_present = False
    gap_start_frame = None
    id_before_gap = None

    dt = 1.0 / max(source_fps, 1e-6)
    t0 = time.time()

    with frame_path.open("w", encoding="utf-8") as f_out, switch_path.open("w", encoding="utf-8") as s_out:
        for row in frames:
            frame_id = int(row["frame_id"])
            if previous_input_frame_id is not None:
                frame_gap_values.append(max(0, frame_id - previous_input_frame_id - 1))
            previous_input_frame_id = frame_id
            ts_ms = int(row.get("timestamp_ms") or 0)
            now = float(ts_ms) / 1000.0
            raw_dets = row.get("detections") or []
            dets = []
            for d in raw_dets:
                dets.append(
                    {
                        "bbox": list(d.get("bbox_xyxy") or []),
                        "confidence": float(d.get("confidence") or 0.0),
                        "keypoints": d.get("keypoints") or [],
                        "keypoint_confidence": float(d.get("avg_keypoint_conf") or 0.0),
                    }
                )
            person_present = len(dets) > 0
            if person_present:
                person_present_frames += 1
                if first_person_frame is None:
                    first_person_frame = frame_id

            before_ids = set(tracker._tracks.keys())
            tracked = tracker.update(dets, now=now)
            diag = tracker.diagnostics()
            events = list(diag.get("lifecycle_events") or [])
            new_n = int(diag.get("new_tracks") or 0)
            lost_n = int(diag.get("lost_tracks") or 0)
            total_new += new_n
            total_lost += lost_n

            active_ids = {int(t["track_id"]) for t in tracked if t.get("track_id") is not None}
            for tid in active_ids:
                track_id_hist[tid] = track_id_hist.get(tid, 0) + 1
            if active_ids and first_track_frame is None and person_present:
                first_track_frame = frame_id

            # dominant track ownership for single-person proxy
            if person_present and tracked:
                # highest-conf detection's track
                best = max(tracked, key=lambda x: float(x.get("confidence") or 0.0))
                tid = best.get("track_id")
                if tid is not None:
                    dominant_track_frames += 1  # provisional; corrected later with global dominant

            # duplicates among active tracked boxes
            tracked_with_id = [
                (int(t["track_id"]), t.get("bbox"))
                for t in tracked
                if t.get("track_id") is not None and t.get("bbox")
            ]
            frame_has_dup = False
            for i in range(len(tracked_with_id)):
                for j in range(i + 1, len(tracked_with_id)):
                    tid_a, box_a = tracked_with_id[i]
                    tid_b, box_b = tracked_with_id[j]
                    if bbox_iou(box_a, box_b) >= 0.7:
                        frame_has_dup = True
                        pair = (min(tid_a, tid_b), max(tid_a, tid_b))
                        unique_duplicate_pairs.add(pair)
            if frame_has_dup:
                duplicate_frames += 1

            # ghost: tracks with missing_frames high while still in tracker
            for tid, tr in (diag.get("tracks") or {}).items():
                miss = int(tr.get("missing_frames") or 0)
                if miss > 0:
                    ghost_sec = miss / max(source_fps, 1e-6)
                    if ghost_sec > 1.0:
                        ghost_events += 1
                        max_ghost_duration = max(max_ghost_duration, ghost_sec)

            # occlusion recovery proxy
            if last_person_present and not person_present:
                gap_start_frame = frame_id
                id_before_gap = next(iter(last_active_ids), None)
            if (not last_person_present) and person_present and gap_start_frame is not None:
                gap_frames = frame_id - gap_start_frame
                gap_sec = gap_frames / max(source_fps, 1e-6)
                if 0.5 <= gap_sec <= 1.0:
                    occlusion_gaps += 1
                    if id_before_gap is not None and id_before_gap in active_ids:
                        occlusion_recovered += 1
                gap_start_frame = None
                id_before_gap = None

            for ev in events:
                if ev.get("event") == "match":
                    mode = str(ev.get("reason") or "").lower().replace("-", "_")
                    if mode in match_mode_counts:
                        match_mode_counts[mode] += 1
                if ev.get("event") == "filter" and ev.get("reason") == "near_duplicate_suppress":
                    near_dup_suppress += 1
                    continue
                if ev.get("event") != "new_track":
                    continue
                switch_events += 1
                reason = ev.get("switchReason") or ev.get("reason") or "UNKNOWN"
                switch_reason_counts[reason] = switch_reason_counts.get(reason, 0) + 1
                if reason == "IOU_BELOW_THRESHOLD":
                    iou_below += 1
                if reason == "MULTI_DET_EXTRA":
                    multi_det_extra += 1
                rejected = list((ev.get("bestRejected") or {}).get("rejectedCandidates") or [])
                prev_tid = ev.get("previousTrackId")
                snap = ev.get("previousTrackSnapshot") or {}
                prev_box = snap.get("bbox")
                pred_box = snap.get("predicted_bbox")
                # Fall back to rejected candidate geometry if snapshot missing.
                if prev_box is None and rejected:
                    prev_box = rejected[0].get("bbox")
                    pred_box = rejected[0].get("predicted_bbox")
                    if prev_tid is None:
                        prev_tid = rejected[0].get("trackId")
                det_box = ev.get("bbox")
                miss = int(snap.get("missing_frames") or (rejected[0].get("missing_frames") if rejected else 0) or 0)
                gap_seconds = miss / max(source_fps, 1e-6)
                # Prefer IoU values already measured against previous/predicted.
                iou_prev = None
                iou_pred = None
                if rejected:
                    iou_prev = rejected[0].get("iou")
                    iou_pred = rejected[0].get("iou_pred")
                if iou_prev is None and prev_box and det_box:
                    iou_prev = bbox_iou(prev_box, det_box)
                if iou_pred is None and pred_box and det_box:
                    iou_pred = bbox_iou(pred_box, det_box)
                center_norm = None
                center_px = None
                if prev_box and det_box and len(prev_box) >= 4 and len(det_box) >= 4:
                    center_norm = center_distance_ratio(prev_box, det_box)
                    pc = ((float(prev_box[0]) + float(prev_box[2])) / 2.0, (float(prev_box[1]) + float(prev_box[3])) / 2.0)
                    dc = ((float(det_box[0]) + float(det_box[2])) / 2.0, (float(det_box[1]) + float(det_box[3])) / 2.0)
                    center_px = round(((pc[0] - dc[0]) ** 2 + (pc[1] - dc[1]) ** 2) ** 0.5, 2)
                if reason == "IOU_BELOW_THRESHOLD":
                    failure = classify_iou_failure(ev, gap_seconds, prev_box, det_box, pred_box)
                elif reason == "MULTI_DET_EXTRA":
                    failure = "COMPETING_TRACK_CLAIM"
                elif reason == "NEW_SCENE":
                    failure = "NEW_SCENE"
                else:
                    failure = reason
                width_ratio = height_ratio = area_ratio = None
                if prev_box and det_box and len(prev_box) >= 4 and len(det_box) >= 4:
                    pw = max(1e-3, float(prev_box[2]) - float(prev_box[0]))
                    ph = max(1e-3, float(prev_box[3]) - float(prev_box[1]))
                    dw = max(1e-3, float(det_box[2]) - float(det_box[0]))
                    dh = max(1e-3, float(det_box[3]) - float(det_box[1]))
                    width_ratio = round(dw / pw, 4)
                    height_ratio = round(dh / ph, 4)
                    area_ratio = round((dw * dh) / (pw * ph), 4)
                rec = {
                    "frame_id": frame_id,
                    "timestamp_ms": ts_ms,
                    "previous_track_id": prev_tid,
                    "previous_bbox": prev_box,
                    "predicted_bbox": pred_box,
                    "current_detection_bbox": det_box,
                    "iou_previous_detection": iou_prev,
                    "iou_predicted_detection": iou_pred,
                    "center_distance_px": center_px,
                    "normalized_center_distance": None if center_norm is None else round(center_norm, 4),
                    "bbox_width_ratio": width_ratio,
                    "bbox_height_ratio": height_ratio,
                    "bbox_area_ratio": area_ratio,
                    "gap_frames": int(round(gap_seconds * source_fps)),
                    "gap_seconds": round(gap_seconds, 4),
                    "detection_confidence": ev.get("confidence"),
                    "candidate_track_count": len(rejected),
                    "competing_track_ids": [r.get("trackId") for r in rejected],
                    "claimed_track_diagnostics": ev.get("claimedTrackDiagnostics") or [],
                    "unmatched_active_tracks": ev.get("unmatchedActiveTracks") or [],
                    "already_assigned_tracks": ev.get("alreadyAssignedTracks") or [],
                    "new_track_id": ev.get("trackId"),
                    "switch_reason": reason,
                    "failure_class": failure,
                }
                s_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            f_out.write(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "timestamp_ms": ts_ms,
                        "raw_detections": len(raw_dets),
                        "tracked": [
                            {
                                "track_id": t.get("track_id"),
                                "bbox": t.get("bbox"),
                                "confidence": t.get("confidence"),
                            }
                            for t in tracked
                        ],
                        "new_tracks": new_n,
                        "lost_tracks": lost_n,
                        "active_ids": sorted(active_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            last_active_ids = active_ids
            last_person_present = person_present

    # dominant track for single-person ID retention
    dominant_id = None
    if track_id_hist:
        dominant_id = max(track_id_hist.items(), key=lambda kv: kv[1])[0]
    # recompute retention properly
    dominant_track_frames = track_id_hist.get(dominant_id, 0) if dominant_id is not None else 0
    # retention over person-present frames is approximate without GT; use coverage of dominant track
    # better: second pass not available; use hist / person_present
    id_retention = (dominant_track_frames / person_present_frames) if person_present_frames else 0.0

    n_frames = len(frames)
    duration_sec = n_frames / max(source_fps, 1e-6)
    duration_min = duration_sec / 60.0
    expected_initial = 1 if person_present_frames > 0 else 0
    unexpected_new = max(0, total_new - expected_initial)
    start_delay_ms = None
    if first_person_frame is not None and first_track_frame is not None:
        start_delay_ms = max(0.0, (first_track_frame - first_person_frame) * 1000.0 / source_fps)

    eligible_tracks = sum(1 for frames_seen in track_id_hist.values() if frames_seen >= 30)
    completed_sequences = sum(frames_seen // 30 for frames_seen in track_id_hist.values())
    incomplete_reason_counts = {"insufficient_track_frames": sum(1 for frames_seen in track_id_hist.values() if frames_seen < 30)}
    total_match_events = sum(match_mode_counts.values())
    summary = {
        "config_name": cfg.name,
        "frames": n_frames,
        "duration_sec": round(duration_sec, 3),
        "source_fps": source_fps,
        "tracker_assumed_fps": cfg.assumed_fps,
        "tracker_fps_source": tracker_fps_source,
        "total_new_tracks": total_new,
        "expected_initial_tracks": expected_initial,
        "unexpected_new_tracks": unexpected_new,
        "unexpected_new_tracks_per_min": round(unexpected_new / max(duration_min, 1e-9), 4),
        "lost_tracks": total_lost,
        "lost_tracks_per_min": round(total_lost / max(duration_min, 1e-9), 4),
        "iou_below_threshold": iou_below,
        "multi_det_extra": multi_det_extra,
        "near_dup_suppress_count": near_dup_suppress,
        "switch_events": switch_events,
        "total_id_switch_events": switch_events,
        "switch_reason_counts": switch_reason_counts,
        "switch_reason_new_scene": switch_reason_counts.get("NEW_SCENE", 0),
        "switch_reason_iou_below_threshold": switch_reason_counts.get("IOU_BELOW_THRESHOLD", 0),
        "switch_reason_multi_det_extra": switch_reason_counts.get("MULTI_DET_EXTRA", 0),
        "switch_reason_no_candidate": switch_reason_counts.get("NO_CANDIDATE", 0),
        "fragmentation_count": unexpected_new,
        "fragmentation_rate": round(unexpected_new / max(person_present_frames, 1), 4),
        "hard_match_count": match_mode_counts["hard"],
        "soft_match_count": match_mode_counts["soft"],
        "ultra_soft_match_count": match_mode_counts["ultra_soft"],
        "sole_match_count": match_mode_counts["sole"],
        "hard_match_rate": round(match_mode_counts["hard"] / max(total_match_events, 1), 4),
        "soft_match_rate": round(match_mode_counts["soft"] / max(total_match_events, 1), 4),
        "ultra_soft_match_rate": round(match_mode_counts["ultra_soft"] / max(total_match_events, 1), 4),
        "sole_match_rate": round(match_mode_counts["sole"] / max(total_match_events, 1), 4),
        "eligible_tracks": eligible_tracks,
        "completed_tracks": eligible_tracks,
        "total_completed_sequences": completed_sequences,
        "sequence_completion_rate": round(eligible_tracks / max(len(track_id_hist), 1), 4),
        "incomplete_reason_counts": incomplete_reason_counts,
        "incomplete_insufficient_track_frames": incomplete_reason_counts["insufficient_track_frames"],
        "maximum_frame_gap": max(frame_gap_values, default=0),
        "average_frame_gap": round(sum(frame_gap_values) / max(len(frame_gap_values), 1), 4),
        "identity_consistency_violation_count": iou_below + multi_det_extra,
        "id_retention_rate": round(id_retention, 4),
        "dominant_track_id": dominant_id,
        "person_present_frames": person_present_frames,
        "track_start_delay_ms": None if start_delay_ms is None else round(start_delay_ms, 1),
        "occlusion_gaps": occlusion_gaps,
        "occlusion_recovered": occlusion_recovered,
        "occlusion_recovery_rate": round(occlusion_recovered / occlusion_gaps, 4) if occlusion_gaps else None,
        "ghost_count": ghost_events,
        "max_ghost_duration_sec": round(max_ghost_duration, 3),
        "duplicate_frames": duplicate_frames,
        "unique_duplicate_pairs": len(unique_duplicate_pairs),
        "processing_fps": round(n_frames / max(time.time() - t0, 1e-6), 2),
        "new_track_thresh": cfg.new_track_thresh,
        "match_thresh": cfg.match_thresh,
        "ultra_soft_enabled": cfg.ultra_soft_enabled and cfg.ultra_soft_mode != "disabled",
        "ultra_soft_mode": cfg.ultra_soft_mode,
        "use_predicted_bbox": cfg.use_predicted_bbox,
        "use_max_prev_pred_iou": cfg.use_max_prev_pred_iou,
        "near_dup_suppress_mode": cfg.near_dup_suppress_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[replay] {cfg.name}: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def write_comparison(exp_dir: Path, summaries: list[dict], baseline_name: str = "A_current") -> None:
    csv_path = exp_dir / "comparison_summary.csv"
    md_path = exp_dir / "comparison_summary.md"
    fields = [
        "config_name",
        "new_track_thresh",
        "total_id_switch_events",
        "switch_reason_new_scene",
        "switch_reason_iou_below_threshold",
        "switch_reason_multi_det_extra",
        "switch_reason_no_candidate",
        "fragmentation_count",
        "fragmentation_rate",
        "hard_match_count",
        "hard_match_rate",
        "soft_match_count",
        "soft_match_rate",
        "ultra_soft_match_count",
        "ultra_soft_match_rate",
        "sole_match_count",
        "sole_match_rate",
        "eligible_tracks",
        "completed_tracks",
        "total_completed_sequences",
        "sequence_completion_rate",
        "incomplete_insufficient_track_frames",
        "maximum_frame_gap",
        "average_frame_gap",
        "identity_consistency_violation_count",
        "match_thresh",
        "ultra_soft_enabled",
        "near_dup_suppress_mode",
        "unexpected_new_tracks",
        "unexpected_new_tracks_per_min",
        "lost_tracks",
        "lost_tracks_per_min",
        "iou_below_threshold",
        "multi_det_extra",
        "near_dup_suppress_count",
        "iou_switch_reduction_pct",
        "id_retention_rate",
        "track_start_delay_ms",
        "occlusion_recovery_rate",
        "ghost_count",
        "max_ghost_duration_sec",
        "duplicate_frames",
        "unique_duplicate_pairs",
        "processing_fps",
        "selected",
        "selection_reason",
    ]
    base = next((s for s in summaries if s["config_name"] == baseline_name), summaries[0])
    base_iou = max(1, int(base.get("iou_below_threshold") or 0))
    base_dup = max(1, int(base.get("duplicate_frames") or 0))

    # selection
    selected = None
    for s in summaries:
        s["iou_switch_reduction_pct"] = round(100.0 * (1.0 - (s.get("iou_below_threshold") or 0) / base_iou), 2)
        s["duplicate_reduction_pct"] = round(100.0 * (1.0 - (s.get("duplicate_frames") or 0) / base_dup), 2)
        s["selected"] = False
        s["selection_reason"] = ""
    # Prefer low multi_det_extra / unexpected new, high retention, low dup/ghost
    ranked = sorted(
        summaries,
        key=lambda s: (
            s.get("multi_det_extra") if s.get("multi_det_extra") is not None else 999,
            s.get("unexpected_new_tracks_per_min") or 999,
            s.get("lost_tracks_per_min") or 999,
            -(s.get("id_retention_rate") or 0),
            s.get("duplicate_frames") or 0,
            s.get("ghost_count") or 0,
        ),
    )
    if ranked:
        selected = ranked[0]
        selected["selected"] = True
        selected["selection_reason"] = (
            "lowest MULTI_DET_EXTRA/unexpected new with higher retention and fewer ghost/duplicate; "
            "near-dup suppress preferred over match_thresh loosening"
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    lines = [
        "# Tracker A/B Comparison",
        "",
        f"Baseline config: `{baseline_name}`",
        "",
        "| config | unexpected new/min | lost/min | MULTI_DET_EXTRA | dup frames | unique dup pairs | retention | start delay | ghost | suppress | FPS | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['config_name']} | {s.get('unexpected_new_tracks_per_min')} | {s.get('lost_tracks_per_min')} | "
            f"{s.get('multi_det_extra')} | {s.get('duplicate_frames')} | {s.get('unique_duplicate_pairs')} | "
            f"{s.get('id_retention_rate')} | {s.get('track_start_delay_ms')} | {s.get('ghost_count')} | "
            f"{s.get('near_dup_suppress_count')} | {s.get('processing_fps')} | {s.get('selected')} |"
        )
    if selected:
        lines.extend(
            [
                "",
                "## Selected",
                f"- config: `{selected['config_name']}`",
                f"- reason: {selected['selection_reason']}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] wrote {csv_path} and {md_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build detection cache and/or replay tracker A/B configs")
    p.add_argument("--mode", choices=["build-cache", "replay", "both"], default="both")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--experiment-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="yolo26n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=3599)
    p.add_argument("--configs", type=str, default=",".join(CONFIGS.keys()))
    p.add_argument("--preset", choices=sorted(PRESETS), default="", help="Named configuration subset; --configs remains available.")
    p.add_argument("--force-rerun", action="store_true", help="Ignore existing summary.json and re-run listed configs")
    p.add_argument(
        "--reuse-summary-from",
        type=str,
        default="",
        help="Optional experiment dir to reuse summary.json for configs not re-run (e.g. A/I baselines)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"build-cache", "both"}:
        if not args.video:
            raise SystemExit("--video required for build-cache")
        if cache_path.exists():
            print(f"[cache] exists, skip build: {cache_path}", flush=True)
        else:
            build_cache_from_video(
                Path(args.video),
                cache_path,
                model_name=args.model,
                imgsz=args.imgsz,
                conf=args.conf,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                device=args.device,
            )

    if args.mode in {"replay", "both"}:
        meta, frames = load_cache(cache_path)
        if not frames:
            raise SystemExit(f"empty cache: {cache_path}")
        source_fps = meta.get("source_fps")
        names = list(PRESETS[args.preset]) if args.preset else [n.strip() for n in args.configs.split(",") if n.strip()]
        reuse_root = Path(args.reuse_summary_from) if args.reuse_summary_from else None
        summaries = []
        for name in names:
            if name not in CONFIGS:
                print(f"[warn] unknown config {name}, skip", flush=True)
                continue
            cfg = CONFIGS[name]
            out = exp_dir / name
            if (not args.force_rerun) and (out / "summary.json").exists():
                print(f"[replay] reuse existing summary for {name}", flush=True)
                summaries.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))
                continue
            if (
                (not args.force_rerun)
                and reuse_root is not None
                and (reuse_root / name / "summary.json").exists()
                and name in {"A_current", "I_new030_prev_bbox"}
            ):
                # Reuse prior A/I only; still allow K/L/M to compute fresh metrics.
                print(f"[replay] reuse baseline summary {name} from {reuse_root}", flush=True)
                summaries.append(json.loads((reuse_root / name / "summary.json").read_text(encoding="utf-8")))
                continue
            summaries.append(
                run_tracker_config(
                    frames,
                    cfg,
                    out,
                    source_fps=source_fps,
                )
            )
        write_comparison(exp_dir, summaries)
        (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
