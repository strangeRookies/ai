"""Fall/Faint incident recovery via temporary ROI re-detection (lying person).

Does NOT change global detector conf/imgsz. Recovery YOLO runs only on an expanded
ROI crop after a fall-suspected track is missing for N consecutive frames.

Tracker track_id and incident_id are separate: a new track_id may be linked to the
same incident_id when recovery relink succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence  # noqa: F401 — Any used by assign_recovery_track
from uuid import uuid4


class IncidentPhase(str, Enum):
    FALL_FAINT_SUSPECTED = "FALL_FAINT_SUSPECTED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    FALL_UNRECOVERED = "FALL_UNRECOVERED"
    RECOVERED = "RECOVERED"


@dataclass
class RecoveryConfig:
    miss_frames_to_start: int = 2
    expand_left_ratio: float = 0.50
    expand_right_ratio: float = 0.50
    expand_down_ratio: float = 0.60
    expand_up_ratio: float = 0.15
    recovery_conf: float = 0.05
    recovery_imgsz: int = 640
    recovery_interval_frames: int = 2
    max_recovery_seconds: float = 3.0
    max_center_distance_ratio: float = 1.50
    min_area_ratio: float = 0.25
    max_area_ratio: float = 4.0
    max_time_gap_seconds: float = 3.0


@dataclass
class IncidentRecord:
    incident_id: str
    camera_login_id: str
    source_track_id: int
    last_bbox: list[float]
    last_seen_ts: float
    last_seen_frame_id: int
    phase: IncidentPhase = IncidentPhase.FALL_FAINT_SUSPECTED
    miss_frames: int = 0
    recovery_attempts: int = 0
    recovery_started_ts: float | None = None
    last_recovery_frame_id: int | None = None
    active_track_id: int | None = None  # currently linked track (may change)
    last_roi: list[int] | None = None
    last_reject_reason: str | None = None
    # True after first successful recovery assigned a (possibly new) track_id.
    recovery_linked: bool = False


@dataclass
class RecoveryStats:
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_rejects: int = 0
    timeouts: int = 0
    wrong_relink: int | None = None
    total_recovery_latency_ms: float = 0.0
    recovery_latency_samples: int = 0

    def note_latency_ms(self, ms: float) -> None:
        self.total_recovery_latency_ms += float(ms)
        self.recovery_latency_samples += 1

    @property
    def mean_recovery_latency_ms(self) -> float | None:
        if self.recovery_latency_samples <= 0:
            return None
        return self.total_recovery_latency_ms / self.recovery_latency_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "recovery_rejects": self.recovery_rejects,
            "timeouts": self.timeouts,
            "wrong_relink": self.wrong_relink,
            "wrong_relink_evaluation_status": "not_evaluated",
            "mean_recovery_latency_ms": self.mean_recovery_latency_ms,
        }


DetectRoiFn = Callable[[Any, float, int], list[dict]]
# detect_roi_fn(crop_bgr, conf, imgsz) -> list of {bbox: [x1,y1,x2,y2] in crop space, confidence, keypoints?}


def bbox_area(bbox: Sequence[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def bbox_diag(bbox: Sequence[float]) -> float:
    w = max(1.0, float(bbox[2]) - float(bbox[0]))
    h = max(1.0, float(bbox[3]) - float(bbox[1]))
    return (w * w + h * h) ** 0.5


def expand_recovery_roi(
    last_bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    config: RecoveryConfig | None = None,
) -> list[int]:
    """Expand last_bbox left/right/down (and slight up), clamp to frame bounds.

    Returns [x1, y1, x2, y2] integer crop ROI in source space.
    """
    cfg = config or RecoveryConfig()
    x1, y1, x2, y2 = [float(v) for v in last_bbox[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    rx1 = x1 - w * cfg.expand_left_ratio
    rx2 = x2 + w * cfg.expand_right_ratio
    ry1 = y1 - h * cfg.expand_up_ratio
    ry2 = y2 + h * cfg.expand_down_ratio
    ix1 = int(max(0, min(frame_width - 1, round(rx1))))
    iy1 = int(max(0, min(frame_height - 1, round(ry1))))
    ix2 = int(max(ix1 + 1, min(frame_width, round(rx2))))
    iy2 = int(max(iy1 + 1, min(frame_height, round(ry2))))
    return [ix1, iy1, ix2, iy2]


def crop_bbox_to_source(bbox_crop: Sequence[float], roi_xyxy: Sequence[int]) -> list[float]:
    """Map Ultralytics crop-local xyxy to source-space by ROI origin offset only (no scale)."""
    ox, oy = float(roi_xyxy[0]), float(roi_xyxy[1])
    return [
        float(bbox_crop[0]) + ox,
        float(bbox_crop[1]) + oy,
        float(bbox_crop[2]) + ox,
        float(bbox_crop[3]) + oy,
    ]


def remap_keypoints_to_source(keypoints: list | None, roi_xyxy: Sequence[int]) -> list | None:
    if not keypoints:
        return keypoints
    ox, oy = float(roi_xyxy[0]), float(roi_xyxy[1])
    out = []
    for kp in keypoints:
        if not isinstance(kp, dict):
            out.append(kp)
            continue
        item = dict(kp)
        if "x" in item:
            item["x"] = round(float(item["x"]) + ox, 2)
        if "y" in item:
            item["y"] = round(float(item["y"]) + oy, 2)
        out.append(item)
    return out


def candidate_passes_geometry(
    last_bbox: Sequence[float],
    cand_bbox: Sequence[float],
    *,
    config: RecoveryConfig | None = None,
) -> tuple[bool, str]:
    cfg = config or RecoveryConfig()
    la = bbox_area(last_bbox)
    ca = bbox_area(cand_bbox)
    if la <= 0 or ca <= 0:
        return False, "invalid_area"
    ar = ca / la
    if ar < cfg.min_area_ratio or ar > cfg.max_area_ratio:
        return False, "area_ratio"
    lc = bbox_center(last_bbox)
    cc = bbox_center(cand_bbox)
    dist = ((lc[0] - cc[0]) ** 2 + (lc[1] - cc[1]) ** 2) ** 0.5
    if dist / bbox_diag(last_bbox) > cfg.max_center_distance_ratio:
        return False, "center_distance"
    return True, "ok"


class IncidentRecoveryManager:
    """Per-camera fall/faint incident recovery context."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        # camera_login_id -> incident_id -> record
        self._incidents: dict[str, dict[str, IncidentRecord]] = {}
        # camera_login_id -> track_id -> incident_id
        self._track_incident: dict[str, dict[int, str]] = {}
        self.stats = RecoveryStats()

    def reset_all(self) -> None:
        self._incidents.clear()
        self._track_incident.clear()

    def reset_camera(self, camera_login_id: str) -> None:
        cam = str(camera_login_id)
        self._incidents.pop(cam, None)
        self._track_incident.pop(cam, None)

    def open_incidents(self, camera_login_id: str) -> list[IncidentRecord]:
        return list((self._incidents.get(str(camera_login_id)) or {}).values())

    def incident_for_track(self, camera_login_id: str, track_id: int) -> str | None:
        return (self._track_incident.get(str(camera_login_id)) or {}).get(int(track_id))

    def get_incident(self, camera_login_id: str, incident_id: str) -> IncidentRecord | None:
        return (self._incidents.get(str(camera_login_id)) or {}).get(str(incident_id))

    def complete_relink(
        self,
        *,
        camera_login_id: str,
        incident_id: str,
        new_track_id: int,
        from_track_id: int | None = None,
    ) -> IncidentRecord | None:
        """Bind a (new) track_id to an existing incident after state migration."""
        cam = str(camera_login_id)
        rec = (self._incidents.get(cam) or {}).get(str(incident_id))
        if rec is None:
            return None
        tid = int(new_track_id)
        old = int(from_track_id) if from_track_id is not None else (
            rec.active_track_id if rec.active_track_id is not None else rec.source_track_id
        )
        tmap = self._track_incident.setdefault(cam, {})
        if old in tmap and tmap[old] == rec.incident_id:
            del tmap[old]
        tmap[tid] = rec.incident_id
        rec.active_track_id = tid
        rec.recovery_linked = True
        rec.phase = IncidentPhase.FALL_FAINT_SUSPECTED
        rec.miss_frames = 0
        rec.recovery_started_ts = None
        rec.last_reject_reason = None
        return rec

    def _reject_recovery_candidate(self, camera_login_id, record, item, reason):
        self.stats.recovery_rejects += 1
        if record is not None:
            record.last_reject_reason = reason
        rejected = dict(item)
        rejected["recovery_rejected"] = True
        rejected["recovery_reject_reason"] = reason
        return rejected

    def assign_recovery_track(
        self,
        *,
        camera_login_id: str,
        incident_id: str | None,
        detection: dict,
        tracker: Any = None,
        now: float | None = None,
        recovered_from_track_id: int | None = None,
    ) -> dict:
        """Choose track_id for a recovery detection without forcing the lost source id.

        - After the first successful recovery link, continue under active_track_id.
        - On first link, mint a NEW track id (via tracker when available) and rebind.
        """
        item = dict(detection)
        cam = str(camera_login_id)
        rec = None
        if incident_id:
            rec = (self._incidents.get(cam) or {}).get(str(incident_id))

        # Continuation: the tracker must still own the active id.
        if rec is not None and rec.recovery_linked and rec.active_track_id is not None:
            tid = int(rec.active_track_id)
            ensure_track = getattr(tracker, "ensure_track", None)
            if not callable(ensure_track):
                return self._reject_recovery_candidate(cam, rec, item, "tracker_registration_unavailable")
            item["track_id"] = tid
            item["recovery_track_continued"] = True
            try:
                refreshed = ensure_track(tid, item, now=now)
            except (RuntimeError, TypeError, ValueError):
                return self._reject_recovery_candidate(cam, rec, item, "tracker_refresh_failed")
            if not isinstance(refreshed, dict) or refreshed.get("track_id") is None:
                return self._reject_recovery_candidate(cam, rec, item, "tracker_refresh_failed")
            self._track_incident.setdefault(cam, {})[tid] = rec.incident_id
            return item

        from_id = recovered_from_track_id
        if from_id is None and rec is not None:
            from_id = rec.active_track_id if rec.active_track_id is not None else rec.source_track_id

        register = getattr(tracker, "register_recovery_detection", None)
        if not callable(register):
            return self._reject_recovery_candidate(cam, rec, item, "tracker_registration_unavailable")
        try:
            minted = register(item, now=now)
        except (RuntimeError, TypeError, ValueError):
            return self._reject_recovery_candidate(cam, rec, item, "tracker_registration_failed")
        if not isinstance(minted, dict) or minted.get("track_id") is None:
            return self._reject_recovery_candidate(cam, rec, item, "tracker_registration_failed")

        item = dict(minted)
        new_id = int(item["track_id"])
        if rec is not None:
            self.complete_relink(
                camera_login_id=cam,
                incident_id=rec.incident_id,
                new_track_id=new_id,
                from_track_id=int(from_id) if from_id is not None else None,
            )
        else:
            self._track_incident.setdefault(cam, {})[new_id] = str(incident_id or "")
        item["recovery_track_minted"] = True
        return item

    def note_fall_faint_suspected(
        self,
        *,
        camera_login_id: str,
        track_id: int,
        bbox: Sequence[float],
        timestamp: float,
        frame_id: int,
        incident_id: str | None = None,
    ) -> IncidentRecord:
        """Register or refresh a fall/faint-suspected incident for a track."""
        cam = str(camera_login_id)
        tid = int(track_id)
        if not bbox or len(bbox) < 4:
            raise ValueError("bbox required")
        self._incidents.setdefault(cam, {})
        self._track_incident.setdefault(cam, {})
        existing_id = self._track_incident[cam].get(tid)
        if existing_id and existing_id in self._incidents[cam]:
            rec = self._incidents[cam][existing_id]
            rec.last_bbox = [float(v) for v in bbox[:4]]
            rec.last_seen_ts = float(timestamp)
            rec.last_seen_frame_id = int(frame_id)
            rec.miss_frames = 0
            if rec.phase in {IncidentPhase.TEMPORARILY_LOST, IncidentPhase.FALL_UNRECOVERED}:
                # Seen again via normal path; keep incident_id, clear temporary loss
                rec.phase = IncidentPhase.FALL_FAINT_SUSPECTED
                rec.recovery_started_ts = None
            rec.active_track_id = tid
            return rec
        iid = incident_id or f"inc-{uuid4().hex[:12]}"
        rec = IncidentRecord(
            incident_id=iid,
            camera_login_id=cam,
            source_track_id=tid,
            last_bbox=[float(v) for v in bbox[:4]],
            last_seen_ts=float(timestamp),
            last_seen_frame_id=int(frame_id),
            phase=IncidentPhase.FALL_FAINT_SUSPECTED,
            active_track_id=tid,
        )
        self._incidents[cam][iid] = rec
        self._track_incident[cam][tid] = iid
        return rec

    def on_tracked_frame(
        self,
        *,
        camera_login_id: str,
        tracked: list[dict],
        timestamp: float,
        frame_id: int,
        frame_shape: tuple[int, int] | None,
        frame_bgr: Any = None,
        detect_roi_fn: DetectRoiFn | None = None,
    ) -> list[dict]:
        """Update miss counters; optionally run ROI recovery and annotate detections.

        Returns the (possibly augmented) tracked list with incident_id fields set.
        """
        cam = str(camera_login_id)
        if cam not in self._incidents:
            return tracked
        present_ids = {
            int(d["track_id"])
            for d in tracked
            if d.get("track_id") is not None
        }
        # Refresh last_bbox when active track still present
        for d in tracked:
            tid = d.get("track_id")
            if tid is None:
                continue
            iid = self._track_incident.get(cam, {}).get(int(tid))
            if not iid:
                continue
            rec = self._incidents[cam].get(iid)
            if rec is None:
                continue
            bb = d.get("bbox") or d.get("smoothed_bbox")
            if bb and len(bb) >= 4:
                rec.last_bbox = [float(v) for v in bb[:4]]
                rec.last_seen_ts = float(timestamp)
                rec.last_seen_frame_id = int(frame_id)
                rec.miss_frames = 0
                rec.phase = IncidentPhase.FALL_FAINT_SUSPECTED
                rec.recovery_started_ts = None
                rec.active_track_id = int(tid)
            d["incident_id"] = rec.incident_id

        # Miss update for incidents whose active track is gone
        for iid, rec in list(self._incidents[cam].items()):
            if rec.phase == IncidentPhase.FALL_UNRECOVERED:
                continue
            active = rec.active_track_id if rec.active_track_id is not None else rec.source_track_id
            if active in present_ids:
                continue
            rec.miss_frames += 1
            if rec.miss_frames < self.config.miss_frames_to_start:
                continue
            # Enter TEMPORARILY_LOST
            if rec.phase != IncidentPhase.TEMPORARILY_LOST:
                rec.phase = IncidentPhase.TEMPORARILY_LOST
                rec.recovery_started_ts = float(timestamp)
            # Timeout
            started = float(rec.recovery_started_ts if rec.recovery_started_ts is not None else timestamp)
            if float(timestamp) - started >= self.config.max_recovery_seconds:
                rec.phase = IncidentPhase.FALL_UNRECOVERED
                rec.last_reject_reason = "timeout"
                self.stats.timeouts += 1
                continue
            # Interval gate
            if rec.last_recovery_frame_id is not None:
                if int(frame_id) - int(rec.last_recovery_frame_id) < self.config.recovery_interval_frames:
                    continue
            if detect_roi_fn is None or frame_bgr is None or frame_shape is None:
                continue
            fh, fw = int(frame_shape[0]), int(frame_shape[1])
            if fh <= 0 or fw <= 0:
                continue
            import time as _time

            t0 = _time.perf_counter()
            result = self._try_roi_recovery(
                rec,
                frame_bgr=frame_bgr,
                frame_width=fw,
                frame_height=fh,
                tracked=tracked,
                present_ids=present_ids,
                detect_roi_fn=detect_roi_fn,
                timestamp=float(timestamp),
                frame_id=int(frame_id),
            )
            self.stats.note_latency_ms((_time.perf_counter() - t0) * 1000.0)
            rec.last_recovery_frame_id = int(frame_id)
            if result is not None:
                tracked = list(tracked) + [result]
        return tracked

    def _try_roi_recovery(
        self,
        rec: IncidentRecord,
        *,
        frame_bgr: Any,
        frame_width: int,
        frame_height: int,
        tracked: list[dict],
        present_ids: set[int],
        detect_roi_fn: DetectRoiFn,
        timestamp: float,
        frame_id: int,
    ) -> dict | None:
        self.stats.recovery_attempts += 1
        rec.recovery_attempts += 1
        roi = expand_recovery_roi(rec.last_bbox, frame_width, frame_height, self.config)
        rec.last_roi = list(roi)
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            rec.last_reject_reason = "empty_roi"
            self.stats.recovery_rejects += 1
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop is None or getattr(crop, "size", 0) == 0:
            rec.last_reject_reason = "empty_crop"
            self.stats.recovery_rejects += 1
            return None
        raw = detect_roi_fn(crop, self.config.recovery_conf, self.config.recovery_imgsz) or []
        # Map to source space
        candidates = []
        for item in raw:
            bb = item.get("bbox")
            if not bb or len(bb) < 4:
                continue
            src_bb = crop_bbox_to_source(bb, roi)
            kps = remap_keypoints_to_source(item.get("keypoints"), roi)
            candidates.append(
                {
                    "bbox": src_bb,
                    "confidence": float(item.get("confidence") or 0.0),
                    "keypoints": kps,
                    "keypoint_confidence": item.get("keypoint_confidence"),
                }
            )
        if len(candidates) != 1:
            rec.last_reject_reason = "multi_or_zero_candidates"
            self.stats.recovery_rejects += 1
            return None
        cand = candidates[0]
        # Exclude if overlaps an active track ownership (high IoU with existing track)
        if self._owned_by_active_track(cand["bbox"], tracked):
            rec.last_reject_reason = "owned_by_active_track"
            self.stats.recovery_rejects += 1
            return None
        ok, reason = candidate_passes_geometry(rec.last_bbox, cand["bbox"], config=self.config)
        if not ok:
            rec.last_reject_reason = reason
            self.stats.recovery_rejects += 1
            return None
        gap = float(timestamp) - float(rec.last_seen_ts)
        if gap > self.config.max_time_gap_seconds:
            rec.last_reject_reason = "time_gap"
            self.stats.recovery_rejects += 1
            return None
        # Success: do NOT force the lost source track_id onto the tracker.
        # Outer finalize mints/continues a recovery track and migrates runtime state.
        from_id = rec.active_track_id if rec.active_track_id is not None else rec.source_track_id
        out = {
            "bbox": cand["bbox"],
            "confidence": cand["confidence"],
            "keypoints": cand.get("keypoints"),
            "keypoint_confidence": cand.get("keypoint_confidence"),
            "incident_id": rec.incident_id,
            "recovery_relink": True,
            "recovered_from_track_id": int(from_id),
            # track_id intentionally omitted — assigned by assign_recovery_track / finalize
        }
        rec.phase = IncidentPhase.FALL_FAINT_SUSPECTED
        rec.miss_frames = 0
        rec.last_bbox = list(cand["bbox"])
        rec.last_seen_ts = float(timestamp)
        rec.last_seen_frame_id = int(frame_id)
        rec.recovery_started_ts = None
        rec.last_reject_reason = None
        self.stats.recovery_successes += 1
        return out

    def _owned_by_active_track(
        self,
        cand_bbox: Sequence[float],
        tracked: list[dict],
        iou_thresh: float = 0.30,
        center_ratio_thresh: float = 0.85,
    ) -> bool:
        """True if another active track already occupies this person-like region."""
        cc = bbox_center(cand_bbox)
        for d in tracked:
            if d.get("track_id") is None:
                continue
            bb = d.get("bbox") or d.get("smoothed_bbox")
            if not bb or len(bb) < 4:
                continue
            if _bbox_iou(cand_bbox, bb) >= iou_thresh:
                return True
            tc = bbox_center(bb)
            dist = ((cc[0] - tc[0]) ** 2 + (cc[1] - tc[1]) ** 2) ** 0.5
            if dist / max(bbox_diag(bb), 1.0) <= center_ratio_thresh:
                return True
        return False

    def diagnostics(self, camera_login_id: str | None = None) -> dict[str, Any]:
        if camera_login_id is None:
            open_count = sum(len(v) for v in self._incidents.values())
            phases: dict[str, int] = {}
            for cam_map in self._incidents.values():
                for rec in cam_map.values():
                    phases[rec.phase.value] = phases.get(rec.phase.value, 0) + 1
        else:
            cam_map = self._incidents.get(str(camera_login_id)) or {}
            open_count = len(cam_map)
            phases = {}
            for rec in cam_map.values():
                phases[rec.phase.value] = phases.get(rec.phase.value, 0) + 1
        out = self.stats.as_dict()
        out["open_incidents"] = open_count
        out["phases"] = phases
        return out


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def make_detect_roi_fn_from_yolo_pose(detector) -> DetectRoiFn:
    """Build ROI detection through the detector's backend-aware public API."""

    def _fn(crop, conf: float, imgsz: int) -> list[dict]:
        detect = getattr(detector, "detect", None)
        if not callable(detect):
            return []
        return list(detect(crop, conf=float(conf), imgsz=int(imgsz)))

    return _fn
