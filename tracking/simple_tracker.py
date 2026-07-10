import time


class SimpleTrackAssigner:
    """Small IoU tracker used until a production tracker such as ByteTrack is wired in."""

    def __init__(
        self,
        iou_threshold=0.3,
        max_missing_seconds=6.0,
        track_thresh=0.1,
        match_thresh=None,
        track_buffer=90,
        min_box_area=10.0,
        bbox_smoothing_alpha=0.6,
        center_match_ratio=0.85,
        soft_iou_scale=0.45,
        soft_center_scale=1.25,
    ):
        self.iou_threshold = float(iou_threshold if match_thresh is None else match_thresh)
        self.max_missing_seconds = float(max_missing_seconds)
        self.track_thresh = float(track_thresh)
        self.track_buffer = max(0, int(track_buffer))
        self.min_box_area = max(0.0, float(min_box_area))
        self.bbox_smoothing_alpha = min(max(float(bbox_smoothing_alpha), 0.0), 1.0)
        self.center_match_ratio = max(0.0, float(center_match_ratio))
        # Soft match: keep ID across fall-like bbox shape change (tall→wide) when center stays near.
        self.soft_iou_scale = max(0.0, float(soft_iou_scale))
        self.soft_center_scale = max(1.0, float(soft_center_scale))
        self._next_track_id = 1
        self._tracks = {}
        self._frame_index = 0
        self.last_diagnostics = self._empty_diagnostics()
        self.last_events: list[dict] = []

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        self._frame_index += 1
        events: list[dict] = []
        self._mark_tracks_missing()
        stale_ids = self._drop_stale_tracks(now, events)

        assigned_track_ids = set()
        output = []
        new_tracks = 0
        id_switch_like_events = 0
        kept, filtered = self._filter_detections_with_reasons(detections, events)
        for detection in kept:
            detection = dict(detection)
            original_track_id = detection.get("track_id")
            track_id = detection.get("track_id")
            match_meta = None
            if track_id is None:
                track_id, match_meta = self._match_existing_track_detailed(detection.get("bbox"), assigned_track_ids)
                if track_id is None:
                    track_id = self._allocate_track_id()
                    new_tracks += 1
                    events.append(
                        {
                            "event": "new_track",
                            "reason": "no_match",
                            "trackId": int(track_id),
                            "bbox": detection.get("bbox"),
                            "confidence": float(detection.get("confidence", 0.0)),
                            "activeTracksBefore": len(self._tracks),
                            "bestRejected": match_meta,
                        }
                    )
                else:
                    events.append(
                        {
                            "event": "match",
                            "reason": match_meta.get("mode") if match_meta else "match",
                            "trackId": int(track_id),
                            "iou": match_meta.get("iou") if match_meta else None,
                            "centerRatio": match_meta.get("centerRatio") if match_meta else None,
                            "bbox": detection.get("bbox"),
                        }
                    )
            elif int(track_id) in self._tracks:
                track_id = int(track_id)
                events.append({"event": "match", "reason": "detector_track_id", "trackId": track_id})
            else:
                track_id = int(track_id)
                new_tracks += 1
                events.append(
                    {
                        "event": "new_track",
                        "reason": "unknown_detector_track_id",
                        "trackId": track_id,
                    }
                )
            track_id = int(track_id)
            if original_track_id is not None and int(original_track_id) != track_id:
                id_switch_like_events += 1
                events.append(
                    {
                        "event": "id_switch_like",
                        "fromTrackId": int(original_track_id),
                        "toTrackId": track_id,
                    }
                )
            detection["track_id"] = track_id
            assigned_track_ids.add(track_id)
            self._update_track(track_id, detection, now)
            self._copy_track_fields(detection, self._tracks[track_id])
            output.append(detection)
        buffer_lost = self._drop_buffer_expired_tracks(events)
        lost_tracks = stale_ids + buffer_lost
        self.last_events = events
        self.last_diagnostics = self._build_diagnostics(
            new_tracks,
            lost_tracks,
            id_switch_like_events,
            events,
            raw_detection_count=len(detections or []),
            filtered_count=filtered,
        )
        return output

    def _allocate_track_id(self):
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _match_existing_track(self, bbox, assigned_track_ids):
        track_id, _meta = self._match_existing_track_detailed(bbox, assigned_track_ids)
        return track_id

    def _match_existing_track_detailed(self, bbox, assigned_track_ids):
        best_track_id = None
        best_score = -1.0
        best_meta = None
        rejected = []
        soft_iou = self.iou_threshold * self.soft_iou_scale
        soft_center = self.center_match_ratio * self.soft_center_scale
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            pred = predicted_bbox(track)
            iou_score = bbox_iou(bbox, pred)
            center_ratio = center_distance_ratio(bbox, pred)
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
            best_meta = {"mode": None, "rejectedCandidates": rejected[-8:]}
        return best_track_id, best_meta

    def _filter_detections(self, detections):
        kept, _ = self._filter_detections_with_reasons(detections, events=None)
        return kept

    def _filter_detections_with_reasons(self, detections, events):
        output = []
        filtered = 0
        for detection in detections or []:
            conf = float(detection.get("confidence", 1.0))
            area = bbox_area(detection.get("bbox"))
            if conf < self.track_thresh:
                filtered += 1
                if events is not None:
                    events.append(
                        {
                            "event": "filter",
                            "reason": "low_confidence",
                            "confidence": conf,
                            "trackThresh": self.track_thresh,
                            "bbox": detection.get("bbox"),
                        }
                    )
                continue
            if area < self.min_box_area:
                filtered += 1
                if events is not None:
                    events.append(
                        {
                            "event": "filter",
                            "reason": "tiny_box",
                            "area": round(area, 2),
                            "minBoxArea": self.min_box_area,
                            "bbox": detection.get("bbox"),
                        }
                    )
                continue
            output.append(detection)
        return output, filtered

    def _mark_tracks_missing(self):
        for track in self._tracks.values():
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1

    def _update_track(self, track_id, detection, now):
        raw_bbox = detection.get("bbox")
        previous = self._tracks.get(track_id, {})
        smoothed_bbox = smooth_bbox(previous.get("smoothed_bbox") or previous.get("bbox"), raw_bbox, self.bbox_smoothing_alpha)
        velocity = bbox_velocity(previous.get("raw_bbox") or previous.get("bbox"), raw_bbox)
        age = int(previous.get("age", 0)) + 1
        self._tracks[track_id] = {
            "bbox": raw_bbox,
            "raw_bbox": raw_bbox,
            "smoothed_bbox": smoothed_bbox,
            "velocity": velocity,
            "last_seen_at": now,
            "age": age,
            "missing_frames": 0,
            "confidence": float(detection.get("confidence", 0.0)),
        }

    def _copy_track_fields(self, detection, track):
        detection["raw_bbox"] = track.get("raw_bbox")
        detection["smoothed_bbox"] = track.get("smoothed_bbox")
        detection["bbox"] = track.get("smoothed_bbox") or detection.get("bbox")
        detection["track_age"] = int(track.get("age", 0))
        detection["missing_frames"] = int(track.get("missing_frames", 0))
        detection["track_confidence"] = float(track.get("confidence", detection.get("confidence", 0.0)))

    def _drop_stale_tracks(self, now, events=None):
        stale_track_ids = []
        for track_id, track in list(self._tracks.items()):
            gap = now - track.get("last_seen_at", 0.0)
            if gap > self.max_missing_seconds:
                stale_track_ids.append(track_id)
                if events is not None:
                    events.append(
                        {
                            "event": "lost",
                            "reason": "max_missing_seconds",
                            "trackId": int(track_id),
                            "missingSeconds": round(gap, 3),
                            "maxMissingSeconds": self.max_missing_seconds,
                            "missingFrames": int(track.get("missing_frames", 0)),
                            "lastBbox": track.get("bbox"),
                        }
                    )
                del self._tracks[track_id]
        return len(stale_track_ids)

    def _drop_buffer_expired_tracks(self, events=None):
        stale_track_ids = []
        for track_id, track in list(self._tracks.items()):
            missing = int(track.get("missing_frames", 0))
            if missing > self.track_buffer:
                stale_track_ids.append(track_id)
                if events is not None:
                    events.append(
                        {
                            "event": "lost",
                            "reason": "track_buffer_exceeded",
                            "trackId": int(track_id),
                            "missingFrames": missing,
                            "trackBuffer": self.track_buffer,
                            "lastBbox": track.get("bbox"),
                        }
                    )
                del self._tracks[track_id]
        return len(stale_track_ids)

    def _build_diagnostics(self, new_tracks, lost_tracks, id_switch_like_events, events=None, raw_detection_count=0, filtered_count=0):
        tracks = {
            str(track_id): {
                "track_age": int(track.get("age", 0)),
                "missing_frames": int(track.get("missing_frames", 0)),
                "detection_conf": float(track.get("confidence", 0.0)),
                "bbox": track.get("bbox"),
                "smoothed_bbox": track.get("smoothed_bbox"),
                "predicted_bbox": predicted_bbox(track),
            }
            for track_id, track in sorted(self._tracks.items())
        }
        removed = [
            int(e["trackId"])
            for e in (events or [])
            if e.get("event") == "lost" and e.get("trackId") is not None
        ]
        return {
            "active_tracks": len(self._tracks),
            "new_tracks": int(new_tracks),
            "lost_tracks": int(lost_tracks),
            "id_switch_like_events": int(id_switch_like_events),
            "removed_track_ids": removed,
            "raw_detection_count": int(raw_detection_count),
            "filtered_detection_count": int(filtered_count),
            "lifecycle_events": list(events or []),
            "tracks": tracks,
        }

    def diagnostics(self):
        return dict(self.last_diagnostics)

    def _empty_diagnostics(self):
        return {
            "active_tracks": 0,
            "new_tracks": 0,
            "lost_tracks": 0,
            "id_switch_like_events": 0,
            "removed_track_ids": [],
            "raw_detection_count": 0,
            "filtered_detection_count": 0,
            "lifecycle_events": [],
            "tracks": {},
        }


def bbox_iou(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]

    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    inter_width = max(ix2 - ix1, 0.0)
    inter_height = max(iy2 - iy1, 0.0)
    intersection = inter_width * inter_height

    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def smooth_bbox(previous_bbox, current_bbox, alpha):
    if not previous_bbox or not current_bbox:
        return current_bbox
    alpha = min(max(float(alpha), 0.0), 1.0)
    return [
        round(alpha * float(current_bbox[idx]) + (1.0 - alpha) * float(previous_bbox[idx]), 3)
        for idx in range(4)
    ]


def bbox_velocity(previous_bbox, current_bbox):
    if not previous_bbox or not current_bbox:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(float(current_bbox[idx]) - float(previous_bbox[idx]), 3)
        for idx in range(4)
    ]


def predicted_bbox(track):
    bbox = track.get("smoothed_bbox") or track.get("bbox")
    if not bbox:
        return bbox
    missing_frames = max(0, int(track.get("missing_frames", 0)))
    velocity = track.get("velocity") or [0.0, 0.0, 0.0, 0.0]
    return [
        round(float(bbox[idx]) + float(velocity[idx]) * missing_frames, 3)
        for idx in range(4)
    ]


def center_distance_ratio(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return float("inf")
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]
    left_cx = (lx1 + lx2) / 2.0
    left_cy = (ly1 + ly2) / 2.0
    right_cx = (rx1 + rx2) / 2.0
    right_cy = (ry1 + ry2) / 2.0
    distance = ((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2) ** 0.5
    left_diag = max(((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5, 1.0)
    right_diag = max(((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2) ** 0.5, 1.0)
    return distance / max(left_diag, right_diag)
