import time


class SimpleTrackAssigner:
    """Small IoU tracker used until a production tracker such as ByteTrack is wired in."""

    def __init__(
        self,
        iou_threshold=0.3,
        max_missing_seconds=2.0,
        track_thresh=0.1,
        match_thresh=None,
        track_buffer=30,
        min_box_area=10.0,
        bbox_smoothing_alpha=0.6,
        center_match_ratio=0.70,
    ):
        self.iou_threshold = float(iou_threshold if match_thresh is None else match_thresh)
        self.max_missing_seconds = float(max_missing_seconds)
        self.track_thresh = float(track_thresh)
        self.track_buffer = max(0, int(track_buffer))
        self.min_box_area = max(0.0, float(min_box_area))
        self.bbox_smoothing_alpha = min(max(float(bbox_smoothing_alpha), 0.0), 1.0)
        self.center_match_ratio = max(0.0, float(center_match_ratio))
        self._next_track_id = 1
        self._tracks = {}
        self._frame_index = 0
        self.last_diagnostics = self._empty_diagnostics()

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        self._frame_index += 1
        self._mark_tracks_missing()
        stale_tracks = self._drop_stale_tracks(now)

        assigned_track_ids = set()
        output = []
        new_tracks = 0
        id_switch_like_events = 0
        for detection in self._filter_detections(detections):
            detection = dict(detection)
            original_track_id = detection.get("track_id")
            track_id = detection.get("track_id")
            if track_id is None:
                track_id = self._match_existing_track(detection.get("bbox"), assigned_track_ids)
                if track_id is None:
                    track_id = self._allocate_track_id()
                    new_tracks += 1
            elif int(track_id) in self._tracks:
                track_id = int(track_id)
            else:
                track_id = int(track_id)
                new_tracks += 1
            track_id = int(track_id)
            if original_track_id is not None and int(original_track_id) != track_id:
                id_switch_like_events += 1
            detection["track_id"] = track_id
            assigned_track_ids.add(track_id)
            self._update_track(track_id, detection, now)
            self._copy_track_fields(detection, self._tracks[track_id])
            output.append(detection)
        lost_tracks = stale_tracks + self._drop_buffer_expired_tracks()
        self.last_diagnostics = self._build_diagnostics(new_tracks, lost_tracks, id_switch_like_events)
        return output

    def _allocate_track_id(self):
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _match_existing_track(self, bbox, assigned_track_ids):
        best_track_id = None
        best_score = -1.0
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            iou_score = bbox_iou(bbox, predicted_bbox(track))
            center_ratio = center_distance_ratio(bbox, predicted_bbox(track))
            if iou_score < self.iou_threshold and center_ratio > self.center_match_ratio:
                continue
            score = iou_score + max(0.0, self.center_match_ratio - center_ratio)
            if score > best_score:
                best_score = score
                best_track_id = track_id
        return best_track_id

    def _filter_detections(self, detections):
        output = []
        for detection in detections:
            if float(detection.get("confidence", 1.0)) < self.track_thresh:
                continue
            if bbox_area(detection.get("bbox")) < self.min_box_area:
                continue
            output.append(detection)
        return output

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

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.get("last_seen_at", 0.0) > self.max_missing_seconds
        ]
        for track_id in stale_track_ids:
            del self._tracks[track_id]
        return len(stale_track_ids)

    def _drop_buffer_expired_tracks(self):
        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if int(track.get("missing_frames", 0)) > self.track_buffer
        ]
        for track_id in stale_track_ids:
            del self._tracks[track_id]
        return len(stale_track_ids)

    def _build_diagnostics(self, new_tracks, lost_tracks, id_switch_like_events):
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
        return {
            "active_tracks": len(self._tracks),
            "new_tracks": int(new_tracks),
            "lost_tracks": int(lost_tracks),
            "id_switch_like_events": int(id_switch_like_events),
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
