import time


class SimpleTrackAssigner:
    """Small IoU tracker used until a production tracker such as ByteTrack is wired in."""

    def __init__(self, iou_threshold=0.3, max_missing_seconds=2.0):
        self.iou_threshold = float(iou_threshold)
        self.max_missing_seconds = float(max_missing_seconds)
        self._next_track_id = 1
        self._tracks = {}

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)

        assigned_track_ids = set()
        output = []
        for detection in detections:
            detection = dict(detection)
            track_id = detection.get("track_id")
            if track_id is None:
                track_id = self._match_existing_track(detection.get("bbox"), assigned_track_ids)
                if track_id is None:
                    track_id = self._allocate_track_id()
            track_id = int(track_id)
            detection["track_id"] = track_id
            assigned_track_ids.add(track_id)
            self._tracks[track_id] = {"bbox": detection.get("bbox"), "last_seen_at": now}
            output.append(detection)
        return output

    def _allocate_track_id(self):
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _match_existing_track(self, bbox, assigned_track_ids):
        best_track_id = None
        best_iou = 0.0
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            score = bbox_iou(bbox, track.get("bbox"))
            if score > best_iou:
                best_iou = score
                best_track_id = track_id
        if best_iou < self.iou_threshold:
            return None
        return best_track_id

    def _drop_stale_tracks(self, now):
        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.get("last_seen_at", 0.0) > self.max_missing_seconds
        ]
        for track_id in stale_track_ids:
            del self._tracks[track_id]


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
