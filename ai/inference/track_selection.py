from __future__ import annotations


class TrackSelector:
    def __init__(self, selected_track_id=None, selected_track_mode="strict", missing_frames_threshold=5):
        self.selected_track_id = int(selected_track_id) if selected_track_id is not None else None
        self.selected_track_mode = selected_track_mode.lower() if selected_track_mode else "strict"
        self.missing_frames_threshold = int(missing_frames_threshold)
        
        self.missing_frames_count = 0
        self.active_fallback_track_id = None
        
    def filter(self, detections):
        if self.selected_track_id is None:
            return list(detections), [], {
                "fallback_active": False,
                "fallback_track_id": None,
                "skipped_reason": None,
                "missing_frames_count": 0
            }
            
        # 1. selected_track_id가 현재 detections에 존재하는지 확인
        selected_detection = None
        for det in detections:
            tid = det.get("track_id")
            if tid is not None and int(float(str(tid))) == self.selected_track_id:
                selected_detection = det
                break
                
        if selected_detection is not None:
            self.missing_frames_count = 0
            self.active_fallback_track_id = None
            
            selected = [selected_detection]
            skipped = []
            for det in detections:
                if det is not selected_detection:
                    tid = det.get("track_id")
                    skipped.append(f"track_id={_track_id_text(tid)} reason=not_selected_track")
                    
            return selected, skipped, {
                "fallback_active": False,
                "fallback_track_id": None,
                "skipped_reason": None,
                "missing_frames_count": 0
            }
            
        # 2. selected_track_id가 존재하지 않는 경우
        self.missing_frames_count += 1
        
        if self.selected_track_mode == "strict":
            selected = []
            skipped = []
            for det in detections:
                tid = det.get("track_id")
                skipped.append(f"track_id={_track_id_text(tid)} reason=selected_track_missing")
            return selected, skipped, {
                "fallback_active": False,
                "fallback_track_id": None,
                "skipped_reason": "selected_track_missing",
                "missing_frames_count": self.missing_frames_count
            }
            
        elif self.selected_track_mode == "fallback":
            if self.missing_frames_count < self.missing_frames_threshold:
                selected = []
                skipped = []
                for det in detections:
                    tid = det.get("track_id")
                    skipped.append(f"track_id={_track_id_text(tid)} reason=selected_track_missing")
                return selected, skipped, {
                    "fallback_active": False,
                    "fallback_track_id": None,
                    "skipped_reason": "selected_track_missing",
                    "missing_frames_count": self.missing_frames_count
                }
            else:
                # threshold 도달 또는 초과
                fallback_detection = None
                if self.active_fallback_track_id is not None:
                    for det in detections:
                        tid = det.get("track_id")
                        if tid is not None and int(float(str(tid))) == self.active_fallback_track_id:
                            fallback_detection = det
                            break
                            
                if fallback_detection is None:
                    candidate_detections = [d for d in detections if d.get("track_id") is not None]
                    if candidate_detections:
                        best_det = max(candidate_detections, key=_selection_score)
                        self.active_fallback_track_id = int(float(str(best_det["track_id"])))
                        fallback_detection = best_det
                        
                if fallback_detection is not None:
                    selected = [fallback_detection]
                    skipped = []
                    for det in detections:
                        if det is not fallback_detection:
                            tid = det.get("track_id")
                            skipped.append(f"track_id={_track_id_text(tid)} reason=not_selected_track")
                    return selected, skipped, {
                        "fallback_active": True,
                        "fallback_track_id": self.active_fallback_track_id,
                        "skipped_reason": "selected_track_missing_fallback",
                        "missing_frames_count": self.missing_frames_count
                    }
                else:
                    self.active_fallback_track_id = None
                    selected = []
                    skipped = []
                    for det in detections:
                        tid = det.get("track_id")
                        skipped.append(f"track_id={_track_id_text(tid)} reason=selected_track_missing")
                    return selected, skipped, {
                        "fallback_active": True,
                        "fallback_track_id": None,
                        "skipped_reason": "selected_track_missing",
                        "missing_frames_count": self.missing_frames_count
                    }
        else:
            selected = []
            skipped = []
            for det in detections:
                tid = det.get("track_id")
                skipped.append(f"track_id={_track_id_text(tid)} reason=selected_track_missing")
            return selected, skipped, {
                "fallback_active": False,
                "fallback_track_id": None,
                "skipped_reason": "selected_track_missing",
                "missing_frames_count": self.missing_frames_count
            }


def filter_selected_track(detections, selected_track_id):
    # 하위 호환성용 일회성 strict 모드 실행
    selector = TrackSelector(selected_track_id=selected_track_id, selected_track_mode="strict")
    selected, skipped, _ = selector.filter(detections)
    return selected, skipped


def deduplicate_tracked_detections(detections, iou_threshold=0.85):
    unique = []
    removed = []
    for detection in detections:
        duplicate_index = _duplicate_index(unique, detection, float(iou_threshold))
        if duplicate_index is None:
            unique.append(detection)
            continue

        existing = unique[duplicate_index]
        if _selection_score(detection) > _selection_score(existing):
            unique[duplicate_index] = detection
            removed.append(f"track_id={_track_id_text(existing.get('track_id'))} reason=duplicate_bbox")
        else:
            removed.append(f"track_id={_track_id_text(detection.get('track_id'))} reason=duplicate_bbox")
    return unique, removed


def _duplicate_index(unique, detection, iou_threshold):
    track_id = detection.get("track_id")
    for index, existing in enumerate(unique):
        existing_track_id = existing.get("track_id")
        if track_id is not None and existing_track_id is not None:
            if int(float(str(track_id))) == int(float(str(existing_track_id))):
                return index
            continue
        if _bbox_iou(existing.get("bbox"), detection.get("bbox")) >= iou_threshold:
            return index
    return None


def _selection_score(detection):
    keypoint_count = len(detection.get("keypoints") or [])
    confidence = float(detection.get("confidence", 0.0))
    has_track_id = detection.get("track_id") is not None
    return has_track_id, keypoint_count, confidence, _bbox_area(detection.get("bbox"))


def _bbox_iou(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def _track_id_text(track_id):
    if track_id is None:
        return "None"
    return str(int(float(str(track_id))))
