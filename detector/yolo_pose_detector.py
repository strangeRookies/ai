class YoloPoseDetector:
    def __init__(self, model_name, device="auto"):
        self.model_name = model_name
        self.device = None if device == "auto" else device
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(f"ultralytics is not installed: {exc}") from exc

        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

    def detect(self, frame):
        results = self.model.predict(frame, device=self.device, verbose=False)
        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            xyxy = boxes.xyxy.detach().float().cpu().tolist()
            confs = boxes.conf.detach().float().cpu().tolist() if boxes.conf is not None else []
            track_ids = (
                boxes.id.detach().int().cpu().tolist()
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )

            keypoint_conf = None
            keypoint_xy = None
            keypoints = getattr(result, "keypoints", None)
            if keypoints is not None and keypoints.conf is not None:
                keypoint_conf = keypoints.conf.detach().float().cpu()
                keypoint_xy = keypoints.xy.detach().float().cpu() if keypoints.xy is not None else None

            for idx, bbox in enumerate(xyxy):
                pose_horizontal = _is_pose_horizontal(keypoint_xy, keypoint_conf, idx)
                keypoints = _extract_keypoints(keypoint_xy, keypoint_conf, idx)
                detections.append(
                    {
                        "track_id": track_ids[idx] if idx < len(track_ids) else None,
                        "bbox": [round(float(v), 2) for v in bbox],
                        "confidence": float(confs[idx]) if idx < len(confs) else 0.0,
                        "pose_state": "LYING" if pose_horizontal else "UNKNOWN",
                        "pose_horizontal": pose_horizontal,
                        "keypoints": keypoints,
                        "keypoint_confidence": _average_keypoint_confidence(keypoints),
                        "model_name": self.model_name,
                    }
                )
        return detections


def _is_pose_horizontal(keypoint_xy, keypoint_conf, person_idx, threshold=0.3):
    if keypoint_xy is None or keypoint_conf is None or person_idx >= keypoint_conf.shape[0]:
        return False

    left_shoulder, right_shoulder, left_hip, right_hip = 5, 6, 11, 12
    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    conf = keypoint_conf[person_idx]
    if any(idx >= conf.numel() or float(conf[idx]) < threshold for idx in required):
        return False

    xy = keypoint_xy[person_idx]
    shoulder_center = (xy[left_shoulder] + xy[right_shoulder]) / 2
    hip_center = (xy[left_hip] + xy[right_hip]) / 2
    torso_dx = abs(float(shoulder_center[0] - hip_center[0]))
    torso_dy = abs(float(shoulder_center[1] - hip_center[1]))
    return torso_dx > 0 and torso_dx / max(torso_dy, 1.0) >= 1.3


def _extract_keypoints(keypoint_xy, keypoint_conf, person_idx):
    if keypoint_xy is None or keypoint_conf is None or person_idx >= keypoint_conf.shape[0]:
        return None
    xy = keypoint_xy[person_idx]
    conf = keypoint_conf[person_idx]
    keypoints = []
    for idx in range(min(xy.shape[0], conf.shape[0])):
        keypoints.append(
            {
                "x": round(float(xy[idx][0]), 2),
                "y": round(float(xy[idx][1]), 2),
                "confidence": round(float(conf[idx]), 4),
            }
        )
    return keypoints


def _average_keypoint_confidence(keypoints):
    if not keypoints:
        return None
    return round(sum(item["confidence"] for item in keypoints) / len(keypoints), 4)
