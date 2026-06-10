DEFAULT_FAINT_THRESHOLD = 0.3
DEFAULT_MIN_CONSECUTIVE_FAINT = 2
DEFAULT_CAMERA_COOLDOWN_SECONDS = 10.0
DEFAULT_ACTION_MODEL = "benchmark/results/lstm_yolo26n_train1000/YOLO26n-pose/best.pt"


class FaintEventPostProcessor:
    def __init__(self, min_consecutive_faint=DEFAULT_MIN_CONSECUTIVE_FAINT, cooldown_seconds=DEFAULT_CAMERA_COOLDOWN_SECONDS):
        self.min_consecutive_faint = max(1, int(min_consecutive_faint))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_by_camera = {}
        self._last_event_time_by_camera = {}

    def should_trigger(self, camera_id, prediction, timestamp, track_id=None):
        key = event_state_key(camera_id, track_id)
        if not is_alert_prediction(prediction):
            self._consecutive_by_camera[key] = 0
            return False
        consecutive = int(self._consecutive_by_camera.get(key, 0)) + 1
        self._consecutive_by_camera[key] = consecutive
        if consecutive < self.min_consecutive_faint:
            return False
        last_event_time = self._last_event_time_by_camera.get(key)
        if last_event_time is not None and float(timestamp) - float(last_event_time) < self.cooldown_seconds:
            return False
        self._last_event_time_by_camera[key] = float(timestamp)
        return True

    def consecutive_count(self, camera_id, track_id=None):
        return int(self._consecutive_by_camera.get(event_state_key(camera_id, track_id), 0))

    def cooldown_active(self, camera_id, timestamp, track_id=None):
        key = event_state_key(camera_id, track_id)
        last_event_time = self._last_event_time_by_camera.get(key)
        if last_event_time is None:
            return False
        return float(timestamp) - float(last_event_time) < self.cooldown_seconds


def event_state_key(camera_id, track_id=None):
    return f"{camera_id}:track:{track_id}" if track_id is not None else str(camera_id)


def is_alert_prediction(prediction):
    return bool(prediction) and prediction.get("label") != "Normal"


def faint_probability(prediction):
    if not prediction:
        return None
    probabilities = prediction.get("probabilities") or {}
    if "Faint" in probabilities:
        return float(probabilities["Faint"])
    if prediction.get("label") == "Faint":
        return float(prediction.get("score", 0.0))
    return None
