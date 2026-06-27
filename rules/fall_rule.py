import time
from collections import deque


class FallRuleEngine:
    def __init__(
        self,
        min_duration_seconds=1.5,
        debounce_seconds=10,
        candidate_threshold=0.7,
        decision_window=3,
        decision_required=2,
    ):
        self.min_duration_seconds = min_duration_seconds
        self.debounce_seconds = debounce_seconds
        self.candidate_threshold = float(candidate_threshold)
        self.decision_window = max(1, int(decision_window))
        self.decision_required = max(1, int(decision_required))
        self._tracks = {}

    def evaluate(self, detections, now=None):
        now = now or time.time()
        events = []
        for detection in detections:
            track_id = detection.get("track_id") or 0
            state = self._tracks.setdefault(
                track_id,
                {
                    "state": "NORMAL",
                    "candidate_since": None,
                    "last_event_at": 0.0,
                    "recent_candidates": deque(maxlen=self.decision_window),
                },
            )

            score = self._score_detection(detection)
            is_candidate = score >= self.candidate_threshold
            state["recent_candidates"].append(is_candidate)
            recent_candidate_count = sum(1 for value in state["recent_candidates"] if value)
            is_stable_candidate = recent_candidate_count >= self.decision_required

            if is_candidate:
                if state["candidate_since"] is None:
                    state["candidate_since"] = now
                    state["state"] = "CANDIDATE"
                elif is_stable_candidate and now - state["candidate_since"] >= self.min_duration_seconds:
                    state["state"] = "CONFIRMED"
                    if now - state["last_event_at"] >= self.debounce_seconds:
                        state["last_event_at"] = now
                        detection["event_status"] = "confirmed"
                        detection["decision_window"] = self.decision_window
                        detection["decision_votes"] = recent_candidate_count
                        events.append((detection, score, "LYING"))
                    else:
                        state["state"] = "NOTIFIED"
            else:
                state["candidate_since"] = None
                state["state"] = "NORMAL"
        return events

    @staticmethod
    def _score_detection(detection):
        bbox = detection.get("bbox") or [0, 0, 0, 0]
        width = max(float(bbox[2]) - float(bbox[0]), 1.0)
        height = max(float(bbox[3]) - float(bbox[1]), 1.0)
        aspect_score = min(width / height / 1.8, 1.0)
        pose_score = 1.0 if detection.get("pose_horizontal") else 0.0
        hint_score = float(detection.get("rule_hint", 0.0))
        confidence = float(detection.get("confidence", 0.0))
        return round((aspect_score * 0.35) + (pose_score * 0.35) + (hint_score * 0.2) + (confidence * 0.1), 4)
