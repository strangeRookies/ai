import itertools


class MockDetector:
    def __init__(self, model_name="mock-detector"):
        self.model_name = model_name
        self._counter = itertools.count(1)

    def detect(self, frame):
        index = next(self._counter)
        if index < 4:
            bbox = [120, 80, 220, 360]
            pose_state = "STANDING"
            rule_hint = 0.2
        elif index < 8:
            bbox = [110, 150, 330, 300]
            pose_state = "FALLING"
            rule_hint = 0.72
        else:
            bbox = [100, 210, 380, 330]
            pose_state = "LYING"
            rule_hint = 0.92

        return [
            {
                "track_id": 1,
                "bbox": bbox,
                "confidence": 0.87,
                "pose_state": pose_state,
                "pose_horizontal": pose_state in {"FALLING", "LYING"},
                "model_name": self.model_name,
                "rule_hint": rule_hint,
            }
        ]
