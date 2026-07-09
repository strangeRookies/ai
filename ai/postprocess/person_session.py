from __future__ import annotations

from dataclasses import dataclass

from tracking.simple_tracker import SimpleTrackAssigner


@dataclass(frozen=True, slots=True)
class PersonSessionConfig:
    track_thresh: float = 0.10
    match_thresh: float = 0.20
    track_buffer: int = 90
    max_missing_seconds: float = 3.0
    center_match_ratio: float = 0.70


class PersonSessionReconnector:
    def __init__(self, config: PersonSessionConfig | None = None) -> None:
        self.config = config or PersonSessionConfig()
        self._assigner = SimpleTrackAssigner(
            track_thresh=self.config.track_thresh,
            match_thresh=self.config.match_thresh,
            track_buffer=self.config.track_buffer,
            min_box_area=10.0,
            bbox_smoothing_alpha=1.0,
            max_missing_seconds=self.config.max_missing_seconds,
            center_match_ratio=self.config.center_match_ratio,
        )
        self._raw_to_session: dict[int, int] = {}
        self._reconnect_success = 0
        self._reconnect_failure = 0

    def update(self, detections: list[dict]) -> list[dict]:
        session_input: list[dict] = []
        for detection in detections:
            raw_track_id = detection.get("track_id")
            item = dict(detection)
            item.pop("track_id", None)
            item["raw_track_id"] = raw_track_id
            session_input.append(item)

        session_output = self._assigner.update(session_input)
        output: list[dict] = []
        for item in session_output:
            result = dict(item)
            raw_track_id = _optional_int(result.get("raw_track_id"))
            session_id = _optional_int(result.get("track_id"))
            if session_id is None:
                output.append(result)
                continue
            result["person_session_id"] = session_id
            if raw_track_id is not None:
                result["raw_track_id"] = raw_track_id
                previous_session = self._raw_to_session.get(raw_track_id)
                if previous_session is None:
                    if session_id in self._raw_to_session.values():
                        self._reconnect_success += 1
                    self._raw_to_session[raw_track_id] = session_id
                elif previous_session != session_id:
                    self._reconnect_failure += 1
            output.append(result)
        return output

    def diagnostics(self) -> dict:
        diagnostics = self._assigner.diagnostics()
        diagnostics["person_session_reconnect_success"] = self._reconnect_success
        diagnostics["person_session_reconnect_failure"] = self._reconnect_failure
        return diagnostics


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)
