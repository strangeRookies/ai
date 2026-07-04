from __future__ import annotations

from collections import Counter


DIAGNOSIS_TO_ISSUE = {
    "detector_person_missing": "detector_missing",
    "pose_quality_low": "low_pose_quality",
    "tracking_association_problem": "tracker_association_issue",
    "sequence_buffer_not_ready": "sequence_buffer_issue",
    "pose_tracking_ok": "normal",
}


class PoseStats:
    def __init__(self) -> None:
        self.total_frames = 0
        self.frames_with_person = 0
        self.frames_without_person = 0
        self.person_count_total = 0
        self.bbox_conf_total = 0.0
        self.bbox_conf_count = 0
        self.keypoint_conf_total = 0.0
        self.keypoint_conf_count = 0
        self.valid_keypoints_total = 0
        self.tracker_active_frames = 0
        self.active_tracks_zero_count = 0
        self.sequence_ready_count = 0
        self.relink_success_count = 0
        self.relink_fail_count = 0
        self.id_switch_like_events = 0
        self.issue_counts = Counter()

    def observe(self, record: dict) -> None:
        self.total_frames += 1
        person_count = int(record["raw_detection_count"])
        self.person_count_total += person_count
        if person_count > 0:
            self.frames_with_person += 1
        else:
            self.frames_without_person += 1
        self._observe_optional_average(record.get("avg_bbox_confidence"), "bbox")
        self._observe_optional_average(record.get("avg_keypoint_confidence"), "keypoint")
        self.valid_keypoints_total += int(record.get("valid_keypoint_count", 0))
        active_tracks = int(record.get("active_tracks", 0))
        if active_tracks > 0:
            self.tracker_active_frames += 1
        else:
            self.active_tracks_zero_count += 1
        self.sequence_ready_count += int(record.get("sequenceReadyCount", 0))
        self.relink_success_count = max(self.relink_success_count, int(record.get("relink_success_count", 0)))
        self.relink_fail_count = max(self.relink_fail_count, int(record.get("relink_fail_count", 0)))
        self.id_switch_like_events += int(record.get("id_switch_like_events", 0))
        issue = DIAGNOSIS_TO_ISSUE.get(str(record.get("diagnosis")), "normal")
        self.issue_counts[issue] += 1

    def summary(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "frames_with_person": self.frames_with_person,
            "frames_without_person": self.frames_without_person,
            "avg_person_count": _rounded_div(self.person_count_total, self.total_frames),
            "avg_bbox_conf": _rounded_div(self.bbox_conf_total, self.bbox_conf_count),
            "avg_keypoint_conf": _rounded_div(self.keypoint_conf_total, self.keypoint_conf_count),
            "avg_valid_keypoints": _rounded_div(self.valid_keypoints_total, self.total_frames),
            "tracker_active_rate": _rounded_div(self.tracker_active_frames, self.total_frames),
            "sequence_ready_count": self.sequence_ready_count,
            "active_tracks_zero_count": self.active_tracks_zero_count,
            "relink_success_count": self.relink_success_count,
            "relink_fail_count": self.relink_fail_count,
        }

    def final_summary(self) -> dict:
        issue_counts = {
            "detector_missing": self.issue_counts.get("detector_missing", 0),
            "low_pose_quality": self.issue_counts.get("low_pose_quality", 0),
            "tracker_association_issue": self.issue_counts.get("tracker_association_issue", 0),
            "frequent_track_id_switch": self.id_switch_like_events,
            "sequence_buffer_issue": self.issue_counts.get("sequence_buffer_issue", 0),
            "normal": self.issue_counts.get("normal", 0),
        }
        summary = self.summary()
        summary["issue_counts"] = issue_counts
        summary["primary_issue"] = _primary_issue(issue_counts)
        return summary

    def _observe_optional_average(self, value, metric: str) -> None:
        if value is None:
            return
        if metric == "bbox":
            self.bbox_conf_total += float(value)
            self.bbox_conf_count += 1
        elif metric == "keypoint":
            self.keypoint_conf_total += float(value)
            self.keypoint_conf_count += 1


def _primary_issue(issue_counts: dict[str, int]) -> str:
    non_normal = {key: value for key, value in issue_counts.items() if key != "normal" and value > 0}
    if not non_normal:
        return "normal"
    return max(non_normal.items(), key=lambda item: item[1])[0]


def _rounded_div(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)

