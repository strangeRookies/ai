import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventLabel:
    video_file_name: str
    frame_count: int
    event_class: str
    event_frames: list[tuple[int, int]]

    def is_active(self, frame_idx):
        return any(start <= frame_idx <= end for start, end in self.event_frames)


def load_event_label(label_path):
    with open(label_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    metadata = data.get("metadata", {})
    annotations = data.get("annotations", {})
    event_frames = [tuple(map(int, item)) for item in annotations.get("event_frame", [])]
    return EventLabel(
        video_file_name=metadata.get("file_name", ""),
        frame_count=int(metadata.get("frame_count", 0) or 0),
        event_class=annotations.get("event_class", "Unknown"),
        event_frames=event_frames,
    )


def load_dataset_rows(csv_path, split=None):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if split and row.get("split") != split:
                continue
            rows.append(
                {
                    "video_path": row["video_path"],
                    "label_path": row["label_path"],
                    "split": row.get("split", split or ""),
                }
            )
    return rows


def resolve_label_for_video(video_path, label_dir):
    video = Path(video_path)
    label_dir = Path(label_dir)
    direct = label_dir / f"{video.stem}.json"
    if direct.exists():
        return direct
    for label_path in label_dir.rglob("*.json"):
        try:
            if load_event_label(label_path).video_file_name == video.name:
                return label_path
        except Exception:
            continue
    return None
