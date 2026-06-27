import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import load_yaml, project_path
from utils.labeling import label_clip, load_fall_intervals
from utils.video_io import video_info, write_clip


def find_annotation(video_path: Path, annotation_dir: Path, extensions: list[str]) -> Path | None:
    for ext in extensions:
        candidate = annotation_dir / f"{video_path.stem}{ext}"
        if candidate.exists():
            return candidate
    for ext in extensions:
        matches = sorted(annotation_dir.rglob(f"{video_path.stem}{ext}"))
        if matches:
            return matches[0]
    return None


def annotation_has_bbox(annotation_path: Path | None) -> bool:
    if annotation_path is None or not annotation_path.exists():
        return False
    if annotation_path.suffix.lower() == ".xml":
        try:
            root = ET.parse(annotation_path).getroot()
        except ET.ParseError:
            return False
        encoded = " ".join([elem.tag.lower() for elem in root.iter()])
        encoded += " " + " ".join(
            f"{key.lower()} {value.lower()}" for elem in root.iter() for key, value in elem.attrib.items()
        )
        return any(key in encoded for key in ("bbox", "bndbox", "bounding_box", "xmin", "ymin", "xmax", "ymax"))
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    encoded = json.dumps(data).lower()
    return any(key in encoded for key in ("bbox", "bounding_box", "box2d", "x_min", "y_min"))


def iter_videos(video_dir: Path, extensions: list[str]) -> list[Path]:
    videos: list[Path] = []
    for ext in extensions:
        videos.extend(video_dir.rglob(f"*{ext}"))
        videos.extend(video_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(videos))


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def percent(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return round(min(100.0, max(0.0, done / total * 100)), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fall/normal clips and metadata.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-save-clips", action="store_true")
    parser.add_argument("--project-root", default=None, help="Override project_root from config.")
    parser.add_argument("--metadata-output", default=None, help="Override metadata CSV output path.")
    parser.add_argument("--max-videos-per-domain", type=int, default=None, help="Limit videos per domain for smoke tests.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domain names to process.")
    parser.add_argument("--log-file", default=None, help="Progress log path. Defaults to metadata_dir/prepare_clips.log.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Write progress after this many generated clips.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.project_root:
        config["project_root"] = args.project_root
    clip_cfg = config["clip"]
    paths = config["paths"]
    positive_label_name = config.get("positive_label_name", "Fall")
    metadata_csv = project_path(config, args.metadata_output or paths["metadata_csv"])
    clips_dir = project_path(config, paths["clips_dir"])
    metadata_dir = project_path(config, paths["metadata_dir"])
    log_path = project_path(config, args.log_file) if args.log_file else metadata_dir / "prepare_clips.log"
    progress_path = metadata_dir / "prepare_clips_progress.json"
    log = make_logger(log_path)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bbox_count = 0
    video_count = 0
    clip_count = 0
    selected_domains = [
        (domain, domain_cfg)
        for domain, domain_cfg in config["domains"].items()
        if not args.domains or domain in args.domains
    ]
    domain_videos: dict[str, list[Path]] = {}
    total_videos = 0
    for domain, domain_cfg in selected_domains:
        video_dir = project_path(config, domain_cfg["video_dir"])
        videos = iter_videos(video_dir, clip_cfg["video_extensions"])
        if args.max_videos_per_domain is not None:
            videos = videos[: args.max_videos_per_domain]
        domain_videos[domain] = videos
        total_videos += len(videos)
    log(f"[prepare] start config={args.config} metadata_csv={metadata_csv} clips_dir={clips_dir}")

    for domain, domain_cfg in selected_domains:
        video_dir = project_path(config, domain_cfg["video_dir"])
        annotation_dir = project_path(config, domain_cfg["annotation_dir"])
        videos = domain_videos[domain]
        log(f"[prepare] domain={domain} videos={len(videos)} video_dir={video_dir}")
        for domain_index, video_path in enumerate(videos, start=1):
            video_count += 1
            video_clip_count = 0
            state = {
                "status": "video_started",
                "updated_at": utc_now(),
                "domain": domain,
                "domain_index": domain_index,
                "domain_total": len(videos),
                "video_count": video_count,
                "total_videos": total_videos,
                "domain_percent": percent(domain_index - 1, len(videos)),
                "overall_percent": percent(video_count - 1, total_videos),
                "clips": clip_count,
                "video_path": str(video_path),
            }
            write_json(progress_path, state)
            log(
                "[prepare] video_start "
                f"domain={domain} index={domain_index}/{len(videos)} "
                f"overall={video_count}/{total_videos} overall_percent={percent(video_count - 1, total_videos):.2f}% "
                f"path={video_path}"
            )
            try:
                annotation_path = find_annotation(video_path, annotation_dir, clip_cfg["annotation_extensions"])
                intervals = load_fall_intervals(annotation_path)
                has_bbox = annotation_has_bbox(annotation_path)
                bbox_count += int(has_bbox)
                frame_count, fps, width, height = video_info(video_path)
                log(
                    "[prepare] video_info "
                    f"frames={frame_count} fps={fps:.3f} size={width}x{height} "
                    f"annotation={annotation_path or ''} fall_intervals={len(intervals)}"
                )
                last_start = max(0, frame_count - clip_cfg["clip_len"])
                for start in range(0, last_start + 1, clip_cfg["stride"]):
                    end = start + clip_cfg["clip_len"] - 1
                    if end >= frame_count:
                        break
                    label = label_clip(
                        start,
                        end,
                        intervals,
                        clip_cfg["min_fall_overlap"],
                        clip_cfg["normal_max_event_overlap"],
                        clip_cfg["ambiguous_policy"],
                    )
                    if label is None:
                        continue
                    clip_name = f"{domain}__{video_path.stem}__{start:06d}_{end:06d}.mp4"
                    clip_path = clips_dir / domain / clip_name
                    if clip_cfg.get("save_video_clips", True) and not args.no_save_clips:
                        write_clip(video_path, clip_path, start, end, fps, width, height)
                    rows.append(
                        {
                            "clip_id": clip_path.stem,
                            "clip_path": str(clip_path),
                            "keypoint_path": "",
                            "label": label,
                            "label_name": positive_label_name if label == 1 else "Normal",
                            "domain": domain,
                            "source_video": str(video_path),
                            "annotation_path": str(annotation_path) if annotation_path else "",
                            "start_frame": start,
                            "end_frame": end,
                            "fps": fps,
                            "width": width,
                            "height": height,
                            "has_bbox_label": has_bbox,
                            "fall_intervals": json.dumps(intervals),
                        }
                    )
                    clip_count += 1
                    video_clip_count += 1
                    if args.progress_interval > 0 and clip_count % args.progress_interval == 0:
                        write_json(
                            progress_path,
                            {
                                "status": "clip_written",
                                "updated_at": utc_now(),
                                "domain": domain,
                                "domain_index": domain_index,
                                "domain_total": len(videos),
                                "video_count": video_count,
                                "total_videos": total_videos,
                                "domain_percent": percent(domain_index - 1, len(videos)),
                                "overall_percent": percent(video_count - 1, total_videos),
                                "current_video_percent": percent(end + 1, frame_count),
                                "clips": clip_count,
                                "video_path": str(video_path),
                                "start_frame": start,
                                "end_frame": end,
                                "frame_count": frame_count,
                                "clip_path": str(clip_path),
                            },
                        )
                        log(
                            "[prepare] progress "
                            f"domain={domain} index={domain_index}/{len(videos)} clips={clip_count} "
                            f"overall_percent={percent(video_count - 1, total_videos):.2f}% "
                            f"current_video_percent={percent(end + 1, frame_count):.2f}% last={start}-{end}"
                        )
            except Exception as exc:
                write_json(
                    progress_path,
                    {
                        "status": "failed",
                        "updated_at": utc_now(),
                        "domain": domain,
                        "domain_index": domain_index,
                        "domain_total": len(videos),
                        "video_count": video_count,
                        "total_videos": total_videos,
                        "domain_percent": percent(domain_index - 1, len(videos)),
                        "overall_percent": percent(video_count - 1, total_videos),
                        "clips": clip_count,
                        "video_path": str(video_path),
                        "error": repr(exc),
                    },
                )
                log(f"[prepare] ERROR video={video_path} error={exc!r}")
                raise
            write_json(
                progress_path,
                {
                    "status": "video_done",
                    "updated_at": utc_now(),
                    "domain": domain,
                    "domain_index": domain_index,
                    "domain_total": len(videos),
                    "video_count": video_count,
                    "total_videos": total_videos,
                    "domain_percent": percent(domain_index, len(videos)),
                    "overall_percent": percent(video_count, total_videos),
                    "clips": clip_count,
                    "video_clips": video_clip_count,
                    "video_path": str(video_path),
                },
            )
            log(
                "[prepare] video_done "
                f"domain={domain} index={domain_index}/{len(videos)} "
                f"overall={video_count}/{total_videos} overall_percent={percent(video_count, total_videos):.2f}% "
                f"video_clips={video_clip_count}"
            )

    if not rows:
        log("[prepare] no clips were generated. Check raw data paths and annotation schema.")
    pd.DataFrame(rows).to_csv(metadata_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    mode = "pose_keypoint_classification" if bbox_count == 0 else "pose_keypoint_classification_bbox_available"
    summary = {
        "videos": video_count,
        "clips": len(rows),
        "videos_with_bbox_labels": bbox_count,
        "recommended_mode": mode,
        "metadata_csv": str(metadata_csv),
    }
    (metadata_csv.parent / "dataset_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_json(progress_path, {"status": "done", "updated_at": utc_now(), **summary})
    log(f"[prepare] done summary={json.dumps(summary, ensure_ascii=False)}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
