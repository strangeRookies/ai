from pathlib import Path

import numpy as np


def cache_path_candidates(row, cache_dir):
    candidates = []
    cache_root = Path(cache_dir)
    keypoint_path = str(row.get("keypoint_path") or "").strip()
    if keypoint_path and keypoint_path.lower() != "nan":
        keypoint = Path(keypoint_path)
        candidates.append(keypoint)
        if not keypoint.is_absolute():
            candidates.append(cache_root / keypoint)
    names = []
    for value in (row.get("clip_id"), row.get("video_path"), row.get("clip_path"), row.get("_resolved_video_path")):
        if value:
            names.append(Path(str(value)).stem)
    for name in dict.fromkeys(names):
        candidates.append(cache_root / f"{name}.npz")
        candidates.append(cache_root / f"{name}.npy")
    return list(dict.fromkeys(candidates))


_CACHE_INDEX = None


def _get_cache_index(cache_dir):
    global _CACHE_INDEX
    if _CACHE_INDEX is None:
        _CACHE_INDEX = {}
        for p in Path(cache_dir).rglob("*.np*"):
            if p.is_file():
                _CACHE_INDEX[p.stem] = p
                if "__" in p.stem:
                    # e.g., 'indoor_chromakey__clip1' -> 'clip1'
                    short_name = p.stem.split("__", 1)[-1]
                    _CACHE_INDEX[short_name] = p
        print(f"[DEBUG] Cache index built with {len(_CACHE_INDEX)} keys", flush=True)


def resolve_keypoint_cache_path(row, cache_dir):
    index = _get_cache_index(cache_dir)
    
    names = []
    for value in (row.get("clip_id"), row.get("video_path"), row.get("clip_path"), row.get("_resolved_video_path")):
        if value:
            names.append(Path(str(value)).stem)
            
    for name in dict.fromkeys(names):
        if name in index:
            return index[name]

    # Fallback to direct checks (for absolute paths or relative keypoint_path entries)
    for candidate in cache_path_candidates(row, cache_dir):
        if candidate.exists():
            return candidate
    return None


def load_keypoint_cache(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    with np.load(path, allow_pickle=True) as data:
        for key in ("data", "keypoints", "poses", "arr_0"):
            if key in data.files:
                return data[key]
        raise KeyError(f"No supported keypoint array key in {path}: {data.files}")


def keypoint_array_to_frames(raw):
    array = np.asarray(raw, dtype=object if np.asarray(raw).dtype == object else np.float32)
    if array.dtype == object:
        return object_keypoint_frames(array)
    if array.ndim == 3:
        return dense_keypoint_frames(array)
    if array.ndim == 4:
        return dense_keypoint_frames(array[:, 0, :, :])
    raise ValueError(f"Expected keypoint cache shape [T,17,C] or [T,N,17,C], got {array.shape}")


def object_keypoint_frames(array):
    frames = []
    for frame_idx, item in enumerate(array.tolist()):
        if isinstance(item, dict):
            detections = item.get("detections") or item.get("persons") or []
            if item.get("keypoints") is not None and not detections:
                detections = [{"keypoints": item["keypoints"], "bbox": item.get("bbox")}]
            shape = item.get("frame_shape") or item.get("shape") or infer_frame_shape(item.get("keypoints", []))
        else:
            detections = [{"keypoints": item}]
            shape = infer_frame_shape(item)
        frames.append({"frame_idx": frame_idx, "detections": normalize_cached_detections(detections), "frame_shape": shape})
    return frames


def dense_keypoint_frames(array):
    frames = []
    for frame_idx, keypoints in enumerate(array):
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": [cached_keypoints_to_detection(keypoints)],
                "frame_shape": infer_frame_shape(keypoints),
            }
        )
    return frames


def normalize_cached_detections(detections):
    normalized = []
    for detection in detections:
        if isinstance(detection, dict):
            keypoints = detection.get("keypoints")
            normalized.append(
                {
                    "bbox": detection.get("bbox"),
                    "keypoints": cached_keypoints_to_points(keypoints if keypoints is not None else []),
                    "track_id": detection.get("track_id"),
                }
            )
        else:
            normalized.append(cached_keypoints_to_detection(detection))
    return normalized


def cached_keypoints_to_detection(keypoints):
    return {"bbox": None, "keypoints": cached_keypoints_to_points(keypoints)}


def cached_keypoints_to_points(keypoints):
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 1:
        return []
    result = []
    for point in points[:17]:
        confidence = float(point[2]) if point.shape[0] >= 3 else 1.0
        result.append({"x": float(point[0]), "y": float(point[1]), "confidence": confidence})
    return result


def infer_frame_shape(keypoints):
    points = np.asarray(keypoints, dtype=np.float32)
    if points.size == 0 or points.ndim < 2:
        return (1, 1, 3)
    xy = points[..., :2]
    finite = xy[np.isfinite(xy)]
    if finite.size == 0 or float(np.nanmax(finite)) <= 1.5:
        return (1, 1, 3)
    return (1, 1, 3)
