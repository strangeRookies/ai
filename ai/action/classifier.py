from pathlib import Path

from ai.action.lstm_contract import (
    DEFAULT_KEYPOINT_COUNT,
    DEFAULT_KEYPOINT_INPUT_SIZE,
    MOTION_KEYPOINT_INPUT_SIZE,
)
from ai.action.feature_schema import (
    KEYPOINT51_SCHEMA_VERSION,
    KEYPOINT_BBOX54_SCHEMA_VERSION,
    KEYPOINT_MOTION54_SCHEMA_VERSION,
    KNOWN_KEYPOINT_SCHEMAS,
    feature_dim_for_schema,
    feature_names_for_schema,
)
from ai.action.motion_features import append_motion_features

DEFAULT_CLASSES = ("Normal", "Faint")
KEYPOINT_FEATURE_DIM = DEFAULT_KEYPOINT_INPUT_SIZE
MOTION_KEYPOINT_FEATURE_DIM = MOTION_KEYPOINT_INPUT_SIZE


class ActionClassifier:
    def predict(self, sequence):
        raise NotImplementedError


class LSTMActionModel:
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=2, dropout=0.0):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(f"torch is required for LSTM action inference: {exc}") from exc

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                output, _ = self.lstm(x)
                return self.head(output[:, -1, :])

        self.torch = torch
        self.model = _Model()


class LSTMActionClassifier(ActionClassifier):
    def __init__(self, checkpoint_path, device="auto", faint_threshold=0.5):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"LSTM action checkpoint not found: {checkpoint_path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"torch is required for LSTM action inference: {exc}") from exc

        self.torch = torch
        self.device = torch.device(normalize_torch_device(device, torch))
        self.faint_threshold = float(faint_threshold)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        self.classes = classes_from_checkpoint(checkpoint)
        model_cfg = checkpoint["model_config"]
        wrapper = LSTMActionModel(**model_cfg)
        wrapper.model.load_state_dict(checkpoint["model_state"])
        self.model = wrapper.model.to(self.device)
        self.model.eval()
        self.input_size = int(model_cfg.get("input_size", checkpoint.get("feature_size", 32)))
        self.crop_feature_size = int(checkpoint.get("crop_feature_size", checkpoint.get("feature_size", 32)))
        self.feature_size = self.crop_feature_size
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sequence_length = optional_int(checkpoint.get("sequence_length"))
        self.checkpoint_sequence_stride = optional_int(checkpoint.get("sequence_stride"))
        self.last_runtime_feature_dim = None
        self.last_tensor_shape = None
        self._runtime_logged = False

        # Load feature schema metadata
        self.feature_schema = checkpoint.get("feature_schema_version")
        if self.feature_schema is None and self.input_size == 51:
            self.feature_schema = KEYPOINT51_SCHEMA_VERSION
        if self.feature_schema is None:
            raise ValueError(
                f"Checkpoint metadata missing feature_schema_version for input_size={self.input_size}. "
                f"checkpoint={checkpoint_path}"
            )
        if self.feature_schema not in KNOWN_KEYPOINT_SCHEMAS and self.input_size in {51, 54}:
            raise ValueError(
                f"Unknown feature_schema_version={self.feature_schema!r} for input_size={self.input_size}. "
                f"checkpoint={checkpoint_path}"
            )
        self.feature_names = list(checkpoint.get("feature_names") or [])

        # Safeguard: validate checkpoint metadata consistency for keypoint schemas
        if self.feature_schema in KNOWN_KEYPOINT_SCHEMAS:
            schema_dim = feature_dim_for_schema(self.feature_schema)
            if self.input_size != schema_dim:
                raise ValueError(
                    f"Checkpoint Metadata Mismatch: model input_size={self.input_size} "
                    f"does not match feature_schema={self.feature_schema} (dimension {schema_dim})"
                )
            if self.feature_names and len(self.feature_names) != self.input_size:
                raise ValueError(
                    f"Checkpoint Metadata Mismatch: feature_names length={len(self.feature_names)} "
                    f"does not match input_size={self.input_size}. checkpoint={checkpoint_path}"
                )
            # Reject motion↔bbox collisions when names are present (strict for 54-dim schemas).
            if self.feature_schema in {
                KEYPOINT_MOTION54_SCHEMA_VERSION,
                KEYPOINT_BBOX54_SCHEMA_VERSION,
            } and self.feature_names:
                expected_names = feature_names_for_schema(self.feature_schema)
                if list(self.feature_names) != list(expected_names):
                    raise ValueError(
                        f"Checkpoint schema collision: feature_schema={self.feature_schema} "
                        f"does not match feature_names (e.g. motion54 vs bbox54). "
                        f"checkpoint={checkpoint_path}"
                    )

    def predict(self, sequence):
        if not sequence:
            return None
        features = sequence_to_lstm_features(
            sequence,
            self.input_size,
            self.crop_feature_size,
            getattr(self, "feature_schema", KEYPOINT51_SCHEMA_VERSION),
        )
        self.last_runtime_feature_dim = int(features.shape[-1])
        self.last_tensor_shape = (1, *tuple(int(dim) for dim in features.shape))

        if "detections" in sequence:
            actual_input_size = int(features.shape[-1])
            if actual_input_size != self.input_size:
                raise ValueError(
                    "LSTMActionClassifier Mismatch: "
                    f"cameraLoginId={sequence.get('camera_login_id', '')} "
                    f"trackId={sequence.get('track_id', '')} "
                    f"expected input_size={self.input_size} "
                    f"runtime_feature_dim={actual_input_size} "
                    f"feature_schema_version={self.feature_schema} "
                    f"checkpoint={self.checkpoint_path}"
                )

        x = self.torch.from_numpy(features).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]
            score, idx = self.torch.max(probs, dim=0)
        if not self._runtime_logged and "detections" in sequence:
            self._log_first_runtime(sequence, features)
        probabilities = {label: float(probs[class_idx].item()) for class_idx, label in enumerate(self.classes)}
        threshold_result = threshold_prediction(probabilities, self.faint_threshold)
        if threshold_result:
            label, score_value = threshold_result
        else:
            label = self.classes[int(idx.item())]
            score_value = float(score.item())
        return {"label": label, "score": float(score_value), "probabilities": probabilities}

    def _log_first_runtime(self, sequence, features) -> None:
        try:
            import numpy as np
        except ImportError:
            return
        arr = np.asarray(features, dtype=np.float32)
        finite = bool(np.isfinite(arr).all())
        motion = arr[:, 51:54] if arr.ndim == 2 and arr.shape[-1] >= 54 else None
        if motion is not None and motion.size:
            center_drop_range = f"{float(motion[:, 0].min()):.6f},{float(motion[:, 0].max()):.6f}"
            velocity_range = f"{float(motion[:, 1].min()):.6f},{float(motion[:, 1].max()):.6f}"
            torso_angle_range = f"{float(motion[:, 2].min()):.6f},{float(motion[:, 2].max()):.6f}"
        else:
            center_drop_range = velocity_range = torso_angle_range = "n/a"
        print(
            "[lstm-runtime] "
            f"cameraLoginId={sequence.get('camera_login_id', '')} "
            f"trackId={sequence.get('track_id', '')} "
            f"tensor_shape={self.last_tensor_shape} "
            f"finite={str(finite).lower()} "
            f"center_drop_range={center_drop_range} "
            f"velocity_range={velocity_range} "
            f"torso_angle_range={torso_angle_range}",
            flush=True,
        )
        self._runtime_logged = True


class MockActionClassifier(ActionClassifier):
    def __init__(self, default_label="Fight", score=0.85):
        self.default_label = default_label
        self.score = score

    def predict(self, sequence):
        # TODO: replace with a PyTorch video/action-recognition model.
        return {"label": self.default_label, "score": float(self.score)}


def crops_to_features(crops, feature_size=32):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"opencv-python and numpy are required for crop features: {exc}") from exc

    features = []
    for crop in crops:
        if crop is None:
            features.append(np.zeros((feature_size * feature_size,), dtype=np.float32))
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if getattr(crop, "ndim", 0) == 3 else crop
        small = cv2.resize(gray, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
        features.append((small.astype(np.float32) / 255.0).reshape(-1))
    return np.stack(features, axis=0).astype(np.float32)


def classes_from_checkpoint(checkpoint):
    return list(checkpoint.get("classes", DEFAULT_CLASSES))


def sequence_to_lstm_features(sequence, input_size=KEYPOINT_FEATURE_DIM, crop_feature_size=32, feature_schema="keypoint51"):
    if "detections" in sequence:
        return keypoint_sequence_to_features(sequence, expected_input_size=int(input_size), feature_schema=feature_schema)
    if "crops" in sequence:
        return crops_to_features(sequence["crops"], crop_feature_size)
    raise RuntimeError("sequence must contain keypoint detections or crops")


def keypoint_sequence_to_features(
    sequence,
    keypoint_count=DEFAULT_KEYPOINT_COUNT,
    expected_input_size=KEYPOINT_FEATURE_DIM,
    feature_schema="keypoint51",
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"numpy is required for keypoint LSTM features: {exc}") from exc

    detections = sequence.get("detections") or []
    frame_shapes = sequence.get("frame_shapes") or []
    rows = []
    for index, detection in enumerate(detections):
        shape = frame_shapes[index] if index < len(frame_shapes) else None
        rows.append(keypoints_to_feature(detection, shape, keypoint_count))
    base_features = np.stack(rows, axis=0).astype(np.float32)

    expected_input_size = int(expected_input_size)
    schema = str(feature_schema or KEYPOINT51_SCHEMA_VERSION)

    if schema == KEYPOINT_BBOX54_SCHEMA_VERSION:
        features = append_bbox_features(base_features, sequence)
    elif schema == KEYPOINT_MOTION54_SCHEMA_VERSION:
        features = append_motion_features(base_features)
    elif schema == KEYPOINT51_SCHEMA_VERSION:
        features = base_features
    elif expected_input_size == 54:
        raise ValueError(
            f"Refusing to build 54-dim features for unknown feature_schema={schema!r}; "
            f"use {KEYPOINT_MOTION54_SCHEMA_VERSION} or {KEYPOINT_BBOX54_SCHEMA_VERSION}"
        )
    else:
        features = base_features

    actual = int(features.shape[-1])
    if actual != expected_input_size:
        if schema in {KEYPOINT_MOTION54_SCHEMA_VERSION, KEYPOINT_BBOX54_SCHEMA_VERSION, KEYPOINT51_SCHEMA_VERSION}:
            raise ValueError(
                f"Feature schema/dimension mismatch: schema={schema} produced dim={actual} "
                f"but expected_input_size={expected_input_size}"
            )
        return normalize_feature_width(features, expected_input_size, feature_schema=schema)
    return features.astype(np.float32)


def append_bbox_features(base_features, sequence):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"numpy is required for bbox features: {exc}") from exc

    seq_len = base_features.shape[0]
    detections = sequence.get("detections") or []
    frame_shapes = sequence.get("frame_shapes") or []
    
    bbox_features = np.zeros((seq_len, 3), dtype=np.float32)
    for idx in range(seq_len):
        if idx >= len(detections):
            continue
        det = detections[idx]
        if not det or "bbox" not in det or det["bbox"] is None:
            continue
        bbox = det["bbox"]
        shape = frame_shapes[idx] if idx < len(frame_shapes) else None
        w_frame, h_frame = infer_frame_size(det, shape)
        
        bx1, by1, bx2, by2 = [float(v) for v in bbox[:4]]
        b_w = max(bx2 - bx1, 0.0)
        b_h = max(by2 - by1, 0.0)
        
        w_norm = b_w / max(w_frame, 1.0)
        h_norm = b_h / max(h_frame, 1.0)
        area_norm = w_norm * h_norm
        
        bbox_features[idx, 0] = w_norm
        bbox_features[idx, 1] = h_norm
        bbox_features[idx, 2] = area_norm
        
    return np.concatenate([base_features, bbox_features], axis=1).astype(np.float32)


def keypoints_to_feature(detection, frame_shape=None, keypoint_count=17):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"numpy is required for keypoint LSTM features: {exc}") from exc

    width, height = infer_frame_size(detection, frame_shape)
    keypoints = detection.get("keypoints") or []
    features = []
    for idx in range(keypoint_count):
        if idx >= len(keypoints) or keypoints[idx] is None:
            features.extend([0.0, 0.0, 0.0])
            continue
        point = keypoints[idx]
        features.extend(
            [
                float(point.get("x", 0.0)) / max(float(width), 1.0),
                float(point.get("y", 0.0)) / max(float(height), 1.0),
                float(point.get("confidence", 0.0)),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def normalize_feature_width(features, expected_input_size, camera_login_id=None, checkpoint_path=None, feature_schema="keypoint51"):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"numpy is required for LSTM feature normalization: {exc}") from exc

    actual_input_size = int(features.shape[-1])
    expected_input_size = int(expected_input_size)
    if actual_input_size == expected_input_size:
        return features.astype(np.float32)
    if actual_input_size < expected_input_size and feature_schema == KEYPOINT_BBOX54_SCHEMA_VERSION:
        raise ValueError(
            "keypoint_bbox54 requires bbox_width_norm/bbox_height_norm/bbox_area_norm "
            f"features; refusing silent padding: actual_input_size={actual_input_size} "
            f"expected_input_size={expected_input_size} checkpoint={checkpoint_path or ''}"
        )
    print(
        "[lstm-feature] "
        f"camera_login_id={camera_login_id or ''} "
        f"expected_input_size={expected_input_size} "
        f"actual_input_size={actual_input_size} "
        f"seq_shape={tuple(features.shape)} "
        f"checkpoint={checkpoint_path or ''}",
        flush=True,
    )
    if actual_input_size > expected_input_size:
        normalized = features[..., :expected_input_size].astype(np.float32)
        print(f"[lstm-feature] normalized keypoint feature: {actual_input_size} -> {expected_input_size}", flush=True)
        return normalized
    pad_width = expected_input_size - actual_input_size
    normalized = np.pad(features, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)
    print(f"[lstm-feature] normalized keypoint feature: {actual_input_size} -> {expected_input_size}", flush=True)
    return normalized


def optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def infer_frame_size(detection, frame_shape=None):
    if frame_shape and len(frame_shape) >= 2:
        return float(frame_shape[1]), float(frame_shape[0])
    bbox = detection.get("bbox") or [0, 0, 1, 1]
    if len(bbox) >= 4:
        return max(float(bbox[2]), 1.0), max(float(bbox[3]), 1.0)
    return 1.0, 1.0


def normalize_torch_device(device, torch_module):
    raw = str(device).strip().lower()
    if raw in {"auto", ""}:
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if raw.isdigit():
        return f"cuda:{raw}" if torch_module.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return raw


def threshold_prediction(probabilities, faint_threshold):
    if "Faint" not in probabilities or "Normal" not in probabilities:
        return None
    faint_prob = float(probabilities["Faint"])
    normal_prob = float(probabilities["Normal"])
    if faint_prob >= float(faint_threshold):
        return "Faint", faint_prob
    return "Normal", normal_prob
