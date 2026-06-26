from pathlib import Path


DEFAULT_CLASSES = ("Normal", "Faint")
KEYPOINT_FEATURE_DIM = 54


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

    def predict(self, sequence):
        if not sequence:
            return None
        features = sequence_to_lstm_features(sequence, self.input_size, self.crop_feature_size)
        x = self.torch.from_numpy(features).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]
            score, idx = self.torch.max(probs, dim=0)
        probabilities = {label: float(probs[class_idx].item()) for class_idx, label in enumerate(self.classes)}
        threshold_result = threshold_prediction(probabilities, self.faint_threshold)
        if threshold_result:
            label, score_value = threshold_result
        else:
            label = self.classes[int(idx.item())]
            score_value = float(score.item())
        return {"label": label, "score": float(score_value), "probabilities": probabilities}


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


def sequence_to_lstm_features(sequence, input_size=KEYPOINT_FEATURE_DIM, crop_feature_size=32):
    """Select features that match the loaded checkpoint input size.

    `input_size=51` means keypoint features: 17 keypoints times x, y, and confidence.
    If a non-keypoint checkpoint receives crops, crop features are used instead.
    Crop feature dim is `crop_feature_size * crop_feature_size`.
    """
    if "detections" in sequence and int(input_size) == KEYPOINT_FEATURE_DIM:
        return keypoint_sequence_to_features(sequence)
    if "crops" in sequence:
        return crops_to_features(sequence["crops"], crop_feature_size)
    if "detections" in sequence:
        return keypoint_sequence_to_features(sequence)
    raise RuntimeError("sequence must contain keypoint detections or crops")


def keypoint_sequence_to_features(sequence, keypoint_count=17):
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
    
    try:
        from .motion_features import append_motion_features
        return append_motion_features(base_features)
    except ImportError:
        return base_features


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
