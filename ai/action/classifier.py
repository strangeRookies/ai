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
    def __init__(self, checkpoint_path, device="auto"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"torch is required for LSTM action inference: {exc}") from exc

        self.torch = torch
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.classes = checkpoint.get("classes", ["Normal", "Fall"])
        model_cfg = checkpoint["model_config"]
        wrapper = LSTMActionModel(**model_cfg)
        wrapper.model.load_state_dict(checkpoint["model_state"])
        self.model = wrapper.model.to(self.device)
        self.model.eval()
        self.feature_size = int(checkpoint.get("feature_size", 32))

    def predict(self, sequence):
        if not sequence:
            return None
        features = crops_to_features(sequence["crops"], self.feature_size)
        x = self.torch.from_numpy(features).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]
            score, idx = self.torch.max(probs, dim=0)
        return {"label": self.classes[int(idx.item())], "score": float(score.item())}


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
