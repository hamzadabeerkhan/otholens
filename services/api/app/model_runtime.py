"""Optional PyTorch inference for the research checkpoint.

The API deliberately keeps the placeholder path available. Real inference is
enabled only when ORTHOLENS_ENABLE_MODEL=true and a checkpoint is mounted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

class ModelNotReady(RuntimeError):
    """Raised when real inference was requested but cannot be performed."""


class CheckpointModel:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._model: Any | None = None
        self._torch: Any | None = None
        self.version = "unknown"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover - exercised in minimal API environments
            raise ModelNotReady("PyTorch inference dependencies are not installed.") from exc
        if not self.path.is_file():
            raise ModelNotReady(f"Model checkpoint was not found at {self.path}.")

        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model
        self._torch = torch
        self.version = str(checkpoint.get("model_version", self.path.stem))

    def predict(self, image: Image.Image) -> tuple[str, list[float]]:
        self._load()
        assert self._model is not None and self._torch is not None
        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        standard_deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = ((array - mean) / standard_deviation).astype(np.float32, copy=False)
        tensor = self._torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)[0].tolist()
        return self.version, [float(value) for value in probabilities]
