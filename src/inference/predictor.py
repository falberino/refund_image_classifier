from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


class ImagePredictor:
    def __init__(self, model_path: str | Path, class_map_path: str | Path, image_size: Tuple[int, int]):
        self.model = tf.keras.models.load_model(model_path)
        with Path(class_map_path).open("r", encoding="utf-8") as f:
            self.class_map: Dict[str, str] = json.load(f)
        self.image_size = image_size

    def _prepare(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.image_size)
        arr = np.asarray(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict_pil(self, image: Image.Image) -> Dict[str, object]:
        arr = self._prepare(image)
        probs = self.model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        prob_map = {
            self.class_map[str(i)]: float(np.round(p, 6))
            for i, p in enumerate(probs.tolist())
        }

        return {
            "predicted_label": self.class_map[str(pred_idx)],
            "predicted_index": pred_idx,
            "probabilities": prob_map,
        }

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, object]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil(image)

    def predict_base64(self, b64_data: str) -> Dict[str, object]:
        raw = base64.b64decode(b64_data)
        return self.predict_bytes(raw)


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
