"""
Open-set object detection integration.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np


class GroundingDINODetector:
    """
    Dependency-gated Grounding DINO wrapper.

    When the package is unavailable we return an empty result, which keeps the
    core SLAM pipeline usable while preserving a real integration point.
    """

    def __init__(
        self,
        device: str = "cpu",
        labels: Sequence[str] = ("object",),
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        config_path: str | None = None,
        weights_path: str | None = None,
    ):
        self.device = device
        self.labels = list(labels)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.config_path = config_path or os.environ.get("GROUNDING_DINO_CONFIG")
        self.weights_path = weights_path or os.environ.get("GROUNDING_DINO_WEIGHTS")
        self._backend = None
        self._model = None
        try:
            from groundingdino.util.inference import load_model, predict  # type: ignore

            self._backend = {"load_model": load_model, "predict": predict}
        except ImportError:
            self._backend = None

    @property
    def available(self) -> bool:
        return self._backend is not None and self.config_path is not None and self.weights_path is not None

    def _ensure_model(self):
        if not self.available or self._model is not None:
            return
        self._model = self._backend["load_model"](self.config_path, self.weights_path, device=self.device)

    def detect(self, image: np.ndarray) -> List[Dict]:
        self._ensure_model()
        if self._model is None or self._backend is None:
            return []

        try:
            caption = " . ".join(self.labels) + " ."
            boxes, logits, phrases = self._backend["predict"](
                model=self._model,
                image=image,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            results = []
            H, W = image.shape[:2]
            for box, confidence, phrase in zip(boxes, logits, phrases):
                cx, cy, w, h = [float(v) for v in box]
                if max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0:
                    cx *= W
                    w *= W
                    cy *= H
                    h *= H
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                results.append(
                    {
                        "label": str(phrase),
                        "score": float(confidence),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
            return results
        except Exception:
            return []
