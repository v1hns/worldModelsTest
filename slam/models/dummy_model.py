"""
DummyModel — deterministic stub used in unit tests and CI.

Requires zero external dependencies: no GPU, no weights, no network.
Outputs are random but seeded per frame for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List

from slam.base_model import BaseReconstructionModel, ModelPrediction


class DummyModel(BaseReconstructionModel):
    """
    Drop-in replacement for any real model.  Always available, zero setup.

    Useful for:
      - Unit / integration tests
      - CI pipelines with no GPU
      - Smoke-testing the SLAM pipeline without downloading weights
      - Rapid prototyping of new pipeline features
    """

    def __init__(self, feat_dim: int = 64, seed: int = 42):
        self._feat_dim = feat_dim
        self._seed = seed
        self._device = "cpu"
        self._call_count = 0

    @property
    def name(self) -> str:
        return "DummyModel"

    def load(self, device: str) -> None:
        self._device = device  # accepted but ignored

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def predict(self, tensors: List[torch.Tensor]) -> ModelPrediction:
        N = len(tensors)
        H, W = tensors[0].shape[1], tensors[0].shape[2]

        rng = np.random.default_rng(self._seed + self._call_count)
        self._call_count += 1

        # Plausible-looking random extrinsics: identity rotation + small translation
        extrinsics = torch.eye(3, 4).unsqueeze(0).expand(N, -1, -1).clone()
        extrinsics[:, :3, 3] = torch.tensor(
            rng.normal(0, 0.1, (N, 3)), dtype=torch.float32
        )

        intrinsics = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()
        depth = torch.tensor(rng.uniform(0.5, 5.0, (N, H, W)), dtype=torch.float32)
        feats = [
            torch.tensor(rng.standard_normal(self._feat_dim), dtype=torch.float32)
            for _ in range(N)
        ]

        return ModelPrediction(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            retrieval_features=feats,
        )
