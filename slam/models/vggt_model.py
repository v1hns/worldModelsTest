"""
VGGT adapter — wraps facebook/VGGT-1B behind BaseReconstructionModel.

Install VGGT before use:
    git clone https://github.com/facebookresearch/vggt && pip install -e ./vggt
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List

from slam.base_model import BaseReconstructionModel, ModelPrediction


class VGGTModel(BaseReconstructionModel):

    def __init__(self, hf_id: str = "facebook/VGGT-1B"):
        self._hf_id = hf_id
        self._model = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return f"VGGT ({self._hf_id})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str) -> None:
        self._device = device
        try:
            from vggt.models.vggt import VGGT
            self._model = VGGT.from_pretrained(self._hf_id).to(device)
            self._model.eval()
            print(f"[{self.name}] Loaded from HuggingFace on {device}.")
        except ImportError:
            self._model = None
            print(
                f"[{self.name}] vggt package not found — "
                "install with: pip install -e ./vggt\n"
                "              Falling back to random dummy outputs."
            )

    # ------------------------------------------------------------------
    # Per-frame operations
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).to(self._device)  # CHW

    def predict(self, tensors: List[torch.Tensor]) -> ModelPrediction:
        N = len(tensors)
        H, W = tensors[0].shape[1], tensors[0].shape[2]

        if self._model is not None:
            batch = torch.stack(tensors).unsqueeze(0)  # 1×N×C×H×W
            with torch.no_grad():
                raw = self._model(batch)

            extrinsics = raw.get("extrinsic", raw.get("extrinsics"))[0]  # N×3×4
            intrinsics = raw.get("intrinsic", raw.get("intrinsics"))[0]  # N×3×3
            depth = raw.get("depth")[0]                                   # N×H×W

            attn = raw.get("attention_features")
            if attn is None:
                attn = [depth[i].mean().unsqueeze(0) for i in range(N)]
        else:
            # ── dummy fallback ─────────────────────────────────────────
            extrinsics = torch.eye(3, 4).unsqueeze(0).expand(N, -1, -1).clone()
            extrinsics[:, :3, 3] = torch.randn(N, 3) * 0.1
            intrinsics = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()
            depth = torch.rand(N, H, W) * 5.0
            attn = [torch.rand(256) for _ in range(N)]

        return ModelPrediction(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            retrieval_features=attn,
        )
