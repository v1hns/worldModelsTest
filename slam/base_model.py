"""
Base interface every reconstruction model must implement.

To add a new model:
  1. Subclass BaseReconstructionModel
  2. Implement all abstract methods
  3. Register it in slam/models/__init__.py
  4. Pass an instance to SLAMPipeline(model=...)
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ModelPrediction:
    """
    Standardised output that every model must return per inference call.

    Fields
    ------
    extrinsics          (N, 3, 4) camera-to-world transforms
    intrinsics          (N, 3, 3) camera intrinsic matrices
    depth               (N, H, W) per-pixel depth maps
    retrieval_features  list of N tensors used for loop-closure retrieval
    """
    extrinsics: torch.Tensor           # (N, 3, 4)
    intrinsics: torch.Tensor           # (N, 3, 3)
    depth: torch.Tensor                # (N, H, W)
    retrieval_features: List[torch.Tensor]   # N × (D,)


class BaseReconstructionModel(ABC):
    """
    Abstract base for any feed-forward reconstruction model that the SLAM
    pipeline can use.  Subclasses only need to override four methods.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, used in log messages."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, device: str) -> None:
        """
        Load weights / initialise the model onto `device`.
        Called once by SLAMPipeline before any frames are processed.
        """

    # ------------------------------------------------------------------
    # Per-frame operations
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert a raw H×W×3 uint8 (or float32 [0,1]) numpy image into
        whatever tensor format the model expects (typically CHW float32).

        Returns a single tensor on the model's device.
        """

    @abstractmethod
    def predict(self, tensors: List[torch.Tensor]) -> ModelPrediction:
        """
        Run inference on a window of N preprocessed image tensors.

        Parameters
        ----------
        tensors : list of N tensors, each CHW float32

        Returns
        -------
        ModelPrediction with all fields populated for all N frames.
        """
