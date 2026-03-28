"""
Attention-based image retrieval for loop-closure detection.

From the paper (Section IV):
  "We leverage attention layers from VGGT for image retrieval verification
   without requiring additional training, enabling false-match rejection."

VGGT's transformer produces per-image attention feature vectors. We index
these with a simple cosine-similarity search. In the full system this would
use approximate nearest-neighbour (e.g. FAISS) for real-time performance.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class AttentionRetrieval:
    """
    Stores per-frame attention feature vectors and provides top-k
    cosine-similarity search, used to propose loop-closure candidates.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._frame_ids: List[int] = []
        self._features: Optional[torch.Tensor] = None   # (N, D)

    def add(self, frame_id: int, feature: torch.Tensor):
        """
        Register a new frame's attention feature.

        feature: 1-D tensor of arbitrary length (will be L2-normalised).
        """
        feat = feature.to(self.device).float().flatten()
        feat = feat / (feat.norm() + 1e-8)

        self._frame_ids.append(frame_id)
        if self._features is None:
            self._features = feat.unsqueeze(0)
        else:
            self._features = torch.cat([self._features, feat.unsqueeze(0)], dim=0)

    def query(
        self,
        query_idx: int,
        k: int = 5,
        min_distance: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k most similar past frames (by cosine similarity)
        that are at least `min_distance` frames before `query_idx`.

        Returns list of (frame_id, similarity_score).
        """
        if self._features is None or len(self._frame_ids) < 2:
            return []

        # Find position of query_idx
        try:
            q_pos = self._frame_ids.index(query_idx)
        except ValueError:
            return []

        if q_pos == 0:
            return []

        q_feat = self._features[q_pos]  # (D,)

        # Only consider frames far enough in the past
        candidates = [
            (i, fid)
            for i, fid in enumerate(self._frame_ids[:q_pos])
            if (query_idx - fid) >= min_distance
        ]
        if not candidates:
            return []

        idxs = [c[0] for c in candidates]
        cand_feats = self._features[idxs]  # (M, D)

        scores = torch.mv(cand_feats, q_feat)  # (M,) cosine sims (already normalised)
        scores_np = scores.cpu().numpy()

        top_k_local = np.argsort(scores_np)[::-1][:k]
        results = []
        for local_i in top_k_local:
            orig_frame_id = candidates[local_i][1]
            sim = float(scores_np[local_i])
            results.append((orig_frame_id, sim))

        return results

    def reset(self):
        self._frame_ids.clear()
        self._features = None

    @property
    def size(self) -> int:
        return len(self._frame_ids)
