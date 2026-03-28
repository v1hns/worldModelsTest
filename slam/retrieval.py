"""
Attention-based image retrieval with optional FAISS acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class RetrievalCandidate:
    frame_id: int
    score: float


class AttentionRetrieval:
    """
    Stores per-frame attention feature vectors and provides top-k similarity
    search. Uses FAISS when requested and available, otherwise falls back to a
    brute-force cosine search over normalized descriptors.
    """

    def __init__(self, device: str = "cpu", backend: str = "faiss"):
        self.device = device
        self.backend = backend
        self._frame_ids: List[int] = []
        self._features: Optional[torch.Tensor] = None
        self._faiss_index = None
        self._faiss_available = False
        if backend == "faiss":
            try:
                import faiss  # type: ignore

                self._faiss_available = True
                self._faiss = faiss
            except ImportError:
                self._faiss_available = False

    def add(self, frame_id: int, feature: torch.Tensor):
        feat = feature.to(self.device).float().flatten()
        feat = feat / (feat.norm() + 1e-8)

        self._frame_ids.append(frame_id)
        if self._features is None:
            self._features = feat.unsqueeze(0)
        else:
            self._features = torch.cat([self._features, feat.unsqueeze(0)], dim=0)

        if self.backend == "faiss" and self._faiss_available:
            self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        if self._features is None:
            self._faiss_index = None
            return
        feats = self._features.detach().cpu().numpy().astype("float32")
        index = self._faiss.IndexFlatIP(feats.shape[1])
        index.add(feats)
        self._faiss_index = index

    def query(
        self,
        query_idx: int,
        k: int = 5,
        min_distance: int = 10,
    ) -> List[Tuple[int, float]]:
        if self._features is None or len(self._frame_ids) < 2:
            return []

        try:
            q_pos = self._frame_ids.index(query_idx)
        except ValueError:
            return []

        if q_pos == 0:
            return []

        valid_positions = [
            pos for pos, fid in enumerate(self._frame_ids[:q_pos]) if (query_idx - fid) >= min_distance
        ]
        if not valid_positions:
            return []

        if self.backend == "faiss" and self._faiss_available and self._faiss_index is not None:
            q_feat = self._features[q_pos : q_pos + 1].detach().cpu().numpy().astype("float32")
            scores, idxs = self._faiss_index.search(q_feat, min(len(self._frame_ids), max(k * 4, k)))
            results = []
            valid_set = set(valid_positions)
            for pos, score in zip(idxs[0], scores[0]):
                if pos < 0 or pos not in valid_set:
                    continue
                results.append((self._frame_ids[pos], float(score)))
                if len(results) >= k:
                    break
            return results

        q_feat = self._features[q_pos]
        cand_feats = self._features[valid_positions]
        scores = torch.mv(cand_feats, q_feat).cpu().numpy()
        top_local = np.argsort(scores)[::-1][:k]
        return [(self._frame_ids[valid_positions[i]], float(scores[i])) for i in top_local]

    def reset(self):
        self._frame_ids.clear()
        self._features = None
        self._faiss_index = None

    @property
    def size(self) -> int:
        return len(self._frame_ids)

    @property
    def using_faiss(self) -> bool:
        return self.backend == "faiss" and self._faiss_available
