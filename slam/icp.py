"""
Small ICP refinement helper for loop-closure alignment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - dependency gated
    cKDTree = None


def refine_relative_pose(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    iterations: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Refine a relative transform using point-to-point ICP.

    Returns the refined transform and a confidence score in [0, 1].
    """
    if (
        source_points.size == 0
        or target_points.size == 0
        or cKDTree is None
        or len(source_points) < 10
        or len(target_points) < 10
    ):
        return init_transform.astype(np.float32), 0.0

    transform = init_transform.astype(np.float32).copy()
    src = source_points.astype(np.float32)
    tgt = target_points.astype(np.float32)
    tree = cKDTree(tgt)

    for _ in range(iterations):
        moved = ((transform[:3, :3] @ src.T).T + transform[:3, 3]).astype(np.float32)
        dists, idxs = tree.query(moved, k=1)
        mask = np.isfinite(dists) & (dists < np.percentile(dists, 80))
        if np.count_nonzero(mask) < 6:
            break

        matched_src = moved[mask]
        matched_tgt = tgt[idxs[mask]]
        src_centroid = matched_src.mean(axis=0)
        tgt_centroid = matched_tgt.mean(axis=0)
        src_zero = matched_src - src_centroid
        tgt_zero = matched_tgt - tgt_centroid
        H = src_zero.T @ tgt_zero
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_centroid - (R @ src_centroid)
        step = np.eye(4, dtype=np.float32)
        step[:3, :3] = R.astype(np.float32)
        step[:3, 3] = t.astype(np.float32)
        transform = step @ transform

    moved = ((transform[:3, :3] @ src.T).T + transform[:3, 3]).astype(np.float32)
    dists, _ = tree.query(moved, k=1)
    rmse = float(np.sqrt(np.mean(np.square(dists)))) if len(dists) else float("inf")
    confidence = 1.0 / (1.0 + rmse)
    return transform.astype(np.float32), float(confidence)
