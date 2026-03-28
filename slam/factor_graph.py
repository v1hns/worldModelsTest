"""
Factor graph for drift-free pose estimation.

Based on the paper's description (Section III):
  - Consecutive-frame odometry factors from VGGT extrinsics
  - Loop-closure factors from attention-based retrieval
  - Gauge fixing to remove the 15-DOF ambiguity inherent in
    feed-forward monocular reconstruction (scale, rotation, translation)

This is a simplified pose-graph implementation. For production use,
drop in GTSAM (pip install gtsam) and replace _optimize() below.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .submap import Submap


@dataclass
class OdometryFactor:
    frame_i: int
    frame_j: int
    relative_pose: np.ndarray   # 4×4
    weight: float = 1.0


@dataclass
class LoopFactor:
    frame_i: int
    frame_j: int
    score: float


class FactorGraph:
    """
    Lightweight pose-graph that collects odometry + loop-closure factors
    and exposes a simple Gauss-Newton optimizer.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.odometry_factors: List[OdometryFactor] = []
        self.loop_factors: List[LoopFactor] = []
        self._submap_anchors: List[np.ndarray] = []   # anchor pose per submap

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_submap(self, submap: Submap):
        anchor = submap.anchor_pose.copy()
        if self._submap_anchors:
            prev = self._submap_anchors[-1]
            rel = np.linalg.inv(prev) @ anchor
            self.odometry_factors.append(
                OdometryFactor(
                    frame_i=len(self._submap_anchors) - 1,
                    frame_j=len(self._submap_anchors),
                    relative_pose=rel,
                )
            )
        self._submap_anchors.append(anchor)

    def add_loop_closure(self, frame_i: int, frame_j: int, score: float):
        self.loop_factors.append(LoopFactor(frame_i=frame_i, frame_j=frame_j, score=score))

    def reset(self):
        self.odometry_factors.clear()
        self.loop_factors.clear()
        self._submap_anchors.clear()

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self, iterations: int = 10) -> List[np.ndarray]:
        """
        Run a simple pose-graph optimisation and return the corrected
        submap anchor poses.

        Currently implements a linear relaxation (averaging loop corrections
        along the trajectory). Replace with GTSAM LevenbergMarquardtOptimizer
        for full nonlinear optimisation.
        """
        if not self._submap_anchors:
            return []

        poses = [p.copy() for p in self._submap_anchors]

        for _ in range(iterations):
            corrections = [np.zeros(6) for _ in poses]  # se(3) perturbation

            for lf in self.loop_factors:
                i = min(lf.frame_i, len(poses) - 1)
                j = min(lf.frame_j, len(poses) - 1)
                if i == j:
                    continue
                # Compute closure residual
                rel_pred = np.linalg.inv(poses[i]) @ poses[j]
                # Treat as perfect loop (relative pose should be close to identity
                # for very short loops; in production use an ICP-refined transform)
                t_err = rel_pred[:3, 3]
                weight = lf.score
                # Distribute error equally between i and j
                corrections[j][:3] -= weight * t_err / 2
                corrections[i][:3] += weight * t_err / 2

            # Apply corrections (translation only for this simplified version)
            for k, c in enumerate(corrections):
                poses[k][:3, 3] += c[:3]

        return poses

    @property
    def num_loop_closures(self) -> int:
        return len(self.loop_factors)

    @property
    def num_submaps(self) -> int:
        return len(self._submap_anchors)
