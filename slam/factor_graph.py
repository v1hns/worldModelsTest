"""
Pose-graph optimizer with optional GTSAM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .submap import Submap


@dataclass
class OdometryFactor:
    submap_i: int
    submap_j: int
    relative_pose: np.ndarray
    weight: float = 1.0


@dataclass
class LoopFactor:
    submap_i: int
    submap_j: int
    relative_pose: np.ndarray
    score: float
    source_frame_i: Optional[int] = None
    source_frame_j: Optional[int] = None


class FactorGraph:
    def __init__(self, device: str = "cpu", backend: str = "gtsam", verbose: bool = False):
        self.device = device
        self.backend = backend
        self.verbose = verbose
        self.odometry_factors: List[OdometryFactor] = []
        self.loop_factors: List[LoopFactor] = []
        self._submap_anchors: List[np.ndarray] = []
        self._gtsam = None
        if backend == "gtsam":
            try:
                import gtsam  # type: ignore

                self._gtsam = gtsam
            except ImportError:
                self._gtsam = None

    def add_submap(self, submap: Submap):
        anchor = submap.anchor_pose.copy().astype(np.float32)
        if self._submap_anchors:
            prev = self._submap_anchors[-1]
            rel = (np.linalg.inv(prev) @ anchor).astype(np.float32)
            self.odometry_factors.append(
                OdometryFactor(
                    submap_i=len(self._submap_anchors) - 1,
                    submap_j=len(self._submap_anchors),
                    relative_pose=rel,
                )
            )
        self._submap_anchors.append(anchor)

    def add_loop_closure(
        self,
        submap_i: int,
        submap_j: int,
        relative_pose: np.ndarray,
        score: float,
        source_frame_i: Optional[int] = None,
        source_frame_j: Optional[int] = None,
    ):
        self.loop_factors.append(
            LoopFactor(
                submap_i=submap_i,
                submap_j=submap_j,
                relative_pose=relative_pose.astype(np.float32),
                score=float(score),
                source_frame_i=source_frame_i,
                source_frame_j=source_frame_j,
            )
        )

    def reset(self):
        self.odometry_factors.clear()
        self.loop_factors.clear()
        self._submap_anchors.clear()

    def optimize(self, iterations: int = 10) -> List[np.ndarray]:
        if not self._submap_anchors:
            return []
        if self.backend == "gtsam" and self._gtsam is not None:
            try:
                return self._optimize_gtsam()
            except Exception:
                if self.verbose:
                    print("[FactorGraph] GTSAM optimization failed, falling back to linear solver.")
        return self._optimize_linear(iterations=iterations)

    def _optimize_linear(self, iterations: int = 10) -> List[np.ndarray]:
        poses = [p.copy() for p in self._submap_anchors]
        if not poses:
            return []

        poses[0][:3, 3] = self._submap_anchors[0][:3, 3]
        for _ in range(iterations):
            corrections = [np.zeros(3, dtype=np.float32) for _ in poses]

            for factor in self.odometry_factors:
                pred = np.linalg.inv(poses[factor.submap_i]) @ poses[factor.submap_j]
                err = factor.relative_pose[:3, 3] - pred[:3, 3]
                corrections[factor.submap_i] -= 0.25 * factor.weight * err
                corrections[factor.submap_j] += 0.25 * factor.weight * err

            for factor in self.loop_factors:
                pred = np.linalg.inv(poses[factor.submap_i]) @ poses[factor.submap_j]
                err = factor.relative_pose[:3, 3] - pred[:3, 3]
                corrections[factor.submap_i] -= 0.5 * factor.score * err
                corrections[factor.submap_j] += 0.5 * factor.score * err

            corrections[0] = np.zeros(3, dtype=np.float32)
            for idx, correction in enumerate(corrections):
                poses[idx][:3, 3] += correction

        scale = np.linalg.norm(self._submap_anchors[-1][:3, 3] - self._submap_anchors[0][:3, 3])
        new_scale = np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
        if scale > 1e-6 and new_scale > 1e-6:
            ratio = scale / new_scale
            origin = poses[0][:3, 3].copy()
            for pose in poses[1:]:
                pose[:3, 3] = origin + (pose[:3, 3] - origin) * ratio
        return [p.astype(np.float32) for p in poses]

    def _optimize_gtsam(self) -> List[np.ndarray]:
        gtsam = self._gtsam
        assert gtsam is not None
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6, dtype=np.float64))
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1] * 6, dtype=np.float64))

        for idx, pose in enumerate(self._submap_anchors):
            key = gtsam.symbol("x", idx)
            gtsam_pose = gtsam.Pose3(pose.astype(np.float64))
            initial.insert(key, gtsam_pose)
            if idx == 0:
                graph.add(gtsam.PriorFactorPose3(key, gtsam_pose, prior_noise))

        for factor in self.odometry_factors:
            graph.add(
                gtsam.BetweenFactorPose3(
                    gtsam.symbol("x", factor.submap_i),
                    gtsam.symbol("x", factor.submap_j),
                    gtsam.Pose3(factor.relative_pose.astype(np.float64)),
                    odom_noise,
                )
            )

        for factor in self.loop_factors:
            sigma = max(0.01, 1.0 - factor.score)
            loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma] * 6, dtype=np.float64))
            graph.add(
                gtsam.BetweenFactorPose3(
                    gtsam.symbol("x", factor.submap_i),
                    gtsam.symbol("x", factor.submap_j),
                    gtsam.Pose3(factor.relative_pose.astype(np.float64)),
                    loop_noise,
                )
            )

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        poses: List[np.ndarray] = []
        for idx in range(len(self._submap_anchors)):
            gtsam_pose = result.atPose3(gtsam.symbol("x", idx))
            poses.append(np.asarray(gtsam_pose.matrix(), dtype=np.float32))
        return poses

    @property
    def num_loop_closures(self) -> int:
        return len(self.loop_factors)

    @property
    def num_submaps(self) -> int:
        return len(self._submap_anchors)

    @property
    def using_gtsam(self) -> bool:
        return self.backend == "gtsam" and self._gtsam is not None
