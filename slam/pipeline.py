"""
SLAMPipeline — model-agnostic online SLAM system.

Pass any BaseReconstructionModel to run the full pipeline:
    from slam import SLAMPipeline, SLAMConfig
    from slam.models import VGGTModel, DummyModel

    slam = SLAMPipeline(model=VGGTModel(), config=SLAMConfig())
    for frame in source:
        result = slam.process_frame(frame)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from .base_model import BaseReconstructionModel
from .submap import Submap
from .factor_graph import FactorGraph
from .retrieval import AttentionRetrieval


@dataclass
class SLAMConfig:
    submap_size: int = 10
    loop_closure_k: int = 5
    loop_closure_thresh: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


@dataclass
class FrameResult:
    frame_id: int
    pose: np.ndarray        # (4, 4)
    depth: np.ndarray       # (H, W)
    intrinsics: np.ndarray  # (3, 3)


class SLAMPipeline:
    """
    Online RGB SLAM pipeline decoupled from any specific model.

    The model handles:  loading weights, preprocessing, inference
    The pipeline handles: submaps, factor graph, loop closure, trajectory
    """

    def __init__(
        self,
        model: BaseReconstructionModel,
        config: Optional[SLAMConfig] = None,
    ):
        self.model = model
        self.cfg = config or SLAMConfig()

        self.model.load(self.cfg.device)
        self.retrieval = AttentionRetrieval(device=self.cfg.device)
        self.factor_graph = FactorGraph(device=self.cfg.device)

        self._frame_buffer: List[dict] = []
        self.submaps: List[Submap] = []
        self._global_poses: List[np.ndarray] = []
        self._frame_count = 0

        if self.cfg.verbose:
            print(f"[SLAMPipeline] model={self.model.name}  device={self.cfg.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, image: np.ndarray) -> FrameResult:
        """
        Ingest one RGB frame (H×W×3, uint8 or float32 [0,1]).
        Returns a FrameResult with the estimated pose and depth map.
        """
        tensor = self.model.preprocess(image)
        self._frame_buffer.append({"image": image, "tensor": tensor})

        if len(self._frame_buffer) >= 2:
            preds = self.model.predict([f["tensor"] for f in self._frame_buffer])
            idx = len(self._frame_buffer) - 1
            pose34 = preds.extrinsics[idx].cpu().numpy()
            depth = preds.depth[idx].cpu().numpy()
            intrinsics = preds.intrinsics[idx].cpu().numpy()
            feat = preds.retrieval_features[idx]
            # Attach depth to the latest buffered frame for point-cloud fusion
            self._frame_buffer[-1]["depth"] = depth
        else:
            pose34 = np.eye(3, 4, dtype=np.float32)
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            intrinsics = np.eye(3, dtype=np.float32)
            feat = None

        pose44 = np.eye(4, dtype=np.float32)
        pose44[:3, :] = pose34

        if feat is not None:
            self.retrieval.add(self._frame_count, feat)

        if self._frame_count > 0 and self._frame_count % self.cfg.submap_size == 0:
            self._try_loop_closure()

        self._global_poses.append(pose44)
        self._frame_count += 1

        if len(self._frame_buffer) >= self.cfg.submap_size:
            self._flush_submap()

        return FrameResult(
            frame_id=self._frame_count - 1,
            pose=pose44,
            depth=depth,
            intrinsics=intrinsics,
        )

    def get_trajectory(self) -> np.ndarray:
        if not self._global_poses:
            return np.empty((0, 4, 4), dtype=np.float32)
        return np.stack(self._global_poses, axis=0)

    def get_point_cloud(self) -> np.ndarray:
        pts = [sm.point_cloud for sm in self.submaps if sm.point_cloud is not None]
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(pts, axis=0)

    def reset(self):
        self._frame_buffer.clear()
        self.submaps.clear()
        self._global_poses.clear()
        self._frame_count = 0
        self.factor_graph.reset()
        self.retrieval.reset()

    # ------------------------------------------------------------------
    # Internal helpers  (no model-specific code below this line)
    # ------------------------------------------------------------------

    def _flush_submap(self):
        frames = self._frame_buffer[:]
        start_idx = self._frame_count - len(frames)
        poses = self._global_poses[start_idx:]

        sm = Submap(submap_id=len(self.submaps), frames=frames, poses=poses)
        sm.build_point_cloud()
        self.factor_graph.add_submap(sm)
        self.submaps.append(sm)
        self._frame_buffer.clear()

        if self.cfg.verbose:
            n_pts = len(sm.point_cloud) if sm.point_cloud is not None else 0
            print(f"[SLAMPipeline] Submap {sm.submap_id} | "
                  f"{len(frames)} frames | {n_pts} pts")

    def _try_loop_closure(self):
        candidates = self.retrieval.query(
            query_idx=self._frame_count - 1,
            k=self.cfg.loop_closure_k,
            min_distance=self.cfg.submap_size * 2,
        )
        for cand_idx, score in candidates:
            if score >= self.cfg.loop_closure_thresh:
                self.factor_graph.add_loop_closure(
                    frame_i=cand_idx,
                    frame_j=self._frame_count - 1,
                    score=float(score),
                )
                if self.cfg.verbose:
                    print(f"[SLAMPipeline] Loop closure: "
                          f"{cand_idx} ↔ {self._frame_count - 1} "
                          f"(score={score:.3f})")
