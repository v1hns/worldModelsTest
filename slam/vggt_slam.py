"""
VGGT-SLAM 2.0 implementation based on:
  "VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction"
  Maggio & Carlone, arXiv:2601.19887

Core ideas implemented here:
  - Feed VGGT predictions (per-frame camera poses + depth) into a sliding-window
    factor graph that removes 15-DOF drift.
  - Use VGGT attention features for image-retrieval-based loop-closure without
    any extra training.
  - Pack processed frames into Submaps; align submaps via loop-closure constraints.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .submap import Submap
from .factor_graph import FactorGraph
from .retrieval import AttentionRetrieval


@dataclass
class SLAMConfig:
    submap_size: int = 10          # frames per submap
    loop_closure_k: int = 5        # top-k candidates to verify
    loop_closure_thresh: float = 0.75  # cosine-similarity gate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


class VGGT_SLAM:
    """
    Online RGB SLAM system wrapping VGGT predictions.

    Usage
    -----
    slam = VGGT_SLAM()
    for frame in video_frames:
        result = slam.process_frame(frame)
        print(result.pose)   # 4x4 world-camera transform
    trajectory = slam.get_trajectory()
    """

    def __init__(self, config: Optional[SLAMConfig] = None):
        self.cfg = config or SLAMConfig()
        self._load_vggt()
        self.retrieval = AttentionRetrieval(device=self.cfg.device)
        self.factor_graph = FactorGraph(device=self.cfg.device)

        self._frame_buffer: List[dict] = []   # frames not yet in a submap
        self.submaps: List[Submap] = []
        self._global_poses: List[np.ndarray] = []   # world poses, one per frame
        self._frame_count = 0

        if self.cfg.verbose:
            print(f"[VGGT-SLAM] Initialised on {self.cfg.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, image: np.ndarray) -> "FrameResult":
        """
        Process a single RGB frame (H×W×3, uint8 or float32 [0,1]).

        Returns a FrameResult with the estimated camera pose and depth map.
        """
        img_tensor = self._preprocess(image)

        # Accumulate into window; run VGGT when window is ready
        self._frame_buffer.append({"image": image, "tensor": img_tensor})

        if len(self._frame_buffer) >= 2:
            preds = self._run_vggt([f["tensor"] for f in self._frame_buffer])
            latest_idx = len(self._frame_buffer) - 1
            pose_vggt = preds["extrinsics"][latest_idx].cpu().numpy()   # 3×4
            depth = preds["depth"][latest_idx].cpu().numpy()
            intrinsics = preds["intrinsics"][latest_idx].cpu().numpy()
            attn_feat = preds["attention_features"][latest_idx]
        else:
            # First frame — identity pose
            pose_vggt = np.eye(3, 4, dtype=np.float32)
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            intrinsics = np.eye(3, dtype=np.float32)
            attn_feat = None

        # Convert 3×4 → 4×4
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :] = pose_vggt

        # Store attention feature for loop-closure retrieval
        if attn_feat is not None:
            self.retrieval.add(self._frame_count, attn_feat)

        # Check for loop closures (every submap boundary)
        if (self._frame_count > 0
                and self._frame_count % self.cfg.submap_size == 0):
            self._try_loop_closure()

        self._global_poses.append(pose_4x4)
        self._frame_count += 1

        # Pack into submap when window is full
        if len(self._frame_buffer) >= self.cfg.submap_size:
            self._create_submap()

        return FrameResult(
            frame_id=self._frame_count - 1,
            pose=pose_4x4,
            depth=depth,
            intrinsics=intrinsics,
        )

    def get_trajectory(self) -> np.ndarray:
        """Return (N, 4, 4) array of world-camera poses for all processed frames."""
        if not self._global_poses:
            return np.empty((0, 4, 4), dtype=np.float32)
        return np.stack(self._global_poses, axis=0)

    def get_point_cloud(self) -> np.ndarray:
        """Concatenate point clouds from all submaps. Returns (M, 3) array."""
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_vggt(self):
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
            self._vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(self.cfg.device)
            self._vggt.eval()
            self._vggt_available = True
            if self.cfg.verbose:
                print("[VGGT-SLAM] VGGT-1B loaded from HuggingFace.")
        except ImportError:
            self._vggt = None
            self._vggt_available = False
            if self.cfg.verbose:
                print("[VGGT-SLAM] WARNING: vggt package not found — running in "
                      "dummy mode. Install with: pip install -e ./vggt")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Normalise a HWC uint8/float image to a CHW float32 tensor."""
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        return tensor.to(self.cfg.device)

    def _run_vggt(self, tensors: List[torch.Tensor]) -> dict:
        """
        Run VGGT on a list of image tensors (each CHW float32).

        Falls back to random dummy outputs when vggt is not installed so the
        rest of the pipeline can still be exercised.
        """
        N = len(tensors)
        H, W = tensors[0].shape[1], tensors[0].shape[2]

        if self._vggt_available:
            batch = torch.stack(tensors).unsqueeze(0)  # 1×N×C×H×W
            with torch.no_grad():
                raw = self._vggt(batch)
            # raw is a dict; key names follow vggt convention
            extrinsics = raw.get("extrinsic", raw.get("extrinsics"))   # 1×N×3×4
            intrinsics = raw.get("intrinsic", raw.get("intrinsics"))   # 1×N×3×3
            depth = raw.get("depth")                                    # 1×N×H×W
            # Attempt to extract last attention layer features for retrieval
            attn_feats = raw.get("attention_features", None)

            if attn_feats is None:
                # Fall back to average-pooled depth as a proxy descriptor
                attn_feats = [depth[0, i].mean().unsqueeze(0) for i in range(N)]

            return {
                "extrinsics": extrinsics[0],     # N×3×4
                "intrinsics": intrinsics[0],     # N×3×3
                "depth":      depth[0],          # N×H×W
                "attention_features": attn_feats,
            }
        else:
            # Dummy mode: random plausible values
            extrinsics = torch.eye(3, 4).unsqueeze(0).expand(N, -1, -1).clone()
            extrinsics[:, :3, 3] = torch.randn(N, 3) * 0.1
            intrinsics = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()
            depth = torch.rand(N, H, W) * 5.0
            attn_feats = [torch.rand(256) for _ in range(N)]
            return {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "depth":      depth,
                "attention_features": attn_feats,
            }

    def _create_submap(self):
        """Package the current frame buffer into a Submap and clear the buffer."""
        frames = self._frame_buffer[:]
        start_idx = self._frame_count - len(frames)
        poses = self._global_poses[start_idx:]

        sm = Submap(
            submap_id=len(self.submaps),
            frames=frames,
            poses=poses,
        )
        sm.build_point_cloud()
        self.factor_graph.add_submap(sm)
        self.submaps.append(sm)
        self._frame_buffer.clear()

        if self.cfg.verbose:
            print(f"[VGGT-SLAM] Submap {sm.submap_id} created "
                  f"({len(frames)} frames, "
                  f"{len(sm.point_cloud) if sm.point_cloud is not None else 0} pts)")

    def _try_loop_closure(self):
        """Query the retrieval index for loop-closure candidates and add constraints."""
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
                    print(f"[VGGT-SLAM] Loop closure: "
                          f"frame {cand_idx} ↔ frame {self._frame_count - 1} "
                          f"(score={score:.3f})")


@dataclass
class FrameResult:
    frame_id: int
    pose: np.ndarray         # 4×4 world-camera transform
    depth: np.ndarray        # H×W depth map
    intrinsics: np.ndarray   # 3×3 camera intrinsics
