"""
Core VGGT-SLAM pipeline.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .detection import GroundingDINODetector
from .factor_graph import FactorGraph
from .icp import refine_relative_pose
from .retrieval import AttentionRetrieval
from .submap import Submap
from .utils import compose, ensure_homogeneous, relative_pose


@dataclass
class SLAMConfig:
    submap_size: int = 10
    loop_closure_k: int = 5
    loop_closure_thresh: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True

    optimizer_backend: str = "gtsam"
    retrieval_backend: str = "faiss"
    async_inference: bool = True
    inference_queue_size: int = 2
    batch_size: int = 1
    fp16: bool = False

    viewer_backend: str = "open3d"
    enable_object_detection: bool = False
    detection_labels: Tuple[str, ...] = ("object",)
    detection_box_threshold: float = 0.35
    detection_text_threshold: float = 0.25
    detector_config_path: Optional[str] = None
    detector_weights_path: Optional[str] = None


@dataclass
class FrameResult:
    frame_id: int
    pose: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    detections: Optional[List[Dict]] = None


class VGGT_SLAM:
    def __init__(self, config: Optional[SLAMConfig] = None):
        self.cfg = config or SLAMConfig()
        self._load_vggt()
        self.retrieval = AttentionRetrieval(
            device=self.cfg.device,
            backend=self.cfg.retrieval_backend,
        )
        self.factor_graph = FactorGraph(
            device=self.cfg.device,
            backend=self.cfg.optimizer_backend,
            verbose=self.cfg.verbose,
        )
        self.detector = None
        if self.cfg.enable_object_detection:
            self.detector = GroundingDINODetector(
                device=self.cfg.device,
                labels=list(self.cfg.detection_labels),
                box_threshold=self.cfg.detection_box_threshold,
                text_threshold=self.cfg.detection_text_threshold,
                config_path=self.cfg.detector_config_path,
                weights_path=self.cfg.detector_weights_path,
            )

        self._frame_buffer: List[Dict] = []
        self._frames: List[Dict] = []
        self._raw_global_poses: List[np.ndarray] = []
        self._frame_to_submap: Dict[int, int] = {}
        self.submaps: List[Submap] = []
        self._frame_count = 0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []

        if self.cfg.async_inference:
            self._executor = ThreadPoolExecutor(max_workers=max(1, self.cfg.batch_size))

        if self.cfg.verbose:
            print(f"[VGGT-SLAM] Initialised on {self.cfg.device}")

    def process_frame(self, image: np.ndarray) -> FrameResult:
        img_tensor = self._preprocess(image)
        frame_id = self._frame_count
        frame_record = {
            "frame_id": frame_id,
            "image": image,
            "tensor": img_tensor,
            "raw_pose": np.eye(4, dtype=np.float32),
            "local_pose": np.eye(4, dtype=np.float32),
            "depth": np.zeros((image.shape[0], image.shape[1]), dtype=np.float32),
            "intrinsics": self._default_intrinsics(image.shape[1], image.shape[0]),
            "attention_feature": None,
            "detections": [],
        }

        self._frames.append(frame_record)
        self._frame_buffer.append(frame_record)

        preds = None
        if len(self._frame_buffer) >= 2:
            tensors = [f["tensor"] for f in self._frame_buffer]
            preds = self._run_window_inference(tensors)
            latest_idx = len(self._frame_buffer) - 1
            pose_4x4 = ensure_homogeneous(preds["extrinsics"][latest_idx].cpu().numpy())
            frame_record["raw_pose"] = pose_4x4
            frame_record["depth"] = preds["depth"][latest_idx].detach().cpu().numpy().astype(np.float32)
            frame_record["intrinsics"] = preds["intrinsics"][latest_idx].detach().cpu().numpy().astype(np.float32)
            frame_record["attention_feature"] = preds["attention_features"][latest_idx]
        else:
            pose_4x4 = frame_record["raw_pose"]

        if frame_record["attention_feature"] is not None:
            self.retrieval.add(frame_id, frame_record["attention_feature"])

        if self.detector is not None:
            frame_record["detections"] = self.detector.detect(image)

        self._raw_global_poses.append(frame_record["raw_pose"])
        self._frame_count += 1

        if len(self._frame_buffer) >= self.cfg.submap_size:
            current_submap = self._create_submap()
            self._try_loop_closure(current_submap)
            self._optimize_submaps()

        return FrameResult(
            frame_id=frame_id,
            pose=self.get_trajectory()[frame_id],
            depth=frame_record["depth"],
            intrinsics=frame_record["intrinsics"],
            detections=frame_record["detections"] or None,
        )

    def get_trajectory(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0, 4, 4), dtype=np.float32)

        poses: List[np.ndarray] = []
        for frame in self._frames:
            frame_id = frame["frame_id"]
            submap_id = self._frame_to_submap.get(frame_id)
            if submap_id is None:
                poses.append(frame["raw_pose"].astype(np.float32))
                continue
            submap = self.submaps[submap_id]
            poses.append(submap.world_pose_for_local(frame["local_pose"]))
        return np.stack(poses, axis=0).astype(np.float32)

    def get_point_cloud(self) -> np.ndarray:
        clouds = [submap.point_cloud for submap in self.submaps if submap.point_cloud.size]
        if not clouds:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(clouds, axis=0).astype(np.float32)

    def export_colmap(self, output_dir: str):
        from .export import export_colmap_model

        export_colmap_model(self, output_dir)

    def start_viewer(self):
        from .viewer import Open3DViewer

        viewer = Open3DViewer()
        viewer.update(self.get_trajectory(), self.get_point_cloud(), self.get_labeled_objects())
        return viewer

    def get_labeled_objects(self) -> List[Dict]:
        labels = []
        for submap in self.submaps:
            R = submap.optimized_anchor_pose[:3, :3]
            t = submap.optimized_anchor_pose[:3, 3]
            for item in submap.point_cloud_labels:
                world_points = ((R @ item["points"].T).T + t).astype(np.float32)
                labels.append(
                    {
                        "frame_id": item["frame_id"],
                        "label": item["label"],
                        "score": item["score"],
                        "points": world_points,
                    }
                )
        return labels

    def reset(self):
        self._frame_buffer.clear()
        self._frames.clear()
        self._raw_global_poses.clear()
        self._frame_to_submap.clear()
        self.submaps.clear()
        self._frame_count = 0
        self.factor_graph.reset()
        self.retrieval.reset()
        self._pending_futures.clear()

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _load_vggt(self):
        try:
            from vggt.models.vggt import VGGT  # type: ignore

            self._vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(self.cfg.device)
            self._vggt.eval()
            self._vggt_available = True
            if self.cfg.verbose:
                print("[VGGT-SLAM] VGGT-1B loaded from HuggingFace.")
        except Exception:
            self._vggt = None
            self._vggt_available = False
            if self.cfg.verbose:
                print("[VGGT-SLAM] WARNING: VGGT unavailable; running in dummy mode.")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return tensor.to(self.cfg.device, non_blocking=True)

    def _default_intrinsics(self, width: int, height: int) -> np.ndarray:
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = width / 2.0
        K[1, 1] = height / 2.0
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
        return K

    def _run_window_inference(self, tensors: Sequence[torch.Tensor]) -> Dict[str, Sequence[torch.Tensor]]:
        if self._executor is None:
            return self._run_vggt(list(tensors))

        while len(self._pending_futures) >= self.cfg.inference_queue_size:
            oldest = self._pending_futures.pop(0)
            oldest.result()

        future = self._executor.submit(self._run_vggt, list(tensors))
        self._pending_futures.append(future)
        result = future.result()
        if self._pending_futures and self._pending_futures[0] is future:
            self._pending_futures.pop(0)
        return result

    def _run_vggt(self, tensors: List[torch.Tensor]) -> Dict[str, Sequence[torch.Tensor]]:
        N = len(tensors)
        H, W = tensors[0].shape[1], tensors[0].shape[2]

        if self._vggt_available:
            batch = torch.stack(tensors).unsqueeze(0)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.fp16 and "cuda" in self.cfg.device
                else nullcontext()
            )
            with torch.no_grad(), autocast_context:
                raw = self._vggt(batch)

            extrinsics = raw.get("extrinsic", raw.get("extrinsics"))
            intrinsics = raw.get("intrinsic", raw.get("intrinsics"))
            depth = raw.get("depth")
            attn_feats = raw.get("attention_features")
            if attn_feats is None:
                attn_feats = [depth[0, i].mean().reshape(1) for i in range(N)]
            elif isinstance(attn_feats, torch.Tensor):
                attn_feats = [attn_feats[0, i].flatten() for i in range(attn_feats.shape[1])]
            else:
                attn_feats = [feat.flatten() for feat in attn_feats]

            return {
                "extrinsics": extrinsics[0],
                "intrinsics": intrinsics[0],
                "depth": depth[0],
                "attention_features": attn_feats,
            }

        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        for idx in range(N):
            extrinsics[idx, 0, 3] = idx * 0.05
            extrinsics[idx, 1, 3] = np.sin(idx / max(N, 1)) * 0.02
        intrinsics = torch.from_numpy(np.stack([self._default_intrinsics(W, H) for _ in range(N)], axis=0))
        depth = torch.rand(N, H, W, dtype=torch.float32) * 4.0 + 1.0
        attn_feats = [torch.rand(256, dtype=torch.float32) for _ in range(N)]
        return {
            "extrinsics": extrinsics[:, :3, :],
            "intrinsics": intrinsics,
            "depth": depth,
            "attention_features": attn_feats,
        }

    def _create_submap(self) -> Submap:
        frames = list(self._frame_buffer)
        anchor_pose = frames[0]["raw_pose"].astype(np.float32)
        local_poses = [relative_pose(anchor_pose, frame["raw_pose"]) for frame in frames]
        submap = Submap(
            submap_id=len(self.submaps),
            frame_ids=[frame["frame_id"] for frame in frames],
            frames=frames,
            local_poses=local_poses,
            anchor_pose=anchor_pose,
        )
        submap.build_point_cloud()
        self.factor_graph.add_submap(submap)
        self.submaps.append(submap)

        for frame, local_pose in zip(frames, local_poses):
            frame["local_pose"] = local_pose
            self._frame_to_submap[frame["frame_id"]] = submap.submap_id

        self._frame_buffer.clear()
        if self.cfg.verbose:
            print(
                f"[VGGT-SLAM] Submap {submap.submap_id} created "
                f"({len(submap.frame_ids)} frames, {len(submap.point_cloud_local)} local pts)"
            )
        return submap

    def _try_loop_closure(self, current_submap: Submap):
        if current_submap.submap_id == 0:
            return

        query_idx = current_submap.frame_ids[-1]
        candidates = self.retrieval.query(
            query_idx=query_idx,
            k=self.cfg.loop_closure_k,
            min_distance=self.cfg.submap_size * 2,
        )

        for frame_id, score in candidates:
            if score < self.cfg.loop_closure_thresh:
                continue
            candidate_submap_id = self._frame_to_submap.get(frame_id)
            if candidate_submap_id is None or candidate_submap_id == current_submap.submap_id:
                continue
            if abs(candidate_submap_id - current_submap.submap_id) <= 1:
                continue

            candidate_submap = self.submaps[candidate_submap_id]
            init_rel = np.linalg.inv(candidate_submap.optimized_anchor_pose) @ current_submap.optimized_anchor_pose
            current_to_candidate, icp_conf = refine_relative_pose(
                current_submap.point_cloud_local,
                candidate_submap.point_cloud_local,
                np.linalg.inv(init_rel),
            )
            refined_rel = np.linalg.inv(current_to_candidate).astype(np.float32)
            final_score = float(max(score, icp_conf))
            self.factor_graph.add_loop_closure(
                submap_i=candidate_submap_id,
                submap_j=current_submap.submap_id,
                relative_pose=refined_rel,
                score=final_score,
                source_frame_i=frame_id,
                source_frame_j=query_idx,
            )
            if self.cfg.verbose:
                print(
                    "[VGGT-SLAM] Loop closure: "
                    f"submap {candidate_submap_id} ↔ {current_submap.submap_id} "
                    f"(frames {frame_id} ↔ {query_idx}, score={final_score:.3f})"
                )

    def _optimize_submaps(self):
        poses = self.factor_graph.optimize()
        if not poses:
            return
        for submap, pose in zip(self.submaps, poses):
            submap.optimized_anchor_pose = pose.astype(np.float32)
