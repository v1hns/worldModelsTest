"""
Submap containers and dense point-cloud fusion helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Back-project a depth image into camera-frame 3D points."""
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    H, W = depth.shape
    fx = float(intrinsics[0, 0]) if intrinsics[0, 0] != 0 else float(W) / 2.0
    fy = float(intrinsics[1, 1]) if intrinsics[1, 1] != 0 else float(H) / 2.0
    cx = float(intrinsics[0, 2]) if intrinsics[0, 2] != 0 else float(W) / 2.0
    cy = float(intrinsics[1, 2]) if intrinsics[1, 2] != 0 else float(H) / 2.0

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    zs = depth.astype(np.float32)
    mask = np.isfinite(zs) & (zs > 1e-3)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    xs = xs[mask].astype(np.float32)
    ys = ys[mask].astype(np.float32)
    zs = zs[mask]
    x3 = (xs - cx) * zs / max(fx, 1e-6)
    y3 = (ys - cy) * zs / max(fy, 1e-6)
    return np.stack([x3, y3, zs], axis=-1).astype(np.float32)


@dataclass
class Submap:
    submap_id: int
    frame_ids: List[int]
    frames: List[Dict]
    local_poses: List[np.ndarray]
    anchor_pose: np.ndarray
    optimized_anchor_pose: np.ndarray = field(init=False)
    point_cloud_local: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    point_cloud_labels: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.anchor_pose = self.anchor_pose.astype(np.float32)
        self.optimized_anchor_pose = self.anchor_pose.copy()

    def build_point_cloud(self, sample_stride: int = 4):
        """
        Fuse per-frame depth into a local submap point cloud.

        The fused cloud is represented in the anchor-frame coordinate system
        so the factor graph can reposition the entire submap by moving only
        the anchor pose.
        """
        all_pts = []
        all_labels: List[Dict] = []

        for frame_id, frame, local_pose in zip(self.frame_ids, self.frames, self.local_poses):
            depth = frame.get("depth")
            intrinsics = frame.get("intrinsics")
            if depth is None or intrinsics is None:
                continue

            pts_cam = _points_from_depth(depth, intrinsics)
            if pts_cam.size == 0:
                continue

            if sample_stride > 1:
                pts_cam = pts_cam[::sample_stride]
            if pts_cam.size == 0:
                continue

            R = local_pose[:3, :3]
            t = local_pose[:3, 3]
            pts_local = (R @ pts_cam.T).T + t
            all_pts.append(pts_local.astype(np.float32))

            for det in frame.get("detections", []) or []:
                bbox_points = self._extract_detection_points(frame, det, local_pose, sample_stride)
                if bbox_points.size == 0:
                    continue
                all_labels.append(
                    {
                        "frame_id": frame_id,
                        "label": det.get("label", "object"),
                        "score": float(det.get("score", 0.0)),
                        "points": bbox_points,
                    }
                )

        if all_pts:
            self.point_cloud_local = np.concatenate(all_pts, axis=0).astype(np.float32)
        else:
            self.point_cloud_local = np.empty((0, 3), dtype=np.float32)
        self.point_cloud_labels = all_labels

    def _extract_detection_points(
        self,
        frame: Dict,
        detection: Dict,
        local_pose: np.ndarray,
        sample_stride: int,
    ) -> np.ndarray:
        depth = frame.get("depth")
        intrinsics = frame.get("intrinsics")
        bbox = detection.get("bbox")
        if depth is None or intrinsics is None or bbox is None:
            return np.empty((0, 3), dtype=np.float32)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, depth.shape[1])
        y2 = min(y2, depth.shape[0])
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 3), dtype=np.float32)

        crop = depth[y1:y2, x1:x2]
        if crop.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        ys, xs = np.meshgrid(
            np.arange(y1, y2, sample_stride),
            np.arange(x1, x2, sample_stride),
            indexing="ij",
        )
        zs = depth[ys, xs].astype(np.float32)
        mask = np.isfinite(zs) & (zs > 1e-3)
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)

        xs = xs[mask].astype(np.float32)
        ys = ys[mask].astype(np.float32)
        zs = zs[mask]
        fx = float(intrinsics[0, 0]) if intrinsics[0, 0] != 0 else float(depth.shape[1]) / 2.0
        fy = float(intrinsics[1, 1]) if intrinsics[1, 1] != 0 else float(depth.shape[0]) / 2.0
        cx = float(intrinsics[0, 2]) if intrinsics[0, 2] != 0 else float(depth.shape[1]) / 2.0
        cy = float(intrinsics[1, 2]) if intrinsics[1, 2] != 0 else float(depth.shape[0]) / 2.0

        pts_cam = np.stack(
            [
                (xs - cx) * zs / max(fx, 1e-6),
                (ys - cy) * zs / max(fy, 1e-6),
                zs,
            ],
            axis=-1,
        ).astype(np.float32)
        R = local_pose[:3, :3]
        t = local_pose[:3, 3]
        return ((R @ pts_cam.T).T + t).astype(np.float32)

    @property
    def point_cloud(self) -> np.ndarray:
        """Return the fused cloud in world coordinates using the optimized anchor."""
        if self.point_cloud_local.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        R = self.optimized_anchor_pose[:3, :3]
        t = self.optimized_anchor_pose[:3, 3]
        return ((R @ self.point_cloud_local.T).T + t).astype(np.float32)

    def world_pose_for_local(self, local_pose: np.ndarray) -> np.ndarray:
        return (self.optimized_anchor_pose @ local_pose).astype(np.float32)

