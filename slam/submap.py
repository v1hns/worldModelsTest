"""
Submap: a short sequence of frames whose poses are jointly estimated by VGGT
and stored together with a fused point cloud.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Submap:
    submap_id: int
    frames: List[dict]              # list of {"image": ndarray, "tensor": tensor}
    poses: List[np.ndarray]         # 4×4 world-camera poses, one per frame
    point_cloud: Optional[np.ndarray] = field(default=None, init=False)

    def build_point_cloud(self):
        """
        Back-project each frame's depth map into 3-D world points using the
        corresponding pose and a simple pinhole intrinsic model.

        For frames that carry no depth (dummy mode), we skip them silently.
        """
        all_pts = []
        for frame, pose in zip(self.frames, self.poses):
            depth = frame.get("depth")
            if depth is None:
                continue

            H, W = depth.shape
            # Assume normalised intrinsics (fx≈fy≈W/2, cx≈W/2, cy≈H/2)
            fx = fy = float(W) / 2.0
            cx, cy = float(W) / 2.0, float(H) / 2.0

            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            zs = depth.astype(np.float32)
            x3 = (xs - cx) / fx * zs
            y3 = (ys - cy) / fy * zs
            z3 = zs

            pts_cam = np.stack([x3, y3, z3], axis=-1).reshape(-1, 3)  # (H*W, 3)
            # Filter out zero-depth pixels
            valid = pts_cam[:, 2] > 0.01
            pts_cam = pts_cam[valid]

            # Transform to world frame: P_w = R @ P_c + t
            R = pose[:3, :3]
            t = pose[:3, 3]
            pts_world = (R @ pts_cam.T).T + t
            all_pts.append(pts_world)

        if all_pts:
            self.point_cloud = np.concatenate(all_pts, axis=0)
        else:
            self.point_cloud = np.empty((0, 3), dtype=np.float32)

    @property
    def anchor_pose(self) -> np.ndarray:
        """Pose of the first frame in this submap (used as submap origin)."""
        return self.poses[0] if self.poses else np.eye(4, dtype=np.float32)
